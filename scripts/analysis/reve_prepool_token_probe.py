"""Build the REVE pre-pool sequence cache used by the Paper-02 v2 order probe.

For v2 reproduction, invoke this script with ``--extract-only``.  The cache
contains channel-averaged pre-pool token sequences and subject identifiers;
the canonical matched-realization analysis is implemented separately in
``paper02_prepool_order_permutation.py``.

The historical exploratory analysis below is retained to preserve the original
cache-building implementation, but its seed-level paired test is superseded and
is not evidence that chronological order isolates LRTC.  Chronological order
also preserves autocorrelation, non-stationarity, arousal drift, artifact
clustering, and recording-position effects.

Usage:
    python scripts/analysis/reve_prepool_token_probe.py \
        --n-per-class 150 --extract-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import mne  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import reve_lrtc_richer_readout as rr  # reuse reve_token_tensor, true_dfa, _order, _car_filt  # noqa: E402
from src.io_safety import safe_np_load  # noqa: E402
from src.seed import set_seed  # noqa: E402  (F3 governance: seed the training loop)

mne.set_log_level("ERROR")
SF = rr.SF
COMMON = rr.COMMON
CAUEEG_ROOT = "data/raw/caueeg-dataset"          # correct path (rr module has the stale one)
P0 = 256                                          # fixed token-sequence length (truncate)
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ── extraction ────────────────────────────────────────────────────────────────
def extract(n_per_class: int, cache: Path):
    if cache.exists():
        d = safe_np_load(str(cache))
        return (d["seqs"], d["pooled"], d["dfa"], d["age"], d["label"], list(d["serials"]))
    dem = json.loads(Path(f"{CAUEEG_ROOT}/dementia.json").read_text())
    items = [it for sp in ("train_split", "validation_split", "test_split")
             for it in dem[sp] if it["class_name"] in ("Normal", "Dementia")]
    by = {"Normal": [], "Dementia": []}
    for it in items:
        by[it["class_name"]].append(it)
    sel = by["Normal"][:n_per_class] + by["Dementia"][:n_per_class]

    W = rr.REVEWrapper.load(ch_names=COMMON, n_times=rr.WIN_S * int(SF), sfreq=200)
    seqs, pooled, dfa, age, label, serials = [], [], [], [], [], []
    n_done = 0
    for it in sel:
        ser = it["serial"]
        fp = f"{CAUEEG_ROOT}/signal/edf/{ser}.edf"
        if not os.path.exists(fp):
            continue
        try:
            r = mne.io.read_raw_edf(fp, preload=False, verbose="ERROR")
            r.crop(tmax=min(240.0, (r.n_times - 1) / r.info["sfreq"]))
            r.load_data(verbose="ERROR")
            oc = rr._order(r.ch_names)
            if oc is None:
                continue
            if r.info["sfreq"] != SF:
                r.resample(SF, verbose="ERROR")
            data = r.get_data()[oc][:, :int(240 * SF)]
            if data.shape[1] < int(240 * SF):
                continue
            data = rr._car_filt(data)
            tokens, pemb = rr.reve_token_tensor(W, data)     # (P,19,512), (512,)
            if tokens is None or tokens.shape[0] < P0:
                continue
            seq = tokens.mean(1)[:P0]                          # (P0, 512) channel-avg
            true = rr.true_dfa(data)
        except Exception as e:
            print(f"  [skip {ser}] {type(e).__name__}: {e}", flush=True)
            continue
        seqs.append(seq.astype(np.float16))
        pooled.append(np.asarray(pemb, np.float32))
        dfa.append(float(true)); age.append(float(it["age"]))
        label.append(1 if it["class_name"] == "Dementia" else 0)
        serials.append(ser)
        n_done += 1
        if n_done % 20 == 0:
            print(f"  ... extracted {n_done} subjects", flush=True)
    try:
        W.free()
    except Exception:
        pass
    seqs = np.stack(seqs); pooled = np.stack(pooled)
    dfa = np.array(dfa); age = np.array(age); label = np.array(label)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, seqs=seqs, pooled=pooled, dfa=dfa, age=age, label=label,
             serials=np.array(serials))
    return seqs, pooled, dfa, age, label, serials


# ── learned order-aware probe (1D-CNN over the patch axis) ──────────────────────
class ConvProbe(nn.Module):
    def __init__(self, d: int = 512, h: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d, h, 5, padding=2), nn.ReLU(),
            nn.Conv1d(h, h, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(h, 1))

    def forward(self, x):                                     # x: (B, d, P)
        return self.net(x).squeeze(-1)


def _train_fold(Xtr, ytr, Xva, yva, epochs=200, patience=20, seed=0):
    torch.manual_seed(seed)                                   # vary CNN weight init across seeds
    m = ConvProbe(d=Xtr.shape[1]).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.MSELoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xva_t = torch.tensor(Xva, dtype=torch.float32).to(DEV)
    yva_t = torch.tensor(yva, dtype=torch.float32).to(DEV)
    best, best_state, bad = 1e9, None, 0
    n = len(Xtr_t); bs = 32
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        m.train()
        for i in range(0, n, bs):
            idx = rng.permutation(n)[i:i + bs] if i == 0 else slice(i, i + bs)
            xb = Xtr_t[idx].to(DEV); yb = ytr_t[idx].to(DEV)
            opt.zero_grad(); loss = lossf(m(xb), yb); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = lossf(m(Xva_t), yva_t).item()
        if vl < best - 1e-4:
            best, best_state, bad = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        m.load_state_dict(best_state)
    return m


def cnn_cv_r2(seqs: np.ndarray, y: np.ndarray, shuffle_order: bool, seed: int | None = None) -> float:
    """5-fold OOS CV R^2 of a 1D-CNN predicting y from the (N,P,512) token sequences.
    shuffle_order=True permutes the patch axis per subject (destroys temporal order).
    seed=None reproduces the original run exactly; an int varies the shuffle permutation,
    fold split, and CNN init/batch order together (for repeated-seed hardening)."""
    shuf_seed = 123 if seed is None else 123 + seed
    fold_seed = 42 if seed is None else 42 + seed
    train_seed = 0 if seed is None else seed
    N = len(y)
    X = seqs.astype(np.float32).copy()                        # (N, P, 512)
    if shuffle_order:
        rng = np.random.default_rng(shuf_seed)
        for i in range(N):
            X[i] = X[i][rng.permutation(X.shape[1])]
    X = np.transpose(X, (0, 2, 1))                            # (N, 512, P) for Conv1d
    pred = np.zeros(N)
    for tr, te in KFold(5, shuffle=True, random_state=fold_seed).split(X):
        va = tr[:max(8, len(tr) // 6)]; tr2 = tr[max(8, len(tr) // 6):]
        mu = X[tr2].mean((0, 2), keepdims=True); sd = X[tr2].std((0, 2), keepdims=True) + 1e-6
        ym, ys = y[tr2].mean(), y[tr2].std() + 1e-6
        Xn = (X - mu) / sd
        m = _train_fold(Xn[tr2], (y[tr2] - ym) / ys, Xn[va], (y[va] - ym) / ys, seed=train_seed)
        m.eval()
        with torch.no_grad():
            p = m(torch.tensor(Xn[te], dtype=torch.float32).to(DEV)).cpu().numpy()
        pred[te] = p * ys + ym
    return float(r2_score(y, pred))


def ridge_cv_r2(X: np.ndarray, y: np.ndarray) -> float:
    from sklearn.model_selection import cross_val_predict
    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 13)))
    pred = cross_val_predict(pipe, X, y, cv=KFold(5, shuffle=True, random_state=42))
    return float(r2_score(y, pred))


def residualize_on_age(y: np.ndarray, age: np.ndarray) -> np.ndarray:
    A = np.column_stack([age, np.ones_like(age)])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def load_classical(serials):
    multi = safe_np_load("data/embeddings/caueeg_3way_multiepoch_19ch_all.npz")["data"].item()
    sme = multi["classical"]["subject_mean_embeds"]
    out = []
    for s in serials:
        v = sme[s]; v = v.item() if getattr(v, "shape", None) == () else v
        out.append(np.asarray(v["embedding"] if isinstance(v, dict) else v).ravel())
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=150)
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--cache", type=Path,
                    default=ROOT / "results/cross_population/reve_prepool_tokens.npz")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/cross_population/reve_prepool_token_probe.json")
    args = ap.parse_args()
    os.chdir(ROOT)
    set_seed(0)

    print(f"[prepool] extracting REVE pre-pool token sequences (n_per_class={args.n_per_class}) ...",
          flush=True)
    seqs, pooled, dfa, age, label, serials = extract(args.n_per_class, args.cache)
    n = len(dfa)
    print(f"[prepool] N={n}; seq shape={seqs.shape}; DFA mean={dfa.mean():.3f} std={dfa.std():.3f}; "
          f"age mean={age.mean():.1f}; device={DEV}", flush=True)
    if args.extract_only:
        return

    dfa_resid = residualize_on_age(dfa, age)
    classical = load_classical(serials)
    r_age_dfa = float(np.corrcoef(age, dfa)[0, 1])
    print(f"[prepool] corr(age, DFA)={r_age_dfa:+.3f}; probing raw DFA and age-partialed DFA", flush=True)

    out = {"meta": {"title": "REVE pre-pool token-sequence learned probe (ordered vs shuffled)",
                    "n": n, "seq_len": int(seqs.shape[1]), "dim": int(seqs.shape[2]),
                    "corr_age_dfa": round(r_age_dfa, 4), "device": str(DEV),
                    "reading": "ORDERED>>SHUFFLED and >pooled => REVE encodes LRTC (pooling hides it); "
                               "ORDERED~SHUFFLED~0 while classical recovers => genuinely under-encoded.",
                    "status": "exploratory (council capacity-vs-readout test)"},
           "cv_r2": {}}
    for tgt_name, y in [("DFA", dfa), ("DFA_age_partialed", dfa_resid)]:
        row = {
            "cnn_ordered": round(cnn_cv_r2(seqs, y, shuffle_order=False), 4),
            "cnn_shuffled": round(cnn_cv_r2(seqs, y, shuffle_order=True), 4),
            "pooled_ridge": round(ridge_cv_r2(pooled, y), 4),
            "classical_ridge": round(ridge_cv_r2(classical, y), 4),
        }
        out["cv_r2"][tgt_name] = row
        print(f"  {tgt_name:18s} | ordered={row['cnn_ordered']:+.3f}  shuffled={row['cnn_shuffled']:+.3f}"
              f"  pooled={row['pooled_ridge']:+.3f}  classical={row['classical_ridge']:+.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    try:
        from src.runs import mirror_results_file
        mirror_results_file(args.out, stem="reve_prepool_token_probe_results")  # no-op unless POC_RUN_DIR set
    except Exception:
        pass
    print(f"[prepool] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
