"""C4-strengthened — how much LRTC does a *richer* order-preserving readout
recover from REVE's pre-pool token sequence?

Baseline C4 (`reve_lrtc_sequence_readout.py`) used a single crude readout:
the L2-norm of the channel-averaged token vector per patch (norm-of-mean).
It recovered ORDERED r=0.41 / R^2=0.17 of the true input DFA, vs ~0 for the
deployed mean-pooled embedding. That 0.17 is a *loose lower bound* — the
norm-of-mean discards channel identity and collapses 512 dims to one scalar.

This experiment keeps the same frozen REVE forward passes but extracts a
RICHER set of order-preserving 1-D trajectories from the per-token tensor,
then asks how much of the true DFA they recover:

  readouts (per patch, over the ~264-patch sequence):
    norm_of_mean : || mean_ch(token) ||            (baseline C4 readout)
    mean_of_norm : mean_ch || token ||             (keeps per-channel magnitude)
    pc1 / pc2 / pc3 : top-3 PCA score trajectories of the channel-avg matrix

  per-subject features = DFA of each ORDERED trajectory.
  multi-feature CV ridge( features -> true_DFA )   <- the strengthened readout.

Decisive comparisons (all on the SAME subjects / forward passes):
    ridge(ORDERED features)  vs  ridge(SHUFFLED features)  vs  ridge(pooled embed)
  - ORDERED >> SHUFFLED rules out that the extra degrees of freedom merely
    overfit (a shuffled trajectory has no order, so its DFA carries no LRTC).
  - ORDERED >> pooled (~0) reconfirms mean-pooling is what discards the signal.
  - ridge(ORDERED) > baseline single-readout r^2 TIGHTENS the recovery bound.

A learned order-aware probe (1D-conv / temporal attention) is the natural next
step but needs N>=200 to avoid overfitting on this many parameters — deliberately
deferred; these readouts are model-free and robust at N~80.

USAGE:
    conda run -n neurogenis python scripts/analysis/reve_lrtc_richer_readout.py \
        --n-per-class 40 --out results/cross_population/reve_lrtc_richer_readout.json

STATUS: exploratory mechanistic control (C4-strengthened) for the
cross-population-criticality work. Exploratory.
References: Hardstone et al. 2012 (DFA); REVE (arXiv:2510.21585).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
import torch  # noqa: E402
from scipy.signal import butter, filtfilt, hilbert  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import KFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.reve_wrapper import REVEWrapper  # noqa: E402

SF = 200.0
SEED = 42
RNG = np.random.default_rng(0)  # reproducible token-shuffle control
WIN_S = 30
N_WIN = 8
N_PC = 3
COMMON = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
          "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]
CMAP = {c.upper(): c for c in COMMON}
CAUEEG_ROOT = "data/raw/caueeg-dataset 2"
READOUTS = ["norm_of_mean", "mean_of_norm"] + [f"pc{k + 1}" for k in range(N_PC)]


# ── harmonisation + DFA helpers (self-contained, mirrors baseline C4) ──────────

def _norm(n: str) -> str:
    return n.upper().replace("-AVG", "").replace("-REF", "").replace("EEG ", "").strip()


def _order(names: list[str]) -> list[int] | None:
    idx = {}
    for i, n in enumerate(names):
        k = _norm(n)
        if k in CMAP:
            idx[CMAP[k]] = i
    return [idx[c] for c in COMMON] if len(idx) == 19 else None


def _car_filt(d: np.ndarray) -> np.ndarray:
    d = d - d.mean(0, keepdims=True)
    b, a = butter(4, [1 / (SF / 2), 45 / (SF / 2)], btype="band")
    return filtfilt(b, a, d, axis=1)


def _dfa(x: np.ndarray, scales: np.ndarray) -> float:
    x = np.asarray(x, float); x = x - x.mean(); y = np.cumsum(x); out = []
    for s in scales:
        n = len(y) // s
        if n < 2:
            out.append(np.nan); continue
        Y = y[:n * s].reshape(n, s).astype(float)
        t = np.arange(s); tm = t.mean(); tc = t - tm; den = (tc * tc).sum()
        sl = (Y * tc).sum(1) / den; ic = Y.mean(1) - sl * tm
        res = Y - (sl[:, None] * t[None, :] + ic[:, None])
        out.append(np.sqrt((res * res).mean(1).mean()))
    out = np.array(out); ok = np.isfinite(out) & (out > 0)
    if ok.sum() < 3:
        return np.nan
    sc = np.array(scales)[ok]
    return float(np.polyfit(np.log(sc), np.log(out[ok]), 1)[0])


def true_dfa(data: np.ndarray) -> float:
    """Alpha-envelope DFA averaged over channels (ground-truth LRTC)."""
    scales = np.unique(np.logspace(np.log10(100), np.log10(6000), 16).astype(int))
    ba = butter(4, [8 / (SF / 2), 13 / (SF / 2)], btype="band")
    vals = [_dfa(np.abs(hilbert(filtfilt(ba[0], ba[1], data[ch]))), scales)
            for ch in range(19)]
    return float(np.nanmean(vals))


# ── REVE token extraction (frozen, inference only) ─────────────────────────────

def reve_token_tensor(wrapper: REVEWrapper, data: np.ndarray):
    """Run REVE over N_WIN windows; return concatenated per-channel token tensor.

    Returns (tokens, pooled) where
        tokens : (total_patch, 19, 512) channel-major per-patch activations
        pooled : (512,) mean over all tokens (the deployed embedding)
    """
    chunks, pooled = [], []
    pos = wrapper.pos_tensor.expand(1, -1, -1)
    for w in range(N_WIN):
        seg = data[:, w * WIN_S * int(SF):(w + 1) * WIN_S * int(SF)]
        if seg.shape[1] < WIN_S * int(SF):
            break
        batch = torch.FloatTensor(seg[None].astype(np.float32)).to(wrapper.device)
        with torch.no_grad():
            feat = wrapper.model(batch, pos, return_output=True)
            if isinstance(feat, (list, tuple)):
                feat = feat[-1]
        f = feat[0].cpu().numpy()                       # (19*npatch, 512)
        npatch = f.shape[0] // 19
        f = f[:npatch * 19].reshape(19, npatch, -1)     # (19, npatch, 512) channel-major
        chunks.append(np.transpose(f, (1, 0, 2)))       # (npatch, 19, 512)
        pooled.append(f.reshape(-1, f.shape[-1]).mean(0))
        if wrapper.device.type == "mps":
            torch.mps.empty_cache()
    if not chunks:
        return None, None
    return np.concatenate(chunks, axis=0), np.mean(pooled, axis=0)


def readout_trajectories(tokens: np.ndarray) -> dict[str, np.ndarray]:
    """Derive order-preserving 1-D trajectories from (total_patch, 19, 512) tokens."""
    M = tokens.mean(1)                                  # (P, 512) channel-averaged
    traj = {
        "norm_of_mean": np.linalg.norm(M, axis=-1),     # baseline C4 readout
        "mean_of_norm": np.linalg.norm(tokens, axis=-1).mean(1),
    }
    # top-N_PC principal-component score trajectories of the channel-avg sequence
    Mc = M - M.mean(0, keepdims=True)
    sd = Mc.std(0, keepdims=True); sd[sd == 0] = 1.0
    Mz = Mc / sd
    try:
        _, _, vt = np.linalg.svd(Mz, full_matrices=False)
        scores = Mz @ vt[:N_PC].T                        # (P, N_PC)
        for k in range(N_PC):
            traj[f"pc{k + 1}"] = scores[:, k]
    except np.linalg.LinAlgError:
        for k in range(N_PC):
            traj[f"pc{k + 1}"] = np.full(M.shape[0], np.nan)
    return traj


# ── recovery stats ─────────────────────────────────────────────────────────────

def _ridge_cv_predict(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 13)))
    return cross_val_predict(pipe, X, y, cv=KFold(5, shuffle=True, random_state=SEED))


def _bootstrap_r2(y: np.ndarray, pred: np.ndarray, n_boot: int = 2000) -> list[float]:
    rng = np.random.default_rng(SEED)
    n = len(y); vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.ptp(y[idx]) == 0:
            continue
        vals.append(r2_score(y[idx], pred[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


def _ridge_recovery(X: np.ndarray, y: np.ndarray) -> dict:
    """Out-of-sample (5-fold CV) ridge R^2 with bootstrap 95% CI."""
    pred = _ridge_cv_predict(np.atleast_2d(X).reshape(len(y), -1), y)
    return {"cv_r2": float(r2_score(y, pred)),
            "cv_r2_boot95": _bootstrap_r2(y, pred),
            "n_features": int(np.atleast_2d(X).reshape(len(y), -1).shape[1])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=40)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/cross_population/reve_lrtc_richer_readout.json")
    args = ap.parse_args()
    os.chdir(ROOT)
    import mne  # noqa: PLC0415

    dem = json.loads(Path(f"{CAUEEG_ROOT}/dementia.json").read_text())
    items = [it for sp in ("train_split", "validation_split", "test_split")
             for it in dem[sp] if it["class_name"] in ("Normal", "Dementia")]
    by: dict[str, list] = {"Normal": [], "Dementia": []}
    for it in items:
        by[it["class_name"]].append(it["serial"])
    sel = by["Normal"][:args.n_per_class] + by["Dementia"][:args.n_per_class]

    # scales for the ~264-pt FM-token series (8 windows x ~33 patches)
    dfa_scales = np.unique(np.logspace(np.log10(8), np.log10(80), 12).astype(int))

    W = REVEWrapper.load(ch_names=COMMON, n_times=WIN_S * int(SF), sfreq=200,
                         device="mps", batch_size=1)

    true_d: list[float] = []
    feat_ord: dict[str, list[float]] = {r: [] for r in READOUTS}
    feat_shuf: dict[str, list[float]] = {r: [] for r in READOUTS}
    pooled_rows: list[np.ndarray] = []

    for i, ser in enumerate(sel):
        fp = f"{CAUEEG_ROOT}/signal/edf/{ser}.edf"
        if not os.path.exists(fp):
            continue
        try:
            r = mne.io.read_raw_edf(fp, preload=False, verbose="ERROR")
            r.crop(tmax=min(WIN_S * N_WIN, (r.n_times - 1) / r.info["sfreq"]))
            r.load_data(verbose="ERROR")
            oc = _order(r.ch_names)
            if oc is None:
                continue
            if r.info["sfreq"] != SF:
                r.resample(SF, verbose="ERROR")
            d = _car_filt(r.get_data()[oc][:, :WIN_S * N_WIN * int(SF)])
            if d.shape[1] < WIN_S * N_WIN * int(SF):
                continue
            tokens, pemb = reve_token_tensor(W, d)
            if tokens is None or tokens.shape[0] < 30:
                continue
            td = true_dfa(d)
            if not np.isfinite(td):
                continue
            traj = readout_trajectories(tokens)
            row_ord, row_shuf = {}, {}
            for rname in READOUTS:
                t = traj[rname]
                row_ord[rname] = _dfa(t, dfa_scales)
                ts = t.copy(); RNG.shuffle(ts)
                row_shuf[rname] = _dfa(ts, dfa_scales)
            # keep subject only if the baseline readout DFA is finite
            if not np.isfinite(row_ord["norm_of_mean"]):
                continue
            true_d.append(td)
            for rname in READOUTS:
                feat_ord[rname].append(row_ord[rname])
                feat_shuf[rname].append(row_shuf[rname])
            pooled_rows.append(pemb)
        except Exception as e:  # noqa: BLE001
            print("fail", ser, e, flush=True)
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(sel)} kept {len(true_d)}", flush=True)
    W.free()

    y = np.array(true_d)
    n = len(y)
    Xo = np.column_stack([np.array(feat_ord[r]) for r in READOUTS])  # (n, n_readout)
    Xs = np.column_stack([np.array(feat_shuf[r]) for r in READOUTS])
    pooled = np.array(pooled_rows)

    # impute NaN features (rare PC degeneracies) with column means to keep ridge valid
    for X in (Xo, Xs):
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(~np.isfinite(X))
        X[inds] = np.take(col_mean, inds[1])

    # per-readout single-trajectory recovery (ordered vs shuffled)
    per_readout = {}
    for j, rname in enumerate(READOUTS):
        ro, po = pearsonr(Xo[:, j], y)
        rs, ps = pearsonr(Xs[:, j], y)
        per_readout[rname] = {
            "ordered": {"pearson_r": float(ro), "r2": float(ro ** 2), "p": float(po)},
            "shuffled": {"pearson_r": float(rs), "r2": float(rs ** 2), "p": float(ps)},
        }

    # CV-ridge recovery — all out-of-sample, so single-feature vs multi-feature
    # vs pooled are directly comparable (unlike the in-sample Pearson r above).
    j0 = READOUTS.index("norm_of_mean")
    ridge = {
        "norm_of_mean_ALONE_ordered": _ridge_recovery(Xo[:, j0], y),
        "norm_of_mean_ALONE_shuffled": _ridge_recovery(Xs[:, j0], y),
        "ORDERED_multifeature": _ridge_recovery(Xo, y),
        "SHUFFLED_multifeature": _ridge_recovery(Xs, y),
        "pooled_embedding": {**_ridge_recovery(StandardScaler().fit_transform(pooled), y),
                             "dim": int(pooled.shape[1])},
    }

    # persist per-subject features so this analysis is re-runnable WITHOUT re-running
    # REVE (plain float arrays — safe to np.load with no allow_pickle).
    npz_path = args.out.with_suffix(".features.npz")
    np.savez(npz_path, true_dfa=y, X_ordered=Xo, X_shuffled=Xs, pooled=pooled,
             readouts=np.array(READOUTS))

    out = {
        "meta": {"n": int(n), "windows": N_WIN, "win_s": WIN_S, "n_pc": N_PC,
                 "readouts": READOUTS, "true_dfa_mean": float(y.mean()),
                 "baseline_c4_ref": "reve_lrtc_sequence_readout.json (norm_of_mean only)",
                 "features_npz": npz_path.name,
                 "status": "exploratory mechanistic control (C4-strengthened)"},
        "per_readout_single_trajectory": per_readout,
        "ridge_recovery_cv": ridge,
        "interpretation": (
            "All ridge_recovery_cv numbers are out-of-sample (5-fold CV) and directly "
            "comparable. Test: does a RICHER order-preserving readout recover MORE LRTC "
            "than the single baseline norm_of_mean? Compare ORDERED_multifeature vs "
            "norm_of_mean_ALONE_ordered. SHUFFLED variants ~0 confirm order-dependence "
            "(no order = no LRTC = no overfitting); pooled ~0 reconfirms mean-pooling "
            "discards the signal. If multifeature <= norm-alone, the extra readouts (PCs) "
            "carry no LRTC and the recovery ceiling is real (objective-bound, H3), not a "
            "readout-crudeness artifact."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
