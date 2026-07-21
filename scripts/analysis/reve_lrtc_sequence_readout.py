"""C4 — does REVE *compute* LRTC but *discard* it at pooling?

The deployed REVE embedding mean-pools over the patch-time token axis; mean is
permutation-invariant, so it mathematically erases long-range temporal autocorrelation
(LRTC/DFA). Prior probe: the mean-pooled embedding cannot predict a subject's DFA (R^2~0).

This experiment keeps the PER-TOKEN sequence (pre-pool) and asks whether the temporal
trajectory of REVE's token activations preserves the LRTC that pooling throws away.

Per subject (CAUEEG, 240 s continuous, harmonised 19ch/200Hz/CAR/1-45Hz):
  - 8 contiguous 30 s windows -> REVE forward -> pre-pool feat (1, 19*33, 512)
    -> reshape (19, 33, 512) -> mean over channels -> (33, 512)
    -> per-patch L2 norm -> 33-pt trajectory -> concat 8 windows -> ~264-pt FM-token series
  - true_DFA      = alpha-envelope DFA of the input EEG (ground truth LRTC)
  - fm_token_DFA  = DFA of the FM-token-norm trajectory (ORDERED)
  - fm_token_DFA_shuffled = same on a token-shuffled trajectory (control: order destroyed)
  - pooled_embed  = mean over all tokens (the deployed embedding)

Decisive comparisons:
  corr(fm_token_DFA_ORDERED, true_DFA)  vs  corr(SHUFFLED, true_DFA)  vs  ridge(pooled -> true_DFA)
  If ORDERED >> SHUFFLED ~ pooled(~0): REVE's token sequence preserves LRTC that
  mean-pooling destroys (confirms the H2 mechanism + hands a fix: order-preserving readout).

USAGE:
    conda run -n neurogenis python scripts/analysis/reve_lrtc_sequence_readout.py \
        --n-per-class 30 --out results/cross_population/reve_lrtc_sequence_readout.json

STATUS: exploratory mechanistic control for the cross-population-criticality work.
References: Hardstone et al. 2012 (DFA), REVE (arXiv:2510.21585).
"""
from __future__ import annotations
import argparse, json, os, sys, glob
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))
import warnings; warnings.filterwarnings("ignore")
import torch
from scipy.signal import butter, filtfilt, hilbert
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from src.reve_wrapper import REVEWrapper

SF = 200.0
SEED = 42
RNG = np.random.default_rng(0)
WIN_S = 30
N_WIN = 8
COMMON = ["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2",
          "F7","F8","T3","T4","T5","T6","Fz","Cz","Pz"]
CMAP = {c.upper(): c for c in COMMON}
CAUEEG_ROOT = "data/raw/caueeg-dataset 2"


def _norm(n): return n.upper().replace("-AVG", "").replace("-REF", "").replace("EEG ", "").strip()


def _order(names):
    idx = {}
    for i, n in enumerate(names):
        k = _norm(n)
        if k in CMAP:
            idx[CMAP[k]] = i
    return [idx[c] for c in COMMON] if len(idx) == 19 else None


def _car_filt(d):
    d = d - d.mean(0, keepdims=True)
    b, a = butter(4, [1 / (SF / 2), 45 / (SF / 2)], btype="band")
    return filtfilt(b, a, d, axis=1)


def _dfa(x, scales):
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


def true_dfa(data):
    """alpha-envelope DFA averaged over channels (ground-truth LRTC)."""
    scales = np.unique(np.logspace(np.log10(100), np.log10(6000), 16).astype(int))
    ba = butter(4, [8 / (SF / 2), 13 / (SF / 2)], btype="band")
    vals = [_dfa(np.abs(hilbert(filtfilt(ba[0], ba[1], data[ch]))), scales) for ch in range(19)]
    return float(np.nanmean(vals))


def reve_token_series(wrapper, data):
    """Run REVE over N_WIN windows; return (fm_token_norm_trajectory, pooled_embedding)."""
    traj = []; pooled = []
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
        f = feat[0].cpu().numpy()              # (19*npatch, 512)
        npatch = f.shape[0] // 19
        f = f[:npatch * 19].reshape(19, npatch, -1)   # channel-major (per-channel temporal patches)
        per_patch = np.linalg.norm(f.mean(0), axis=-1)  # (npatch,) channel-avg token-norm over time
        traj.append(per_patch)
        pooled.append(f.reshape(-1, f.shape[-1]).mean(0))  # mean-pool over all tokens
        if wrapper.device.type == "mps":
            torch.mps.empty_cache()
    if not traj:
        return None, None
    return np.concatenate(traj), np.mean(pooled, axis=0)


def cv_r2(X, y):
    X = np.atleast_2d(np.asarray(X));
    if X.shape[0] == 1:
        X = X.T
    good = np.isfinite(y)
    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 13)))
    pred = cross_val_predict(pipe, X[good], np.asarray(y)[good],
                             cv=KFold(5, shuffle=True, random_state=0))
    return float(r2_score(np.asarray(y)[good], pred))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=30)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/cross_population/reve_lrtc_sequence_readout.json")
    args = ap.parse_args()
    os.chdir(ROOT)
    import mne

    dem = json.loads(Path(f"{CAUEEG_ROOT}/dementia.json").read_text())
    items = [it for sp in ("train_split", "validation_split", "test_split")
             for it in dem[sp] if it["class_name"] in ("Normal", "Dementia")]
    by = {"Normal": [], "Dementia": []}
    for it in items:
        by[it["class_name"]].append(it["serial"])
    sel = by["Normal"][:args.n_per_class] + by["Dementia"][:args.n_per_class]

    dfa_scales = np.unique(np.logspace(np.log10(8), np.log10(80), 12).astype(int))  # for ~264-pt series
    W = REVEWrapper.load(ch_names=COMMON, n_times=WIN_S * int(SF), sfreq=200,
                         device="mps", batch_size=1)
    true_d, fm_ord, fm_shuf, pooled = [], [], [], []
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
            traj, pemb = reve_token_series(W, d)
            if traj is None or len(traj) < 30:
                continue
            td = true_dfa(d)
            fo = _dfa(traj, dfa_scales)
            shuffled = traj.copy(); RNG.shuffle(shuffled)
            fs = _dfa(shuffled, dfa_scales)
            if not np.isfinite(td) or not np.isfinite(fo):
                continue
            true_d.append(td); fm_ord.append(fo); fm_shuf.append(fs); pooled.append(pemb)
        except Exception as e:
            print("fail", ser, e, flush=True)
        if (i + 1) % 10 == 0:
            print(f"{i+1}/{len(sel)} kept {len(true_d)}", flush=True)
    W.free()

    true_d = np.array(true_d); fm_ord = np.array(fm_ord)
    fm_shuf = np.array(fm_shuf); pooled = np.array(pooled)
    r_ord, p_ord = pearsonr(fm_ord, true_d)
    r_shuf, p_shuf = pearsonr(fm_shuf, true_d)
    out = {
        "meta": {"n": int(len(true_d)), "windows": N_WIN, "win_s": WIN_S,
                 "true_dfa_mean": float(true_d.mean()), "fm_token_dfa_mean": float(fm_ord.mean()),
                 "status": "exploratory mechanistic control (C4)"},
        "sequence_readout_ORDERED": {"pearson_r": float(r_ord), "r2": float(r_ord**2), "p": float(p_ord)},
        "sequence_readout_SHUFFLED": {"pearson_r": float(r_shuf), "r2": float(r_shuf**2), "p": float(p_shuf)},
        "pooled_readout_ridge_R2": cv_r2(pooled, true_d),
        "interpretation": ("If ORDERED >> SHUFFLED ~ pooled(~0): REVE's per-token sequence "
                           "preserves LRTC that mean-pooling destroys (H2 confirmed)."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
