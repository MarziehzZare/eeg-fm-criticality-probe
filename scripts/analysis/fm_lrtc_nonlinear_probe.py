"""P0.1 — Does the "FM is blind to LRTC" result survive NON-LINEAR probes?

The multi-FM encoding probe (`multifm_lrtc_encoding.json`) used a *linear* ridge
decoder and found no EEG FM (REVE/CBraMod/LaBraM/BIOT) encodes the alpha-envelope
DFA/LRTC exponent (CV R^2 ~ 0), while CBraMod/BIOT partly encode the static 1/f
aperiodic slope (0.33-0.40). A reviewer's first objection: "you only showed DFA is
not *linearly* decodable." This script closes that hole by adding three non-linear
probes on the SAME frozen embeddings and targets:

  probes (all 5-fold CV R^2, identical folds):
    ridge_linear_full : StandardScaler -> RidgeCV                (baseline; full dim)
    ridge_pca50       : StandardScaler -> PCA(50) -> RidgeCV      (PCA-preserves control)
    kernelridge_rbf   : StandardScaler -> PCA(50) -> RBF KernelRidge (grid alpha,gamma)
    hist_gboost       : StandardScaler -> PCA(50) -> HistGradientBoosting
    mlp               : StandardScaler -> PCA(50) -> MLP(64), early stopping

  targets (per subject, mean over 19 channels):
    DFA_full  : alpha-envelope DFA exponent, scales 0.5-30 s
    DFA_short : alpha-envelope DFA exponent, scales 0.5-2 s  (inside FM context)
    1/f       : aperiodic exponent (specparam, fixed mode)     [positive control]

CLAIM DEFENDED iff: across ALL probes, DFA_full/DFA_short recovery stays ~0 for every
FM, while 1/f stays recoverable for CBraMod/BIOT (and classical recovers DFA). i.e.
the blindness is to the *temporal-scaling* exponent specifically, not a decoder-power
artifact. `classical` is the positive control (DFA should be recoverable from it).

CAVEAT (stated in output): non-linear probes run on the top-50 PCs of each embedding
(full-dim non-linear regression is degenerate at n~200, d up to 38912). The full-dim
*linear* ridge is reported alongside so the reader sees both. PCA-preservation is
verified by ridge_pca50 vs ridge_linear_full on the 1/f positive control.

USAGE:
    /opt/anaconda3/bin/conda run -n neurogenis python \
        scripts/analysis/fm_lrtc_nonlinear_probe.py --n 200 \
        --out results/cross_population/fm_lrtc_nonlinear_probe.json

STATUS: exploratory (P0.1 hardening for the cross-population-criticality paper).
References: Hardstone et al. 2012 (DFA); Donoghue et al. 2020 (specparam).
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
warnings.filterwarnings("ignore")

from scipy.signal import butter, filtfilt, hilbert, welch  # noqa: E402
from specparam import SpectralModel  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import (HistGradientBoostingRegressor,  # noqa: E402
                              RandomForestRegressor)
from sklearn.kernel_ridge import KernelRidge  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.io_safety import safe_np_load  # noqa: E402  (governance-safe deserialisation)

SF = 200.0
CAUEEG_ROOT = "data/raw/caueeg-dataset"
COMMON = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
          "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]
CMAP = {c.upper(): c for c in COMMON}


def _norm(n: str) -> str:
    return n.upper().replace("-AVG", "").replace("-REF", "").replace("EEG ", "").strip()


def _order(names: list[str]) -> list[int] | None:
    idx: dict[str, int] = {}
    for i, n in enumerate(names):
        k = _norm(n)
        if k in CMAP:
            idx[CMAP[k]] = i
    return [idx[c] for c in COMMON] if len(idx) == 19 else None


def _car_filt(data: np.ndarray) -> np.ndarray:
    data = data - data.mean(0, keepdims=True)
    b, a = butter(4, [1 / (SF / 2), 45 / (SF / 2)], btype="band")
    return filtfilt(b, a, data, axis=1)


def _dfa(x: np.ndarray, scales: np.ndarray, with_r2: bool = False):
    """Detrended fluctuation analysis exponent of a 1-D signal over `scales` (samples)."""
    x = x - x.mean()
    y = np.cumsum(x)
    out = []
    for s in scales:
        s = int(s)
        n = len(y) // s
        if n < 2:
            out.append(np.nan)
            continue
        Y = y[:n * s].reshape(n, s).astype(float)
        t = np.arange(s)
        tm = t.mean()
        tc = t - tm
        den = (tc * tc).sum()
        sl = (Y * tc).sum(1) / den
        ic = Y.mean(1) - sl * tm
        res = Y - (sl[:, None] * t[None, :] + ic[:, None])
        out.append(np.sqrt((res * res).mean(1).mean()))
    out = np.array(out)
    ok = np.isfinite(out) & (out > 0)
    if ok.sum() < 3:
        return (np.nan, np.nan) if with_r2 else np.nan
    ls = np.log(np.asarray(scales, float)[ok])
    lf = np.log(out[ok])
    co = np.polyfit(ls, lf, 1)
    if not with_r2:
        return co[0]
    pred = np.polyval(co, ls)
    r2 = 1 - ((lf - pred) ** 2).sum() / (((lf - lf.mean()) ** 2).sum() + 1e-12)
    return co[0], r2


FULL_SCALES = np.unique(np.logspace(np.log10(100), np.log10(6000), 16).astype(int))
SHORT_SCALES = np.unique(np.logspace(np.log10(100), np.log10(400), 8).astype(int))


def targets_from_signal(data19: np.ndarray) -> tuple[float, float, float, float]:
    """Harmonised [19,T] -> (mean DFA_full, mean DFA_short, mean 1/f, mean DFA fit R^2)."""
    ba = butter(4, [8 / (SF / 2), 13 / (SF / 2)], btype="band")
    dfa_full, dfa_short, exps, r2s = [], [], [], []
    for ch in range(19):
        x = data19[ch]
        f, p = welch(x, fs=SF, nperseg=int(2 * SF))
        m = (f >= 1) & (f <= 40)
        try:
            sm = SpectralModel(peak_width_limits=[1, 8], max_n_peaks=6,
                               aperiodic_mode="fixed", verbose=False)
            sm.fit(f[m], p[m])
            exps.append(sm.get_params("aperiodic")[-1])
        except Exception:
            exps.append(np.nan)
        env = np.abs(hilbert(filtfilt(ba[0], ba[1], x)))
        df, r2 = _dfa(env, FULL_SCALES, with_r2=True)
        dfa_full.append(df)
        r2s.append(r2)
        dfa_short.append(_dfa(env, SHORT_SCALES))
    return (float(np.nanmean(dfa_full)), float(np.nanmean(dfa_short)),
            float(np.nanmean(exps)), float(np.nanmean(r2s)))


def load_embeddings() -> dict[str, dict[str, np.ndarray]]:
    """serial -> flat embedding, for each encoder."""
    reve = safe_np_load("data/embeddings/caueeg_3way_reve_19ch_subjects.npz")["data"].item()
    labram = safe_np_load(
        "data/embeddings/caueeg_3way_labram_multiepoch_subjects.npz")["data"].item()
    # classical, CBraMod and BIOT subject-mean embeddings all come from the SAME
    # multiepoch file (the source the original multi-FM probe used) so the encoding
    # comparison is apples-to-apples; standalone hash-named CBraMod/BIOT files differ.
    multi = safe_np_load(
        "data/embeddings/caueeg_3way_multiepoch_19ch_all.npz")["data"].item()

    def _flat(v):
        return np.asarray(v).ravel()

    def _sme(vv):
        vv = vv.item() if getattr(vv, "shape", None) == () else vv
        return _flat(vv["embedding"] if isinstance(vv, dict) else vv)

    enc: dict[str, dict[str, np.ndarray]] = {"REVE": {}, "CBraMod": {}, "BIOT": {},
                                             "LaBraM": {}, "classical": {}}
    for s, v in reve.items():
        enc["REVE"][s] = _flat(v["embedding"])
    for s, v in labram.items():
        enc["LaBraM"][s] = _flat(v["mean_embed"])
    for src, name in [("classical", "classical"), ("cbramod", "CBraMod"), ("biot", "BIOT")]:
        for s, vv in multi[src]["subject_mean_embeds"].items():
            enc[name][s] = _sme(vv)
    return enc


def build_dataset(n: int, cache: Path):
    """Return serials, per-encoder X matrices, and target dict — with a target cache."""
    import mne

    dem = json.loads(Path(f"{CAUEEG_ROOT}/dementia.json").read_text())
    items = [it for sp in ("train_split", "validation_split", "test_split")
             for it in dem[sp] if it["class_name"] in ("Normal", "Dementia")]
    enc = load_embeddings()
    common_serials = set(enc["REVE"]) & set(enc["CBraMod"]) & set(enc["BIOT"]) \
        & set(enc["LaBraM"]) & set(enc["classical"])

    tcache: dict[str, list[float]] = {}
    if cache.exists():
        tcache = {k: list(v) for k, v in safe_np_load(str(cache))["data"].item().items()}

    serials, y_full, y_short, y_1f, y_r2 = [], [], [], [], []
    n_new = 0
    for it in items:
        if len(serials) >= n:
            break
        s = it["serial"]
        fp = f"{CAUEEG_ROOT}/signal/edf/{s}.edf"
        if s not in common_serials or not os.path.exists(fp):
            continue
        if s in tcache:
            tf, ts, t1, tr = tcache[s]
        else:
            r = mne.io.read_raw_edf(fp, preload=False, verbose="ERROR")
            r.crop(tmax=min(240.0, (r.n_times - 1) / r.info["sfreq"]))
            r.load_data(verbose="ERROR")
            oc = _order(r.ch_names)
            if oc is None:
                continue
            if r.info["sfreq"] != SF:
                r.resample(SF, verbose="ERROR")
            d = r.get_data()[oc][:, :int(240 * SF)]
            if d.shape[1] < int(240 * SF):
                continue
            tf, ts, t1, tr = targets_from_signal(_car_filt(d))
            tcache[s] = [tf, ts, t1, tr]
            n_new += 1
            if n_new % 10 == 0:
                print(f"  ... computed targets for {n_new} new subjects "
                      f"({len(serials) + 1} kept)", flush=True)
        serials.append(s)
        y_full.append(tf)
        y_short.append(ts)
        y_1f.append(t1)
        y_r2.append(tr)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, data=tcache)

    X = {name: np.vstack([enc[name][s] for s in serials]) for name in enc}
    y = {"DFA_full": np.array(y_full), "DFA_short": np.array(y_short),
         "1/f": np.array(y_1f)}
    return serials, X, y, {"dfa_fit_r2_mean": float(np.nanmean(y_r2))}


def _pca_k(d: int, n: int) -> int:
    # PCA runs inside 5-fold CV, so cap by the per-fold train size (~0.8 n), not n.
    n_train = (n * 4) // 5
    return int(max(2, min(50, d, n_train - 2)))


def make_probes(d: int, n: int) -> dict[str, callable]:
    k = _pca_k(d, n)
    # early stopping carves a validation slice from each ~0.8n train fold; only
    # enable it when that slice is big enough, else train for a fixed budget.
    es = ((n * 4) // 5) >= 60
    # Two linear probes (full-dim baseline + PCA control) and three progressively
    # more flexible NON-LINEAR probes. kernel-ridge grid-searches a regularised RBF on
    # whitened PCs; gboost and random-forest are tree ensembles that are scale-free and
    # robust at n~200. Every non-linear probe is validated on the classical positive
    # control (must recover DFA/1f there) before its FM verdict is trusted.
    return {
        "ridge_linear_full": lambda: make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 13))),
        "ridge_pca50": lambda: make_pipeline(
            StandardScaler(), PCA(k, random_state=0), RidgeCV(alphas=np.logspace(-1, 5, 13))),
        "kernelridge_rbf": lambda: make_pipeline(
            StandardScaler(), PCA(k, whiten=True, random_state=0),
            GridSearchCV(KernelRidge(kernel="rbf"),
                         {"alpha": [1.0, 10.0, 100.0, 1000.0],
                          "gamma": [1.0 / (8 * k), 1.0 / (4 * k), 1.0 / (2 * k)]}, cv=3)),
        "hist_gboost": lambda: make_pipeline(
            StandardScaler(), PCA(k, random_state=0),
            HistGradientBoostingRegressor(max_depth=3, max_iter=300,
                                          l2_regularization=1.0,
                                          early_stopping=es, random_state=0)),
        "random_forest": lambda: make_pipeline(
            StandardScaler(), PCA(k, random_state=0),
            RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf=5,
                                  random_state=0, n_jobs=-1)),
    }


def cv_r2(estimator_fn, X: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(y)
    Xo, yo = X[ok], y[ok]
    pred = cross_val_predict(estimator_fn(), Xo, yo,
                             cv=KFold(5, shuffle=True, random_state=0))
    return float(r2_score(yo, pred))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/cross_population/fm_lrtc_nonlinear_probe.json")
    ap.add_argument("--cache", type=Path,
                    default=ROOT / "results/cross_population/p01_targets.npz")
    args = ap.parse_args()
    os.chdir(ROOT)

    print(f"[P0.1] building dataset (n<={args.n}) ...", flush=True)
    serials, X, y, qc = build_dataset(args.n, args.cache)
    n = len(serials)
    print(f"[P0.1] N={n} subjects; encoders={ {k: X[k].shape[1] for k in X} }", flush=True)
    print(f"[P0.1] target QC: DFA_full mean={np.nanmean(y['DFA_full']):.3f} "
          f"std={np.nanstd(y['DFA_full']):.3f}; 1/f mean={np.nanmean(y['1/f']):.3f}; "
          f"DFA fit R^2={qc['dfa_fit_r2_mean']:.3f}", flush=True)

    out = {
        "meta": {
            "title": "P0.1 non-linear probe: is FM LRTC-blindness a decoder artifact?",
            "n": n, "encoders": {k: int(X[k].shape[1]) for k in X},
            "targets": ["DFA_full", "DFA_short", "1/f"],
            "probes": ["ridge_linear_full", "ridge_pca50", "kernelridge_rbf",
                       "hist_gboost", "random_forest"],
            "cv": "5-fold KFold shuffle seed 0, R^2",
            "caveat": ("non-linear probes fit on top-50 PCs (full-dim non-linear is "
                       "degenerate at n~200); ridge_linear_full is full-dim; "
                       "ridge_pca50 vs full verifies PCA preserves the 1/f signal."),
            "target_qc": qc,
            "status": "exploratory (P0.1 hardening)",
        },
        "cv_r2": {},
    }

    for name in ["REVE", "CBraMod", "BIOT", "LaBraM", "classical"]:
        d = X[name].shape[1]
        probes = make_probes(d, n)
        out["cv_r2"][name] = {}
        for tname, yv in y.items():
            out["cv_r2"][name][tname] = {}
            for pname, pfn in probes.items():
                r2 = cv_r2(pfn, X[name], yv)
                out["cv_r2"][name][tname][pname] = round(r2, 4)
            row = out["cv_r2"][name][tname]
            print(f"  {name:9s} {tname:9s} | "
                  + "  ".join(f"{p}={row[p]:+.3f}" for p in probes), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    try:
        from src.runs import mirror_results_file
        mirror_results_file(args.out, stem="fm_lrtc_nonlinear_probe_results")  # no-op unless POC_RUN_DIR set
    except Exception:
        pass
    print(f"[P0.1] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
