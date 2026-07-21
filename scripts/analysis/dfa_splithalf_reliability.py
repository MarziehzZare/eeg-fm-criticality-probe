"""Split-half reliability of the alpha-envelope DFA_full exponent on CAUEEG.

Addresses reviewer point #6: "narrow target range caps R^2" is not technically
right (R^2 is scale-invariant); what actually bounds achievable R^2 is
measurement reliability. This computes DFA_full independently on the first and
second half of each subject's 240s recording (0-120s vs 120-240s) and reports
the cross-subject R^2 of predicting one half from the other -- an empirical
reliability ceiling any probe (however good) is bound by, since both halves
are noisy realizations of the same underlying per-subject quantity.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import mne  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import KFold, cross_val_predict  # noqa: E402

import scripts.analysis.fm_lrtc_nonlinear_probe as p01  # noqa: E402
from src.io_safety import safe_np_load  # noqa: E402

mne.set_log_level("ERROR")
SF = p01.SF
CACHE = ROOT / "results/cross_population/p01_targets_N1187.npz"
OUT = ROOT / "results/cross_population/dfa_splithalf_reliability.json"


def dfa_full_of_half(data19: np.ndarray) -> float:
    """Mean-over-channels DFA_full exponent of a [19, T] harmonised signal."""
    from scipy.signal import hilbert, filtfilt, butter
    ba = butter(4, [8 / (SF / 2), 13 / (SF / 2)], btype="band")
    vals = []
    for ch in range(19):
        env = np.abs(hilbert(filtfilt(ba[0], ba[1], data19[ch])))
        df = p01._dfa(env, p01.FULL_SCALES)
        vals.append(df)
    return float(np.nanmean(vals))


def main() -> None:
    tcache = {k: list(v) for k, v in safe_np_load(str(CACHE))["data"].item().items()}
    serials = list(tcache.keys())
    os.chdir(ROOT)

    half1, half2, kept = [], [], []
    n_done = 0
    for s in serials:
        fp = f"{p01.CAUEEG_ROOT}/signal/edf/{s}.edf"
        if not os.path.exists(fp):
            continue
        r = mne.io.read_raw_edf(fp, preload=False, verbose="ERROR")
        r.crop(tmax=min(240.0, (r.n_times - 1) / r.info["sfreq"]))
        r.load_data(verbose="ERROR")
        oc = p01._order(r.ch_names)
        if oc is None:
            continue
        if r.info["sfreq"] != SF:
            r.resample(SF, verbose="ERROR")
        d = r.get_data()[oc][:, :int(240 * SF)]
        if d.shape[1] < int(240 * SF):
            continue
        d = p01._car_filt(d)
        mid = d.shape[1] // 2
        try:
            df1 = dfa_full_of_half(d[:, :mid])
            df2 = dfa_full_of_half(d[:, mid:])
        except Exception:
            continue
        if not (np.isfinite(df1) and np.isfinite(df2)):
            continue
        half1.append(df1)
        half2.append(df2)
        kept.append(s)
        n_done += 1
        if n_done % 50 == 0:
            print(f"  ... {n_done} subjects", flush=True)

    half1 = np.array(half1)
    half2 = np.array(half2)
    n = len(kept)
    print(f"N={n} subjects with both halves computed", flush=True)

    r_pearson = float(np.corrcoef(half1, half2)[0, 1])
    # cross-validated R^2 of predicting half2 from half1 (linear, 5-fold) --
    # the empirical ceiling any probe faces when the target itself has this
    # much within-subject (across-half) noise.
    pred = cross_val_predict(LinearRegression(), half1.reshape(-1, 1), half2,
                              cv=KFold(5, shuffle=True, random_state=0))
    r2_ceiling = float(r2_score(half2, pred))
    spearman_brown = float(2 * r_pearson / (1 + r_pearson))

    out = {
        "meta": {
            "title": "Split-half reliability of alpha-envelope DFA_full, CAUEEG",
            "n": n,
            "method": ("DFA_full computed independently on 0-120s and 120-240s "
                       "of each subject's recording; r_pearson and 5-fold "
                       "cross-validated R^2 of predicting half2 from half1 "
                       "report the empirical reliability ceiling."),
            "status": "exploratory (reviewer-requested reliability check)",
        },
        "half1_mean": float(half1.mean()), "half1_std": float(half1.std()),
        "half2_mean": float(half2.mean()), "half2_std": float(half2.std()),
        "r_pearson_half1_half2": r_pearson,
        "r2_ceiling_cv_linear": r2_ceiling,
        "spearman_brown_corrected_r": spearman_brown,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"r_pearson={r_pearson:.3f}  R2_ceiling(cv)={r2_ceiling:.3f}  "
          f"Spearman-Brown r={spearman_brown:.3f}", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
