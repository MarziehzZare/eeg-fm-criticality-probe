"""Cross-population criticality transfer — does scale-free temporal dynamics (LRTC/DFA)
cross the population gap where foundation models collapse?

ANALYSIS: train a disease-vs-control classifier on one population, test on another, for
robust *montage-tolerant* criticality features (1/f aperiodic exponent + alpha-envelope DFA),
vs REVE (foundation model) and classical band-power baselines. Avalanche metrics are
deliberately EXCLUDED (power-law xmin/threshold fitting is too acquisition-sensitive for
cross-dataset comparison).

Populations (binary, disease vs control):
  - Western: ds004504 (Greek), AD vs HC (raw .set, 500 Hz)
  - Korean : CAUEEG, Dementia vs Normal (raw EDF, 200 Hz; MCI dropped)

Harmonisation (identical for both, the acquisition-confound control):
  19ch common order (T3/T4 nomenclature matches both datasets), resample 200 Hz,
  common average reference, 1-45 Hz band-pass, first 240 s continuous.

Three questions:
  1. Transfer  — train-one / test-other AUROC + bootstrap 95% CI, both directions.
  2. Rescue    — does label-free per-site standardisation (poor-man's ComBat, no
                 empirical-Bayes) recover the collapsed baselines? (batch-effect test)
  3. Encoding  — can REVE/classical embeddings *predict* the DFA/1f scalar (ridge CV R^2)?
                 If REVE cannot predict DFA, the FM does not encode the invariant signal.

USAGE:
    conda run -n neurogenis python scripts/analysis/cross_population_criticality.py \
        --out results/cross_population/cross_population_criticality_transfer.json

OUTPUT FILE:
    JSON with within/cross-population AUROC (+CI), harmonised transfer, DFA QC
    (exponent distribution + log-log fit R^2), and the FM-encoding probe R^2.

STATUS: exploratory (2 populations, single-split transfer with test-set bootstrap).
    treated as exploratory pending repeated splits + a 3rd population.

References:
  Hardstone et al. 2012 (DFA/LRTC), Donoghue et al. 2020 (specparam/aperiodic),
  Peng et al. 1994 (DFA algorithm). Datasets: Miltiadous et al. 2023 (ds004504),
  CAUEEG (Kim et al. 2023).
"""
from __future__ import annotations
import argparse, json, os, sys, glob
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))
from src.io_safety import safe_pickle_load, safe_np_load  # safe deserialisation (governance F1/F2)

import warnings; warnings.filterwarnings("ignore")
from scipy.signal import welch, butter, filtfilt, hilbert
from specparam import SpectralModel
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, r2_score

SF = 200.0
SEED = 42
RNG = np.random.default_rng(0)
COMMON = ["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2",
          "F7","F8","T3","T4","T5","T6","Fz","Cz","Pz"]
CMAP = {c.upper(): c for c in COMMON}
CAUEEG_ROOT = "data/raw/caueeg-dataset"
ALZ_PKL = "data/processed/alzheimer_processed.pkl"


def _norm(n: str) -> str:
    return n.upper().replace("-AVG", "").replace("-REF", "").replace("EEG ", "").strip()


def _order(names):
    idx = {}
    for i, n in enumerate(names):
        k = _norm(n)
        if k in CMAP:
            idx[CMAP[k]] = i
    return [idx[c] for c in COMMON] if len(idx) == 19 else None


def _car_filt(data):
    data = data - data.mean(0, keepdims=True)            # common average reference
    b, a = butter(4, [1 / (SF / 2), 45 / (SF / 2)], btype="band")
    return filtfilt(b, a, data, axis=1)


def _dfa(x, scales, with_r2=False):
    x = x - x.mean(); y = np.cumsum(x); out = []
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
        return (np.nan, np.nan) if with_r2 else np.nan
    sc = np.array(scales)[ok]; lf = np.log(out[ok]); ls = np.log(sc)
    co = np.polyfit(ls, lf, 1)
    if not with_r2:
        return co[0]
    pred = np.polyval(co, ls)
    r2 = 1 - ((lf - pred) ** 2).sum() / (((lf - lf.mean()) ** 2).sum() + 1e-12)
    return co[0], r2


def extract(data, with_r2=False):
    """data [19,T] harmonised -> per-channel 1/f exponent, offset, DFA (and DFA R^2)."""
    scales = np.unique(np.logspace(np.log10(100), np.log10(6000), 16).astype(int))
    ba = butter(4, [8 / (SF / 2), 13 / (SF / 2)], btype="band")
    exps = np.full(19, np.nan); offs = np.full(19, np.nan)
    dfas = np.full(19, np.nan); r2s = np.full(19, np.nan)
    for ch in range(19):
        x = data[ch]
        f, p = welch(x, fs=SF, nperseg=int(2 * SF)); m = (f >= 1) & (f <= 40)
        try:
            sm = SpectralModel(peak_width_limits=[1, 8], max_n_peaks=6,
                               aperiodic_mode="fixed", verbose=False)
            sm.fit(f[m], p[m]); ap = sm.get_params("aperiodic")
            offs[ch] = ap[0]; exps[ch] = ap[-1]
        except Exception:
            pass
        env = np.abs(hilbert(filtfilt(ba[0], ba[1], x)))
        if with_r2:
            dfas[ch], r2s[ch] = _dfa(env, scales, with_r2=True)
        else:
            dfas[ch] = _dfa(env, scales)
    return (exps, offs, dfas, r2s) if with_r2 else (exps, offs, dfas)


def load_western():
    """ds004504 raw .set -> harmonised criticality features + REVE/classical embeddings."""
    import mne
    reve = safe_np_load("data/embeddings/ds004504_3way_reve_19ch_subjects.npz")["data"].item()
    cls = safe_np_load("data/embeddings/ds004504_3way_classical_subjects.npz")["data"].item()
    X, y, R, C, r2 = [], [], [], [], []
    for fp in sorted(glob.glob("data/raw/ds004504/sub-*/eeg/*task-eyesclosed_eeg.set")):
        sid = os.path.basename(fp).split("_")[0]
        if sid not in reve or reve[sid]["label"] not in (0, 1):
            continue
        r = mne.io.read_raw_eeglab(fp, preload=True, verbose="ERROR")
        oc = _order(r.ch_names)
        if oc is None:
            continue
        if r.info["sfreq"] != SF:
            r.resample(SF, verbose="ERROR")
        d = r.get_data()[oc][:, :int(240 * SF)]
        if d.shape[1] < int(240 * SF):
            continue
        e, o, fa, rr = extract(_car_filt(d), with_r2=True)
        X.append(np.concatenate([e, o, fa])); y.append(1 if reve[sid]["label"] == 0 else 0)
        R.append(np.asarray(reve[sid]["embedding"]).ravel())
        C.append(np.asarray(cls[sid]["embedding"]).ravel()); r2.append(np.nanmean(rr))
    return (np.array(X), np.array(y), np.array(R), np.array(C), float(np.nanmean(r2)))


def load_korean(per_class=None):
    """CAUEEG raw EDF -> harmonised criticality features + REVE/classical embeddings."""
    import mne
    dem = json.loads(Path(f"{CAUEEG_ROOT}/dementia.json").read_text())
    items = [it for sp in ("train_split", "validation_split", "test_split")
             for it in dem[sp] if it["class_name"] in ("Normal", "Dementia")]
    reve = safe_np_load("data/embeddings/caueeg_3way_reve_19ch_subjects.npz")["data"].item()
    cls = {s: (v.item() if getattr(v, "shape", None) == () else v) for s, v in
           safe_np_load("data/embeddings/caueeg_3way_multiepoch_19ch_all.npz")["data"]
           .item()["classical"]["subject_mean_embeds"].items()}
    if per_class:
        nrm = [it for it in items if it["class_name"] == "Normal"][:per_class]
        dm = [it for it in items if it["class_name"] == "Dementia"][:per_class]
        items = nrm + dm
    X, y, R, C, r2 = [], [], [], [], []
    for it in items:
        ser = it["serial"]; fp = f"{CAUEEG_ROOT}/signal/edf/{ser}.edf"
        if not os.path.exists(fp) or ser not in reve or ser not in cls:
            continue
        r = mne.io.read_raw_edf(fp, preload=False, verbose="ERROR")
        r.crop(tmax=min(240.0, (r.n_times - 1) / r.info["sfreq"])); r.load_data(verbose="ERROR")
        oc = _order(r.ch_names)
        if oc is None:
            continue
        if r.info["sfreq"] != SF:
            r.resample(SF, verbose="ERROR")
        d = r.get_data()[oc][:, :int(240 * SF)]
        if d.shape[1] < int(240 * SF):
            continue
        e, o, fa, rr = extract(_car_filt(d), with_r2=True)
        X.append(np.concatenate([e, o, fa])); y.append(1 if it["class_name"] == "Dementia" else 0)
        R.append(np.asarray(reve[ser]["embedding"]).ravel())
        C.append(np.asarray(cls[ser]["embedding"]).ravel()); r2.append(np.nanmean(rr))
    return (np.array(X), np.array(y), np.array(R), np.array(C), float(np.nanmean(r2)))


def _impute(M):
    M = M.copy().astype(float); col = np.nanmean(M, 0)
    idx = np.where(np.isnan(M)); M[idx] = np.take(col, idx[1]); return M


def _clf():
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True,
                                               class_weight="balanced", random_state=SEED))


def _within(X, y, seeds=(42, 7, 1337)):
    a = []
    for s in seeds:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            a.append(roc_auc_score(y[te], _clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]))
    return float(np.mean(a))


def _transfer_ci(Xtr, ytr, Xte, yte, nb=1000):
    c = _clf().fit(Xtr, ytr); p = c.predict_proba(Xte)[:, 1]; auc = roc_auc_score(yte, p)
    n = len(yte); bs = []
    for _ in range(nb):
        idx = RNG.integers(0, n, n)
        if len(np.unique(yte[idx])) > 1:
            bs.append(roc_auc_score(yte[idx], p[idx]))
    return [float(auc), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def _persite(Xw, Xk):
    z = lambda M: (M - M.mean(0)) / (M.std(0) + 1e-8)
    return z(Xw), z(Xk)


def _cv_r2(X, y):
    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 13)))
    pred = cross_val_predict(pipe, X, y, cv=KFold(5, shuffle=True, random_state=0))
    return float(r2_score(y, pred))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/cross_population/cross_population_criticality_transfer.json")
    ap.add_argument("--korean-per-class", type=int, default=None,
                    help="subsample Korean N per class (default: all Normal+Dementia)")
    args = ap.parse_args()
    os.chdir(ROOT)

    Xw, yw, Rw, Cw, r2w = load_western()
    Xk, yk, Rk, Ck, r2k = load_korean(args.korean_per_class)
    Xw, Xk = _impute(Xw), _impute(Xk)
    fam = {"1/f": slice(0, 38), "DFA": slice(38, 57), "1/f+DFA": slice(0, 57)}

    out = {"meta": {"western": "ds004504 AD/HC", "korean": "CAUEEG Dem/Normal",
                    "n_western": int(len(yw)), "n_korean": int(len(yk)),
                    "harmonisation": "19ch/200Hz/CAR/1-45Hz/240s", "status": "exploratory"},
           "transfer": {}, "dfa_qc": {"fit_r2_western": r2w, "fit_r2_korean": r2k}}

    for name, sl in fam.items():
        out["transfer"][name] = {
            "within_W": _within(Xw[:, sl], yw), "within_K": _within(Xk[:, sl], yk),
            "raw_WtoK": _transfer_ci(Xw[:, sl], yw, Xk[:, sl], yk),
            "raw_KtoW": _transfer_ci(Xk[:, sl], yk, Xw[:, sl], yw)}
        hw, hk = _persite(Xw[:, sl], Xk[:, sl])
        out["transfer"][name]["harm_WtoK"] = _transfer_ci(hw, yw, hk, yk)
        out["transfer"][name]["harm_KtoW"] = _transfer_ci(hk, yk, hw, yw)
    for name, (A, B) in {"REVE": (Rw, Rk), "classical": (Cw, Ck)}.items():
        out["transfer"][name] = {
            "within_W": _within(A, yw), "within_K": _within(B, yk),
            "raw_WtoK": _transfer_ci(A, yw, B, yk), "raw_KtoW": _transfer_ci(B, yk, A, yw)}
        hw, hk = _persite(A, B)
        out["transfer"][name]["harm_WtoK"] = _transfer_ci(hw, yw, hk, yk)
        out["transfer"][name]["harm_KtoW"] = _transfer_ci(hk, yk, hw, yw)

    # DFA exponent distribution
    dw, dk = Xw[:, 38:57], Xk[:, 38:57]
    out["dfa_qc"].update({
        "exp_mean_W": float(dw.mean()), "exp_mean_K": float(dk.mean()),
        "exp_frac_in_0.5_1.0_W": float(np.mean((dw >= 0.5) & (dw <= 1.0))),
        "exp_frac_in_0.5_1.0_K": float(np.mean((dk >= 0.5) & (dk <= 1.0)))})

    # FM-encoding probe: can embeddings predict mean DFA / mean 1/f?
    out["fm_encoding_probe"] = {}
    for pop, (R, C, Xfeat) in [("Western", (Rw, Cw, Xw)), ("Korean", (Rk, Ck, Xk))]:
        meanDFA = Xfeat[:, 38:57].mean(1); mean1f = Xfeat[:, 0:19].mean(1)
        out["fm_encoding_probe"][pop] = {
            "DFA_from_REVE": _cv_r2(R, meanDFA), "1f_from_REVE": _cv_r2(R, mean1f),
            "DFA_from_classical": _cv_r2(C, meanDFA), "1f_from_classical": _cv_r2(C, mean1f)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    print(json.dumps(out["transfer"], indent=2))


if __name__ == "__main__":
    main()
