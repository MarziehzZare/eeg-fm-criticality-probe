"""Control A (re-run against the v2 target): is decoded DFA a shadow of the aperiodic slope?

The legacy paper reported that envelope-DFA and the aperiodic exponent were empirically
orthogonal (r = -0.06 on the 0.5-30 s target) and that residualizing DFA on the slope left
every encoder's recovery unchanged.  That control was computed on the superseded target and
was dropped from the revision.  It is re-established here against the v2 estimand.

The control is now LOAD-BEARING rather than a nicety.  Under the v2 target, BIOT decodes DFA
in CAUEEG (R2 = 0.232) *and* decodes the aperiodic exponent strongly (R2 = 0.606); CBraMod
does likewise at lower magnitude.  If the DFA recovery is merely the aperiodic recovery in
disguise, the single positive temporal-decoding result in the revised manuscript dissolves.

Design (identical readout to paper02_primary_encoding_v2.py so numbers are comparable):
  target A : v2 alpha-envelope DFA exponent (mean over 19 channels)
  target B : the same exponent residualized on the fixed-mode aperiodic exponent
  readout  : 5-fold OOF StandardScaler -> PCA50 -> RidgeCV, everything fit inside the fold

The beta-residualization is fit INSIDE each outer training fold (slope and intercept from
training subjects only, then applied to the held-out subjects), so the control introduces no
target leakage.  A global-residualization variant is reported alongside for comparability with
the legacy control, which residualized on the full sample.

Reading:
  preserved point estimate -> inconsistent with simple leakage from the measured linear
                              aperiodic term, but does not exclude nonlinear dependence,
                              other spectral summaries, or measurement error
  collapsed point estimate -> compatible with linear aperiodic dependence carrying the signal

The pair bootstrap below resamples fixed out-of-fold target/prediction pairs. Because the
training folds overlap and the full pipeline is not refitted, its interval is conditional on
that prediction vector; it is not a full-pipeline generalization interval.

Scope: the aperiodic comparison target exists for 199 CAUEEG subjects and 79 BrainLat
subjects, so this control runs on that intersection, not the full 764. Residualized
decoding is evaluated for CBraMod and BIOT, the two models with replicated aperiodic
decoding, plus the classical positive control.

USAGE:
  python scripts/analysis/paper02_aperiodic_orthogonality_v2.py
STATUS: exploratory diagnostic control for Paper 02 revision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scipy import stats  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

import paper02_primary_encoding_v2 as pe  # noqa: E402  (identical loaders + readout)

SEED = pe.SEED
OUTER_FOLDS = pe.OUTER_FOLDS
N_BOOTSTRAP = 2000


def _residualize(y: np.ndarray, beta: np.ndarray, fit_idx: np.ndarray) -> np.ndarray:
    """Remove the linear beta component from y, with the fit taken on fit_idx only."""
    design = np.column_stack([beta[fit_idx], np.ones(len(fit_idx))])
    coef, *_ = np.linalg.lstsq(design, y[fit_idx], rcond=None)
    return y - (coef[0] * beta + coef[1])


def _oof_residualized(x: np.ndarray, y: np.ndarray, beta: np.ndarray):
    """OOF predictions for the beta-residualized target, residualization fit in-fold.

    Returns (residual_target, prediction) where residual_target[test] is defined by the
    training-fold regression, so scoring never uses a test-fold-derived residual.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    prediction = np.full(len(y), np.nan)
    residual_target = np.full(len(y), np.nan)
    outer = KFold(OUTER_FOLDS, shuffle=True, random_state=SEED)
    for fold, (train, test) in enumerate(outer.split(x)):
        resid = _residualize(y, beta, train)          # fit on train, applied to all
        residual_target[test] = resid[test]
        n_components = max(2, min(50, x.shape[1], len(train) - 2))
        scaler = StandardScaler().fit(x[train])
        train_scaled = scaler.transform(x[train])
        test_scaled = scaler.transform(x[test])
        pca = PCA(n_components=n_components, svd_solver="randomized",
                  random_state=SEED + fold).fit(train_scaled)
        ridge = RidgeCV(alphas=np.logspace(-2, 4, 9), cv=None)
        ridge.fit(pca.transform(train_scaled), resid[train])
        prediction[test] = ridge.predict(pca.transform(test_scaled))
    return residual_target, prediction


def _bootstrap_r2(y: np.ndarray, prediction: np.ndarray) -> list[float]:
    """Conditional pair-bootstrap interval for a fixed OOF prediction vector."""
    rng = np.random.default_rng(SEED)
    scores = []
    idx = np.arange(len(y))
    for _ in range(N_BOOTSTRAP):
        pick = rng.choice(idx, size=len(idx), replace=True)
        if np.std(y[pick]) < 1e-12:
            continue
        scores.append(r2_score(y[pick], prediction[pick]))
    return [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))]


def run(cohort: str) -> dict:
    target = pe._load_target(cohort)
    embeddings = pe.LOADERS[cohort]()
    aperiodic = pe._load_aperiodic_target(cohort)

    eligible = target["eligible"]
    serials = target["serial"][eligible]
    values = target["target"][eligible]
    keep = np.isfinite(values)
    serials, values = serials[keep], values[keep]
    dfa_by_serial = dict(zip(serials, values))

    common = [s for s in serials if s in aperiodic and np.isfinite(aperiodic[s])]
    y = np.array([dfa_by_serial[s] for s in common], dtype=float)
    beta = np.array([aperiodic[s] for s in common], dtype=float)

    pearson_r, pearson_p = stats.pearsonr(y, beta)
    spearman_r, spearman_p = stats.spearmanr(y, beta)
    print(f"[{cohort}] n={len(common)}  corr(DFA_v2, aperiodic): "
          f"pearson r={pearson_r:+.3f} (p={pearson_p:.3g}), "
          f"spearman rho={spearman_r:+.3f}", flush=True)
    print(f"[{cohort}] shared variance r^2 = {pearson_r**2*100:.1f}%", flush=True)

    out = {
        "meta": {
            "cohort": cohort,
            "n": len(common),
            "target": "v2 alpha-envelope DFA (mean 19ch), 2 s to <=10% duration, FIR 8-13 Hz",
            "control": "aperiodic-shadow control re-run against the v2 estimand",
            "residualization": "linear DFA~aperiodic fit inside each outer training fold",
            "readout": "identical to paper02_primary_encoding_v2 (scaler->PCA50->RidgeCV, in-fold)",
            "aperiodic_target_scope": "fixed-mode specparam 1-40 Hz cache; 199 CAUEEG / 79 BrainLat",
            "reading": (
                "A preserved point estimate is inconsistent with simple leakage "
                "from the measured linear aperiodic term, but it does not exclude "
                "nonlinear dependence, other spectral summaries, or measurement "
                "error."
            ),
            "interval_scope": (
                "The subject pair bootstrap conditions on fixed, dependent OOF "
                "predictions from overlapping training folds. It does not refit "
                "residualization, PCA, or RidgeCV and is not a full-pipeline "
                "generalization interval."
            ),
            "status": "exploratory diagnostic control for the Paper 02 revision",
        },
        "target_correlation": {
            "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
            "spearman_rho": float(spearman_r), "spearman_p": float(spearman_p),
            "shared_variance_fraction": float(pearson_r ** 2),
        },
        "models": {},
    }

    for name in ("classical", "BIOT", "CBraMod"):
        table = embeddings.get(name)
        if not table:
            continue
        usable = [s for s in common if s in table]
        if len(usable) < 40:
            print(f"  [skip {name}] only {len(usable)} subjects", flush=True)
            continue
        x = np.vstack([table[s] for s in usable])
        yy = np.array([dfa_by_serial[s] for s in usable], dtype=float)
        bb = np.array([aperiodic[s] for s in usable], dtype=float)

        raw_pred, _, _ = pe._nested_oof_predictions(x, yy)
        raw_r2 = r2_score(yy, raw_pred)
        raw_ci = _bootstrap_r2(yy, raw_pred)

        resid_target, resid_pred = _oof_residualized(x, yy, bb)
        res_r2 = r2_score(resid_target, resid_pred)
        res_ci = _bootstrap_r2(resid_target, resid_pred)

        # global residualization, matching the legacy control's convention
        glob = _residualize(yy, bb, np.arange(len(yy)))
        glob_pred, _, _ = pe._nested_oof_predictions(x, glob)
        glob_r2 = r2_score(glob, glob_pred)

        out["models"][name] = {
            "n": len(usable),
            "r2_raw_dfa": float(raw_r2), "r2_raw_dfa_ci95": raw_ci,
            "r2_residualized_infold": float(res_r2),
            "r2_residualized_infold_ci95": res_ci,
            "r2_residualized_global": float(glob_r2),
            "delta_r2_infold": float(res_r2 - raw_r2),
            "retained_fraction": (float(res_r2 / raw_r2) if raw_r2 > 1e-9 else None),
            "above_descriptive_reference_after_residualization": bool(
                res_r2 >= pe.DESCRIPTIVE_REFERENCE_R2
            ),
        }
        print(f"  {name:10s} n={len(usable):4d} rawR2={raw_r2:+.3f} "
              f"-> resid(in-fold)={res_r2:+.3f} CI[{res_ci[0]:+.3f},{res_ci[1]:+.3f}] "
              f"(global={glob_r2:+.3f}, delta={res_r2-raw_r2:+.3f})", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=["caueeg", "brainlat"])
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/paper02/aperiodic_orthogonality_v2.json")
    args = ap.parse_args()

    results = {}
    for cohort in args.cohorts:
        print("=" * 72, flush=True)
        results[cohort] = run(cohort)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n[orthogonality-v2] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
