"""Site/cohort decodability of the revised DFA target.

Two questions are kept separate:

1. Can BrainLat acquisition site (Argentina vs Chile) be decoded within one
   cohort?
2. Can cohort (BrainLat vs CAUEEG) be decoded across datasets?

The second contrast is not a pure site effect: cohort, diagnosis definition,
hardware, recruitment, and geography all co-vary.  Both analyses fit
preprocessing inside CV and use label permutations for a finite-sample null.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.io_safety import safe_np_load  # noqa: E402


SEED = 20260729
N_PERMUTATIONS = 1000


def _load(cohort: str) -> dict:
    data = safe_np_load(str(ROOT / f"results/paper02/dfa_v2_{cohort}.npz"))
    eligible = np.asarray(data["eligible"], dtype=bool)
    exponent = np.asarray(data["exponent"], dtype=float)[eligible]
    return {
        "exponent": exponent,
        "mean": np.nanmean(exponent, axis=1, keepdims=True),
        "label": np.asarray(data["label"], dtype=int)[eligible],
        "diagnosis": np.asarray(data["diagnosis"]).astype(str)[eligible],
        "site": np.asarray(data["site"]).astype(str)[eligible],
    }


def _cv_score(features: np.ndarray, outcome: np.ndarray, seed: int) -> float:
    splitter = StratifiedKFold(5, shuffle=True, random_state=seed)
    estimator = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
    )
    prediction = cross_val_predict(estimator, features, outcome, cv=splitter)
    return float(balanced_accuracy_score(outcome, prediction))


def _permutation_test(
    features: np.ndarray,
    outcome: np.ndarray,
    *,
    n_permutations: int = N_PERMUTATIONS,
) -> dict:
    observed = _cv_score(features, outcome, SEED)
    rng = np.random.default_rng(SEED)
    null = np.asarray(
        [
            _cv_score(features, rng.permutation(outcome), SEED + index + 1)
            for index in range(n_permutations)
        ]
    )
    return {
        "balanced_accuracy": observed,
        "chance": float(1 / len(np.unique(outcome))),
        "permutation_p": float((1 + np.sum(null >= observed)) / (len(null) + 1)),
        "null_mean": float(np.mean(null)),
        "null_ci95": [float(x) for x in np.percentile(null, [2.5, 97.5])],
        "n": len(outcome),
        "n_permutations": len(null),
        "preprocessing": (
            "Median imputation and StandardScaler fitted inside each CV "
            "training fold"
        ),
    }


def main() -> None:
    brainlat = _load("brainlat")
    caueeg = _load("caueeg")
    output = {
        "meta": {
            "target": "DFA v2",
            "status": "site/cohort confound sensitivity analysis",
            "warning": (
                "Cross-cohort decodability is not a pure recording-site effect "
                "because recruitment, diagnoses, geography, and hardware co-vary."
            ),
        },
        "brainlat_internal_site": {},
        "cross_cohort": {},
    }

    internal = brainlat["site"] != "unknown"
    for feature_name in ("mean", "exponent"):
        output["brainlat_internal_site"][feature_name] = _permutation_test(
            brainlat[feature_name][internal],
            brainlat["site"][internal],
        )

    # Use equal cohort sizes so sample imbalance cannot drive the classifier.
    rng = np.random.default_rng(SEED)
    cap = min(len(brainlat["label"]), len(caueeg["label"]))
    brain_index = rng.choice(len(brainlat["label"]), cap, replace=False)
    caueeg_index = rng.choice(len(caueeg["label"]), cap, replace=False)
    cohort_outcome = np.r_[np.zeros(cap, dtype=int), np.ones(cap, dtype=int)]
    for feature_name in ("mean", "exponent"):
        features = np.vstack(
            [
                brainlat[feature_name][brain_index],
                caueeg[feature_name][caueeg_index],
            ]
        )
        output["cross_cohort"][feature_name] = _permutation_test(
            features,
            cohort_outcome,
        )

    path = ROOT / "results/paper02/dfa_site_v2.json"
    path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
