"""Exploratory two-cohort transfer using the revised DFA target.

This is restricted to CAUEEG and BrainLat because the ds004504 raw files
needed to re-extract DFA v2 are not present in the current workspace.  It
reports both the full BrainLat dementia-spectrum definition (AD+bvFTD vs HC)
and an AD-vs-HC sensitivity analysis.  Differences in diagnosis definitions
and unavailable BrainLat demographics preclude a causal disease claim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.io_safety import safe_np_load  # noqa: E402


SEED = 20260729
N_BOOTSTRAP = 2000
N_PERMUTATIONS = 1000


def _load(cohort: str) -> dict:
    data = safe_np_load(str(ROOT / f"results/paper02/dfa_v2_{cohort}.npz"))
    eligible = np.asarray(data["eligible"], dtype=bool)
    return {
        "features": np.asarray(data["exponent"], dtype=float)[eligible],
        "label": np.asarray(data["label"], dtype=int)[eligible],
        "diagnosis": np.asarray(data["diagnosis"]).astype(str)[eligible],
    }


def _estimator():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
        ),
    )


def _fit_score(
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> tuple[float, np.ndarray]:
    estimator = _estimator().fit(source_x, source_y)
    score = estimator.decision_function(target_x)
    return float(roc_auc_score(target_y, score)), score


def _inference(
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> dict:
    auc, score = _fit_score(source_x, source_y, target_x, target_y)
    rng = np.random.default_rng(SEED)
    classes = [np.flatnonzero(target_y == value) for value in np.unique(target_y)]
    bootstrap = []
    for _ in range(N_BOOTSTRAP):
        sample = np.concatenate(
            [rng.choice(group, len(group), replace=True) for group in classes]
        )
        bootstrap.append(roc_auc_score(target_y[sample], score[sample]))

    permutation = []
    for _ in range(N_PERMUTATIONS):
        shuffled = rng.permutation(source_y)
        permuted_auc, _ = _fit_score(source_x, shuffled, target_x, target_y)
        permutation.append(permuted_auc)
    permutation = np.asarray(permutation)
    two_sided_extreme = np.abs(permutation - 0.5) >= abs(auc - 0.5)
    return {
        "auc": auc,
        "bootstrap_ci95": [float(x) for x in np.percentile(bootstrap, [2.5, 97.5])],
        "two_sided_source_label_permutation_p": float(
            (1 + two_sided_extreme.sum()) / (N_PERMUTATIONS + 1)
        ),
        "n_source": len(source_y),
        "n_target": len(target_y),
    }


def _both_directions(brainlat: dict, caueeg: dict) -> dict:
    return {
        "BrainLat_to_CAUEEG": _inference(
            brainlat["features"],
            brainlat["label"],
            caueeg["features"],
            caueeg["label"],
        ),
        "CAUEEG_to_BrainLat": _inference(
            caueeg["features"],
            caueeg["label"],
            brainlat["features"],
            brainlat["label"],
        ),
    }


def main() -> None:
    brainlat = _load("brainlat")
    caueeg = _load("caueeg")
    ad_hc = np.isin(brainlat["diagnosis"], ["AD", "HC"])
    output = {
        "meta": {
            "target": "19-channel DFA v2",
            "classifier": (
                "source-fitted median imputation + StandardScaler + fixed "
                "RBF-SVM"
            ),
            "status": "exploratory two-cohort transfer",
            "limitation": (
                "BrainLat age/sex/medication are unavailable and diagnosis "
                "definitions differ; ds004504 DFA v2 is unavailable because "
                "its raw files are absent from the workspace."
            ),
        },
        "dementia_spectrum": _both_directions(brainlat, caueeg),
        "AD_vs_HC_sensitivity": _both_directions(
            {
                key: value[ad_hc]
                for key, value in brainlat.items()
            },
            caueeg,
        ),
    }
    path = ROOT / "results/paper02/dfa_transfer_v2.json"
    path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
