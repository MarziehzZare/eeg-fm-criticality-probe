#!/usr/bin/env python3
"""P1.1 addendum — matched random-Gaussian null for the frozen-REVE site probe.

The site probe in lrtc_p1_1_irrecoverability.py recovers cohort identity from
the frozen REVE embedding at 0.996 balanced accuracy (chance 0.333). That
number alone does not rule out a mundane alternative: with N=264 samples
reduced through PCA-50 then fit with logistic regression, could a
comparably-shaped *noise* embedding already decode "site" well above chance
through sheer probe capacity relative to sample size?

This script answers that by running the identical pipeline (StandardScaler ->
PCA-50 -> StandardScaler+LogisticRegression, 5-fold CV, balanced accuracy) on
K=20 independent draws of iid Gaussian noise shaped exactly like the real
pre-PCA REVE embedding (N=264, D=38912), reproducing the site-label pattern
(88 subjects x 3 sites) so the null is matched on both class balance and
raw dimensionality. This mirrors the random-Gaussian null control used by
Lin, Wu & Jung (2026), "The Identity Trap in EEG Foundation Models"
(arXiv:2606.06647), Appendix B.

Run from repo root:
  python scripts/analysis/lrtc_p1_1_null_baseline.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "cross_population" / "lrtc_p1_1_null_baseline.json"

N_PER_SITE = 88  # matches balanced_cap_per_cohort in lrtc_p1_1_irrecoverability.py
N_SITES = 3
D_RAW = 38912  # frozen REVE 19-channel subject-mean embedding dimension
N_PCA = 50
K_SEEDS = 20
REAL_BALANCED_ACCURACY = 0.9962962962962962  # from lrtc_irrecoverability_p1_1.json
CHANCE = 1.0 / N_SITES


def site_probe(Z: np.ndarray, site: np.ndarray, fold_seed: int) -> float:
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=fold_seed)
    accs = []
    for tr, te in skf.split(Z, site):
        clf.fit(Z[tr], site[tr])
        accs.append(balanced_accuracy_score(site[te], clf.predict(Z[te])))
    return float(np.mean(accs))


def main() -> None:
    site = np.concatenate([np.full(N_PER_SITE, i) for i in range(N_SITES)])
    n = len(site)

    null_accs: list[float] = []
    for seed in range(K_SEEDS):
        rng = np.random.default_rng(seed)
        x_null = rng.standard_normal((n, D_RAW))
        z_null = PCA(n_components=N_PCA, random_state=seed).fit_transform(
            StandardScaler().fit_transform(x_null)
        )
        acc = site_probe(z_null, site, fold_seed=seed)
        null_accs.append(acc)
        print(f"seed {seed:2d}: null site-probe balanced accuracy = {acc:.4f}")

    null_mean = float(np.mean(null_accs))
    null_std = float(np.std(null_accs))
    excess = REAL_BALANCED_ACCURACY / null_mean if null_mean > 0 else float("inf")

    result = {
        "meta": {
            "title": "P1.1 addendum - matched random-Gaussian null for the frozen-REVE site probe",
            "method": (
                "K iid Gaussian draws of shape (N=264, D=38912) matching the real "
                "pre-PCA REVE embedding and site-label balance (88/88/88), each run "
                "through the identical StandardScaler->PCA-50->StandardScaler+"
                "LogisticRegression 5-fold pipeline as lrtc_p1_1_irrecoverability.py."
            ),
            "reference": (
                "Lin, Wu & Jung (2026), 'The Identity Trap in EEG Foundation Models', "
                "arXiv:2606.06647, Appendix B (matched random-Gaussian null control)"
            ),
            "n_per_site": N_PER_SITE,
            "n_sites": N_SITES,
            "d_raw": D_RAW,
            "pca_components": N_PCA,
            "k_seeds": K_SEEDS,
            "status": "exploratory",
            "reproduce": "python scripts/analysis/lrtc_p1_1_null_baseline.py",
        },
        "real_balanced_accuracy": REAL_BALANCED_ACCURACY,
        "chance": CHANCE,
        "null_balanced_accuracy_mean": null_mean,
        "null_balanced_accuracy_std": null_std,
        "null_per_seed": null_accs,
        "excess_ratio_real_over_null": excess,
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nreal={REAL_BALANCED_ACCURACY:.4f}  chance={CHANCE:.4f}  "
          f"null={null_mean:.4f}+/-{null_std:.4f}  excess x{excess:.2f}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
