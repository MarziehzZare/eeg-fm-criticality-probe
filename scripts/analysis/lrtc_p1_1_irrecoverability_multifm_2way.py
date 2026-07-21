#!/usr/bin/env python3
"""P1.1 multi-FM site-decoding on the SAME leak-free 2-cohort binary task as REVE.

lrtc_p1_1_irrecoverability_multifm.py runs the four non-REVE FMs on a 3-way site
task (chance 0.333) that includes TDBRAIN. That mixes a harder 3-class problem
with TDBRAIN's REVE-pretraining-leak status and is not comparable to REVE's clean
2-way probe (lrtc_p1_1_irrecoverability_clean2.py, ds004504 vs CAUEEG, chance
0.500). This script puts LaBraM/CBraMod/BIOT/BENDR on that identical leak-free
binary task so all five FMs share chance 0.500, making the raw-vs-spectral
comparison apples-to-apples. Result: site is decoded near-perfectly by every
model (0.98-1.00), i.e. site dominance is universal, not architecture-graded.

Run from repo root:
  /opt/anaconda3/envs/neurogenis/bin/python scripts/analysis/lrtc_p1_1_irrecoverability_multifm_2way.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
from lrtc_p1_1_irrecoverability_multifm import LOADERS  # noqa: E402  (reuse per-model loaders)

OUT = ROOT / "results" / "cross_population" / "lrtc_irrecoverability_p1_1_multifm_2way.json"
SEED = 42
N_PCA = 50
K_NULL = 20


def site_probe(Z: np.ndarray, site: np.ndarray, fold_seed: int = SEED) -> tuple[float, float]:
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=fold_seed)
    accs = []
    for tr, te in skf.split(Z, site):
        clf.fit(Z[tr], site[tr])
        accs.append(balanced_accuracy_score(site[te], clf.predict(Z[te])))
    return float(np.mean(accs)), float(np.std(accs))


def main() -> None:
    out = {}
    print(f"{'Model':<10}{'n/site':>7}{'balacc':>9}{'null':>8}{'excess':>8}")
    for name, loader in LOADERS.items():
        coh = loader()  # {'W_Greek':.., 'K_Korean':.., 'D_Dutch':..}
        W = np.stack(list(coh["W_Greek"].values()))
        K = np.stack(list(coh["K_Korean"].values()))
        rng = np.random.default_rng(SEED)
        cap = min(len(W), len(K))

        def bal(X):
            return X if len(X) <= cap else X[rng.choice(len(X), cap, replace=False)]

        W, K = bal(W), bal(K)
        X = np.vstack([W, K])
        site = np.concatenate([np.zeros(len(W)), np.ones(len(K))])
        n, d = X.shape
        npca = min(N_PCA, n - 1, d)
        Z = PCA(n_components=npca, random_state=SEED).fit_transform(StandardScaler().fit_transform(X))
        acc, sd = site_probe(Z, site)

        nulls = []
        for s in range(K_NULL):
            nr = np.random.default_rng(s)
            xn = nr.standard_normal((n, d))
            zn = PCA(n_components=npca, random_state=s).fit_transform(StandardScaler().fit_transform(xn))
            nulls.append(site_probe(zn, site, fold_seed=s)[0])
        nm, nsd = float(np.mean(nulls)), float(np.std(nulls))
        out[name] = {"balanced_accuracy": acc, "std": sd, "chance": 0.5, "null_mean": nm,
                     "null_std": nsd, "excess": acc / nm, "n_per_cohort": int(cap),
                     "embedding_dim": int(d), "pca": int(npca)}
        print(f"{name:<10}{cap:>7d}{acc:>9.4f}{nm:>8.4f}{acc/nm:>7.2f}x")

    meta = {"title": "2-way (leak-free) multi-FM site probe: ds004504+CAUEEG only, chance 0.500",
            "rationale": ("Same binary task and chance baseline (0.500) as the REVE clean2 probe, so the "
                          "raw-vs-spectral magnitude ladder is comparable. TDBRAIN excluded for all here."),
            "cohorts": {"W": "ds004504", "K": "CAUEEG"}, "seed": SEED, "null_seeds": K_NULL, "pca": N_PCA,
            "reproduce": "python scripts/analysis/lrtc_p1_1_irrecoverability_multifm_2way.py"}
    OUT.write_text(json.dumps({"meta": meta, "models": out}, indent=2))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
