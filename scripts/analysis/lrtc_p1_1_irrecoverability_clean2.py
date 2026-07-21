#!/usr/bin/env python3
"""P1.1 clean rerun — site-decoding + ComBat, restricted to pretraining-clean cohorts.

lrtc_p1_1_irrecoverability.py's 3-cohort site probe (W_Greek, K_Korean, D_Dutch)
includes TDBRAIN, which the REVE paper (El Ouahidi et al., NeurIPS 2025, Appendix B)
lists in REVE's own pretraining corpus. Any REVE-embedding
result on TDBRAIN cannot distinguish "REVE encodes site identity generally" from
"REVE memorized these specific recordings." This script reruns the identical
site-probe + null-baseline + ComBat pipeline restricted to the two cohorts
confirmed absent from REVE's pretraining corpus (ds004504/Greek, CAUEEG/Korean),
so the headline site-dominance claim rests on uncontaminated data only.

Run from repo root:
  /opt/anaconda3/envs/neurogenis/bin/python scripts/analysis/lrtc_p1_1_irrecoverability_clean2.py
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
from src.io_safety import safe_np_load  # noqa: E402

EMB = ROOT / "data" / "embeddings"
OUT = ROOT / "results" / "cross_population" / "lrtc_irrecoverability_p1_1_clean2.json"

COHORTS = {
    "W_Greek": "ds004504_3way_reve_19ch_subjects.npz",
    "K_Korean": "caueeg_3way_reve_19ch_subjects.npz",
}
SEED = 42
N_PCA = 50
K_NULL_SEEDS = 20


def _emb(sub: dict) -> np.ndarray:
    for k in ("embedding", "reve_embedding"):
        if k in sub:
            return np.asarray(sub[k], dtype=np.float64)
    raise KeyError(f"no embedding key in {list(sub.keys())}")


def load_embeddings(fname: str) -> np.ndarray:
    d = safe_np_load(EMB / fname)["data"].item()
    return np.stack([_emb(v) for v in d.values()])


def balanced_sample(X: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    if len(X) <= cap:
        return X
    idx = rng.choice(len(X), size=cap, replace=False)
    return X[idx]


def site_probe(Z: np.ndarray, site: np.ndarray, fold_seed: int = SEED) -> dict:
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=fold_seed)
    accs = []
    for tr, te in skf.split(Z, site):
        clf.fit(Z[tr], site[tr])
        accs.append(balanced_accuracy_score(site[te], clf.predict(Z[te])))
    n_sites = len(np.unique(site))
    return {
        "balanced_accuracy_mean": float(np.mean(accs)),
        "balanced_accuracy_std": float(np.std(accs)),
        "chance": float(1.0 / n_sites),
        "n_sites": int(n_sites),
        "folds": [float(a) for a in accs],
    }


def combat_harmonize(Z: np.ndarray, site: np.ndarray) -> np.ndarray:
    from neuroCombat import neuroCombat
    import pandas as pd

    covars = pd.DataFrame({"batch": site})
    out = neuroCombat(dat=Z.T, covars=covars, batch_col="batch")["data"]
    return np.asarray(out).T


def main() -> None:
    rng = np.random.default_rng(SEED)
    raw = {name: load_embeddings(f) for name, f in COHORTS.items()}
    cap = min(len(v) for v in raw.values())
    print(f"Loaded cohorts: " + ", ".join(f"{k}={len(v)}" for k, v in raw.items())
          + f"  -> balanced cap={cap}")

    Xs, sites = [], []
    for i, (name, X) in enumerate(raw.items()):
        Xb = balanced_sample(X, cap, rng)
        Xs.append(Xb)
        sites.append(np.full(len(Xb), i))
    X = np.vstack(Xs)
    site = np.concatenate(sites)
    n, d_raw = X.shape

    Z = PCA(n_components=N_PCA, random_state=SEED).fit_transform(StandardScaler().fit_transform(X))

    # (a) site-probe on the frozen embedding
    probe_raw = site_probe(Z, site)
    print(f"(a) site-probe (frozen REVE, W+K only): balanced acc "
          f"{probe_raw['balanced_accuracy_mean']:.4f} (chance {probe_raw['chance']:.3f})")

    # (a2) matched random-Gaussian null, same N/D/pipeline
    null_accs = []
    for seed in range(K_NULL_SEEDS):
        nrng = np.random.default_rng(seed)
        x_null = nrng.standard_normal((n, d_raw))
        z_null = PCA(n_components=N_PCA, random_state=seed).fit_transform(
            StandardScaler().fit_transform(x_null)
        )
        acc = site_probe(z_null, site, fold_seed=seed)["balanced_accuracy_mean"]
        null_accs.append(acc)
    null_mean, null_std = float(np.mean(null_accs)), float(np.std(null_accs))
    excess = probe_raw["balanced_accuracy_mean"] / null_mean if null_mean > 0 else float("inf")
    print(f"(a2) matched null (K={K_NULL_SEEDS} seeds): {null_mean:.4f} +/- {null_std:.4f}  "
          f"(real/null excess = {excess:.2f}x)")

    # (b) site-probe after transductive ComBat
    combat_note = ""
    try:
        Zc = combat_harmonize(Z, site)
        probe_combat = site_probe(Zc, site)
        print(f"(b) site-probe (after ComBat): balanced acc "
              f"{probe_combat['balanced_accuracy_mean']:.4f}")
    except Exception as e:  # pragma: no cover - environment guard
        probe_combat = None
        combat_note = f"ComBat arm failed: {type(e).__name__}: {e}"
        print("(b)", combat_note)

    result = {
        "meta": {
            "title": "P1.1 clean rerun -- site-decoding restricted to pretraining-clean cohorts",
            "rationale": (
                "TDBRAIN is in REVE's pretraining "
                "corpus (REVE paper Appendix B), so the original 3-cohort (W,K,D) site "
                "probe cannot rule out memorization on the D_Dutch leg. This rerun uses "
                "only ds004504 and CAUEEG, both confirmed absent from REVE pretraining."
            ),
            "encoder": "REVE (frozen, 19ch, 38912-d subject-mean)",
            "cohorts": {k: int(len(v)) for k, v in raw.items()},
            "balanced_cap_per_cohort": int(cap),
            "pca_components": N_PCA,
            "seed": SEED,
            "null_seeds": K_NULL_SEEDS,
            "status": "exploratory",
            "reproduce": "python scripts/analysis/lrtc_p1_1_irrecoverability_clean2.py",
        },
        "site_probe_frozen": probe_raw,
        "null_baseline": {"mean": null_mean, "std": null_std, "k_seeds": K_NULL_SEEDS,
                           "excess_over_null": excess},
        "site_probe_after_combat": probe_combat,
        "combat_note": combat_note,
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
