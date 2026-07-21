#!/usr/bin/env python3
"""P1.1 — first-party irrecoverability evidence for the LRTC paper.

Claim under test: the cross-population collapse of a frozen EEG foundation model is
NOT recoverable by post-hoc harmonization, because the frozen embedding already
encodes recording-site identity in a way that survives batch correction.

Arms implemented here:
  (a) Site-probe. Can a probe recover which cohort a subject came from, from the
      frozen REVE embedding alone? Expect >> chance.
  (b) Harmonization. Apply ComBat (site as batch) to the embedding, then re-probe
      site. If site survives ComBat, harmonizing a frozen site-encoding embedding
      does not remove the site axis => the FM collapse is not a recoverable batch
      effect. We run the harder, transductive ComBat (test rows visible to ComBat):
      if site survives even that, the claim is strong.

Arm (c) — DFA features carry ~no site info — is DEFERRED: it needs per-subject DFA
recomputation from raw EEG (not cached). The zero-shot DFA transfer result already
implies low site content; a direct site-probe on DFA features is a cheap follow-up
once the feature matrices are cached.

Run from repo root:
  python scripts/analysis/lrtc_p1_1_irrecoverability.py
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
from src.io_safety import safe_np_load  # noqa: E402  (governance-safe deserialisation)
EMB = ROOT / "data" / "embeddings"
OUT = ROOT / "results" / "cross_population" / "lrtc_irrecoverability_p1_1.json"

# Frozen REVE 19-channel subject embeddings, one file per cohort. All must share the
# same 38912-d token space (ASZED is pooled 512-d and is excluded here).
COHORTS = {
    "W_Greek": "ds004504_3way_reve_19ch_subjects.npz",
    "K_Korean": "caueeg_3way_reve_19ch_subjects.npz",
    "D_Dutch": "tdbrain_mdd_healthy_reve_subjects.npz",
}
SEED = 42
N_PCA = 50


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


def site_probe(Z: np.ndarray, site: np.ndarray) -> dict:
    """Multiclass site decoding, subject-level 5-fold; balanced accuracy vs chance."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
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
    """Transductive ComBat on the PCA space, batch = site, no biological covariate.

    This is the aggressive, harmonization-favorable setting: ComBat sees all rows and
    removes per-site location/scale. If site is still decodable afterward, the site
    axis is not a first/second-moment batch effect a frozen embedding can shed.
    """
    from neuroCombat import neuroCombat
    import pandas as pd

    covars = pd.DataFrame({"batch": site})
    # neuroCombat expects data as (features, samples).
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

    # Reduce the 38912-d embedding to a stable PCA space (fit on pooled data).
    Z = PCA(n_components=N_PCA, random_state=SEED).fit_transform(StandardScaler().fit_transform(X))

    # (a) site-probe on the frozen embedding
    probe_raw = site_probe(Z, site)
    print(f"(a) site-probe (frozen REVE): balanced acc "
          f"{probe_raw['balanced_accuracy_mean']:.3f} (chance {probe_raw['chance']:.3f})")

    # (b) site-probe after transductive ComBat
    combat_note = ""
    try:
        Zc = combat_harmonize(Z, site)
        probe_combat = site_probe(Zc, site)
        print(f"(b) site-probe (after ComBat): balanced acc "
              f"{probe_combat['balanced_accuracy_mean']:.3f}")
    except Exception as e:  # pragma: no cover - environment guard
        probe_combat = None
        combat_note = f"ComBat arm failed: {type(e).__name__}: {e}"
        print("(b)", combat_note)

    result = {
        "meta": {
            "title": "P1.1 irrecoverability — site is decodable from the frozen FM embedding",
            "encoder": "REVE (frozen, 19ch, 38912-d subject-mean)",
            "cohorts": {k: int(len(v)) for k, v in raw.items()},
            "balanced_cap_per_cohort": int(cap),
            "pca_components": N_PCA,
            "seed": SEED,
            "status": "exploratory",
            "reproduce": "python scripts/analysis/lrtc_p1_1_irrecoverability.py",
        },
        "site_probe_frozen": probe_raw,
        "site_probe_after_combat": probe_combat,
        "combat_note": combat_note,
        "arm_c_deferred": (
            "DFA-feature site-probe deferred: needs per-subject DFA recomputation "
            "from raw EEG (not cached). Zero-shot DFA transfer already implies low "
            "site content; direct site-probe is a cheap follow-up once features cache."
        ),
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
