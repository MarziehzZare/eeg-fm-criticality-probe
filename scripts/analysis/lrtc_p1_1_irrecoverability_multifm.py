#!/usr/bin/env python3
"""P1.1 multi-FM extension -- site-decoding for LaBraM, CBraMod, BIOT, BENDR.

lrtc_p1_1_irrecoverability.py's site-decoding probe (0.996 -> 0.994 clean2 rerun)
is REVE-only. This runs the identical site-probe + matched-null-baseline pipeline
for the four other FMs, across all three cohorts (ds004504/W, CAUEEG/K, TDBRAIN/D).

Unlike REVE, none of these four models have TDBRAIN in their pretraining corpus
(results/metrics/fm_pretraining_corpus_audit.json, "Green" verification: LaBraM,
CBraMod, BENDR, BIOT all list TDBRAIN as "no"/OOD) -- so all three cohorts are
usable for these models without the leak caveat that excludes TDBRAIN from the
REVE analysis.

Each model's per-cohort embeddings live in a different pre-existing cache format;
this script's load_* functions reconcile each to a flat {sid: vector} dict before
handing off to the shared site_probe/null-baseline machinery (identical to
lrtc_p1_1_irrecoverability_clean2.py's REVE version).

Run from repo root:
  /opt/anaconda3/envs/neurogenis/bin/python scripts/analysis/lrtc_p1_1_irrecoverability_multifm.py
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
XP = ROOT / "results" / "cross_population"
OUT = XP / "lrtc_irrecoverability_p1_1_multifm.json"

SEED = 42
N_PCA = 50
K_NULL_SEEDS = 20


# ── per-model, per-cohort loaders -> {sid: flat np.ndarray} ─────────────────

def _load_flat_embedding_file(fname: str, key: str = "embedding") -> dict:
    d = safe_np_load(EMB / fname)["data"].item()
    return {sid: np.asarray(v[key]).ravel() for sid, v in d.items()}


def load_labram() -> dict:
    W = _load_flat_embedding_file("ds004504_3way_labram_19ch_subjects.npz")
    K_raw = safe_np_load(EMB / "caueeg_3way_labram_multiepoch_subjects.npz")["data"].item()
    K = {sid: np.asarray(v["mean_embed"]).ravel() for sid, v in K_raw.items()}
    D = _load_flat_embedding_file("tdbrain_multifm_labram_19ch_subjects.npz")
    return {"W_Greek": W, "K_Korean": K, "D_Dutch": D}


def load_cbramod() -> dict:
    W = _load_flat_embedding_file("ds004504_3way_cbramod_19ch_subjects.npz")
    K_raw = safe_np_load(EMB / "caueeg_3way_multiepoch_19ch_all.npz")["data"].item()
    K = {sid: np.asarray(v["embedding"] if isinstance(v, dict) else v).ravel()
         for sid, v in K_raw["cbramod"]["subject_mean_embeds"].items()}
    D = _load_flat_embedding_file("tdbrain_multifm_cbramod_19ch_subjects.npz")
    return {"W_Greek": W, "K_Korean": K, "D_Dutch": D}


def load_biot() -> dict:
    W = _load_flat_embedding_file("ds004504_3way_biot_19ch_subjects.npz")
    K_raw = safe_np_load(EMB / "caueeg_3way_multiepoch_19ch_all.npz")["data"].item()
    K = {sid: np.asarray(v["embedding"] if isinstance(v, dict) else v).ravel()
         for sid, v in K_raw["biot"]["subject_mean_embeds"].items()}
    D = _load_flat_embedding_file("tdbrain_multifm_biot_19ch_subjects.npz")
    return {"W_Greek": W, "K_Korean": K, "D_Dutch": D}


def load_bendr() -> dict:
    W = _load_flat_embedding_file("ds004504_3way_bendr_19ch_subjects.npz")
    K_raw = safe_np_load(XP / "extra_fm_embeds.npz")["data"].item()
    K = {sid: np.asarray(v["BENDR"]).ravel() for sid, v in K_raw.items() if "BENDR" in v}
    D = _load_flat_embedding_file("tdbrain_multifm_bendr_19ch_subjects.npz")
    return {"W_Greek": W, "K_Korean": K, "D_Dutch": D}


LOADERS = {"LaBraM": load_labram, "CBraMod": load_cbramod, "BIOT": load_biot, "BENDR": load_bendr}


# ── shared site-probe machinery (identical to lrtc_p1_1_irrecoverability_clean2.py) ──

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


def run_model(name: str, cohorts: dict) -> dict:
    rng = np.random.default_rng(SEED)
    raw = {k: np.stack(list(v.values())) for k, v in cohorts.items()}
    cap = min(len(v) for v in raw.values())
    print(f"[{name}] cohorts: " + ", ".join(f"{k}={len(v)}" for k, v in raw.items())
          + f"  -> balanced cap={cap}", flush=True)

    Xs, sites = [], []
    for i, (cname, X) in enumerate(raw.items()):
        Xb = balanced_sample(X, cap, rng)
        Xs.append(Xb)
        sites.append(np.full(len(Xb), i))
    X = np.vstack(Xs)
    site = np.concatenate(sites)
    n, d_raw = X.shape

    n_pca = min(N_PCA, n - 1, d_raw)
    Z = PCA(n_components=n_pca, random_state=SEED).fit_transform(StandardScaler().fit_transform(X))
    probe_raw = site_probe(Z, site)
    print(f"  site-probe: balanced acc {probe_raw['balanced_accuracy_mean']:.4f} "
          f"(chance {probe_raw['chance']:.3f})", flush=True)

    null_accs = []
    for seed in range(K_NULL_SEEDS):
        nrng = np.random.default_rng(seed)
        x_null = nrng.standard_normal((n, d_raw))
        z_null = PCA(n_components=n_pca, random_state=seed).fit_transform(
            StandardScaler().fit_transform(x_null)
        )
        null_accs.append(site_probe(z_null, site, fold_seed=seed)["balanced_accuracy_mean"])
    null_mean, null_std = float(np.mean(null_accs)), float(np.std(null_accs))
    excess = probe_raw["balanced_accuracy_mean"] / null_mean if null_mean > 0 else float("inf")
    print(f"  null (K={K_NULL_SEEDS}): {null_mean:.4f} +/- {null_std:.4f}  (excess={excess:.2f}x)", flush=True)

    return {
        "embedding_dim": int(d_raw),
        "pca_components": int(n_pca),
        "cohort_n": {k: int(len(v)) for k, v in raw.items()},
        "balanced_cap_per_cohort": int(cap),
        "site_probe": probe_raw,
        "null_baseline": {"mean": null_mean, "std": null_std, "k_seeds": K_NULL_SEEDS,
                           "excess_over_null": excess},
    }


def main() -> None:
    results = {}
    for name, loader in LOADERS.items():
        cohorts = loader()
        results[name] = run_model(name, cohorts)

    out = {
        "meta": {
            "title": "P1.1 multi-FM site-decoding -- LaBraM/CBraMod/BIOT/BENDR, 3 cohorts (W,K,D)",
            "rationale": (
                "REVE excludes TDBRAIN (it is in REVE's pretraining corpus; REVE paper Appendix B). "
                "These four models are confirmed OOD for TDBRAIN per "
                "results/metrics/fm_pretraining_corpus_audit.json (Green verification), "
                "so all three cohorts are used without caveat."
            ),
            "cohorts": {"W_Greek": "ds004504", "K_Korean": "CAUEEG", "D_Dutch": "TDBRAIN"},
            "seed": SEED,
            "null_seeds": K_NULL_SEEDS,
            "status": "exploratory",
            "reproduce": "python scripts/analysis/lrtc_p1_1_irrecoverability_multifm.py",
        },
        "models": results,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")

    print("\n=== summary ===")
    print(f"{'Model':<10} {'N/site':>7} {'BalAcc':>8} {'Chance':>7} {'Null':>8} {'Excess':>7}")
    for name, r in results.items():
        sp, nb = r["site_probe"], r["null_baseline"]
        print(f"{name:<10} {r['balanced_cap_per_cohort']:>7d} {sp['balanced_accuracy_mean']:>8.4f} "
              f"{sp['chance']:>7.3f} {nb['mean']:>8.4f} {nb['excess_over_null']:>6.2f}x")


if __name__ == "__main__":
    main()
