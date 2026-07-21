#!/usr/bin/env python3
"""P1.1 arm (c) multi-FM extension -- transfer-rescue for LaBraM, CBraMod, BIOT, BENDR.

lrtc_p1_1_transfer_rescue.py answers "does leakage-safe harmonization rescue
cross-population disease transfer of the frozen embedding?" for REVE only (via
results/cross_population/p02_matrices.npz, which stores feature matrices but no
subject IDs, so it can't be reused directly for other models). This script asks
the same question for the four other FMs, using their own cached embeddings and
freshly-built W/K label sets keyed by subject ID:

  W (ds004504 AD/HC, n=65: 36 AD + 29 Control; FTD dropped) -- labels from
    data/processed/alzheimer_processed.pkl's "group" field.
  K (CAUEEG Dementia/Normal, MCI dropped) -- labels from
    data/raw/caueeg-dataset/dementia.json's class_name field, same filter
    fm_lrtc_nonlinear_probe.py uses for the N=770 encoding-probe cohort.

Note: W here is n=65 (all available AD+HC subjects), not the original REVE
analysis's n=53 -- that script's p02_matrices.npz has no stored subject IDs, so
its exact 53-subject subset can't be reconstructed. This is a freshly-built,
slightly larger cohort, not a discrepancy with the original REVE numbers.

Same 3 harmonization regimes (raw / persite_z / combat, all label-free w.r.t.
disease) and the same leakage-safe transfer + bootstrap-CI machinery as the
REVE script, reusing cross_population_criticality.py's helpers directly.

Run from repo root:
  /opt/anaconda3/envs/neurogenis/bin/python scripts/analysis/lrtc_p1_1_transfer_rescue_multifm.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
from src.io_safety import safe_np_load, safe_pickle_load  # noqa: E402
import cross_population_criticality as cpc  # noqa: E402

EMB = ROOT / "data" / "embeddings"
XP = ROOT / "results" / "cross_population"
CAUEEG_ROOT = ROOT / "data" / "raw" / "caueeg-dataset"
OUT = XP / "lrtc_p1_1_transfer_rescue_multifm.json"

SEED = 42
N_PCA = 50


# ── labels ───────────────────────────────────────────────────────────────────

def w_labels() -> dict[str, int]:
    """ds004504 AD/HC binary labels, keyed by subject ID. FTD dropped."""
    d = safe_pickle_load(ROOT / "data/processed/alzheimer_processed.pkl")
    return {sid: (1 if v["group"] == "A" else 0) for sid, v in d.items() if v["group"] in ("A", "C")}


def k_labels() -> dict[str, int]:
    """CAUEEG Dementia/Normal binary labels, keyed by serial. MCI dropped."""
    dem = json.loads((CAUEEG_ROOT / "dementia.json").read_text())
    out = {}
    for sp in ("train_split", "validation_split", "test_split"):
        for it in dem[sp]:
            if it["class_name"] in ("Normal", "Dementia"):
                out[it["serial"]] = 1 if it["class_name"] == "Dementia" else 0
    return out


# ── per-model embedding loaders -> {sid: flat vector} (no TDBRAIN needed here) ──

def _load_flat(fname: str, key: str = "embedding") -> dict:
    d = safe_np_load(EMB / fname)["data"].item()
    return {sid: np.asarray(v[key]).ravel() for sid, v in d.items()}


def load_labram() -> tuple[dict, dict]:
    W = _load_flat("ds004504_3way_labram_19ch_subjects.npz")
    K_raw = safe_np_load(EMB / "caueeg_3way_labram_multiepoch_subjects.npz")["data"].item()
    K = {sid: np.asarray(v["mean_embed"]).ravel() for sid, v in K_raw.items()}
    return W, K


def load_cbramod() -> tuple[dict, dict]:
    W = _load_flat("ds004504_3way_cbramod_19ch_subjects.npz")
    K_raw = safe_np_load(EMB / "caueeg_3way_multiepoch_19ch_all.npz")["data"].item()
    K = {sid: np.asarray(v["embedding"] if isinstance(v, dict) else v).ravel()
         for sid, v in K_raw["cbramod"]["subject_mean_embeds"].items()}
    return W, K


def load_biot() -> tuple[dict, dict]:
    W = _load_flat("ds004504_3way_biot_19ch_subjects.npz")
    K_raw = safe_np_load(EMB / "caueeg_3way_multiepoch_19ch_all.npz")["data"].item()
    K = {sid: np.asarray(v["embedding"] if isinstance(v, dict) else v).ravel()
         for sid, v in K_raw["biot"]["subject_mean_embeds"].items()}
    return W, K


def load_bendr() -> tuple[dict, dict]:
    W = _load_flat("ds004504_3way_bendr_19ch_subjects.npz")
    K_raw = safe_np_load(XP / "extra_fm_embeds.npz")["data"].item()
    K = {sid: np.asarray(v["BENDR"]).ravel() for sid, v in K_raw.items() if "BENDR" in v}
    return W, K


LOADERS = {"LaBraM": load_labram, "CBraMod": load_cbramod, "BIOT": load_biot, "BENDR": load_bendr}


# ── harmonization + transfer machinery (identical logic to REVE's script) ──────

def combat_labelfree(Fw: np.ndarray, Fk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from neuroCombat import neuroCombat
    import pandas as pd

    F = np.vstack([Fw, Fk]).astype(np.float64)
    site = np.array([0] * len(Fw) + [1] * len(Fk))
    covars = pd.DataFrame({"batch": site})
    out = np.asarray(neuroCombat(dat=F.T, covars=covars, batch_col="batch")["data"]).T
    return out[: len(Fw)], out[len(Fw):]


def pooled_pca(Fw: np.ndarray, Fk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pooled = np.vstack([Fw, Fk]).astype(np.float64)
    n_comp = min(N_PCA, pooled.shape[0] - 1, pooled.shape[1])
    Z = PCA(n_components=n_comp, random_state=SEED).fit_transform(StandardScaler().fit_transform(pooled))
    return Z[: len(Fw)], Z[len(Fw):]


def regimes(Fw: np.ndarray, Fk: np.ndarray) -> dict:
    out = {"raw": (Fw, Fk), "persite_z": cpc._persite(Fw, Fk)}
    try:
        out["combat"] = combat_labelfree(Fw, Fk)
    except Exception as e:  # pragma: no cover
        out["combat"] = None
        out["_combat_note"] = f"{type(e).__name__}: {e}"
    return out


def transfer_both(Fw, yw, Fk, yk) -> dict:
    return {"WtoK": cpc._transfer_ci(Fw, yw, Fk, yk), "KtoW": cpc._transfer_ci(Fk, yk, Fw, yw)}


def run_model(name: str, W_emb: dict, K_emb: dict, wl: dict, kl: dict) -> dict:
    w_ids = sorted(set(W_emb) & set(wl))
    k_ids = sorted(set(K_emb) & set(kl))
    Fw = np.stack([W_emb[s] for s in w_ids])
    yw = np.array([wl[s] for s in w_ids])
    Fk = np.stack([K_emb[s] for s in k_ids])
    yk = np.array([kl[s] for s in k_ids])
    print(f"[{name}] W(AD/HC) n={len(yw)} (AD={int(yw.sum())})  "
          f"K(Dem/Normal) n={len(yk)} (Dem={int(yk.sum())})", flush=True)

    Zw, Zk = pooled_pca(Fw, Fk)
    reg = regimes(Zw, Zk)
    note = reg.pop("_combat_note", None)
    results = {}
    for rname, pair in reg.items():
        if pair is None:
            results[rname] = None
            continue
        results[rname] = transfer_both(pair[0], yw, pair[1], yk)
        r = results[rname]
        print(f"  {rname:10s} W->K {r['WtoK'][0]:.3f} CI{r['WtoK'][1:]}  "
              f"K->W {r['KtoW'][0]:.3f} CI{r['KtoW'][1:]}", flush=True)
    if note:
        results["combat_note"] = note

    def cells():
        for rn, r in results.items():
            if isinstance(r, dict) and "WtoK" in r:
                for d in ("WtoK", "KtoW"):
                    yield rn, d, r[d]

    sig = [f"{rn}:{d}={c[0]:.3f} CI[{c[1]:.3f},{c[2]:.3f}]" for rn, d, c in cells() if c[1] > 0.5]
    rescued = len(sig) > 0

    return {
        "cohort_n": {"W_AD_HC": int(len(yw)), "K_Dem_Normal": int(len(yk))},
        "embedding_dim": int(Fw.shape[1]),
        "pca_components": int(Zw.shape[1]),
        "results": results,
        "significant_cells_ci_lb_gt_0.5": sig,
        "rescued_by_any_regime": bool(rescued),
    }


def main() -> None:
    cpc.RNG = np.random.default_rng(SEED)
    cpc.SEED = SEED
    wl, kl = w_labels(), k_labels()
    print(f"W label pool: {len(wl)} (AD/HC)   K label pool: {len(kl)} (Dementia/Normal)")

    out = {
        "meta": {
            "title": "P1.1 arm (c) multi-FM -- transfer-rescue for LaBraM/CBraMod/BIOT/BENDR",
            "question": "Does leakage-safe harmonization rescue cross-population disease "
                        "transfer of these frozen embeddings, the way it fails to for REVE?",
            "task": "binary dementia-vs-control, ds004504 (AD/HC) <-> CAUEEG (Dementia/Normal)",
            "note_on_cohort_size": (
                "W here is all available AD+HC ds004504 subjects (n=65) across each model's "
                "cache, not the original REVE script's n=53 -- p02_matrices.npz stores no "
                "subject IDs so that exact subset can't be reconstructed; this is a freshly "
                "built cohort, not a discrepancy."
            ),
            "harmonization_regimes": {
                "raw": "strict zero-shot, train-only standardization, no target info",
                "persite_z": "per-cohort z by own mean/std (unlabelled target batch)",
                "combat": "neuroCombat batch=site, no diagnosis covariate (unlabelled target batch)",
            },
            "ci": "1000x subject-level bootstrap 95% CI on the target",
            "classifier": "StandardScaler(train) + SVC-RBF (class-weight balanced)",
            "seed": SEED,
            "status": "exploratory",
            "reproduce": "python scripts/analysis/lrtc_p1_1_transfer_rescue_multifm.py",
        },
        "models": {},
    }
    for name, loader in LOADERS.items():
        W_emb, K_emb = loader()
        out["models"][name] = run_model(name, W_emb, K_emb, wl, kl)

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")

    print("\n=== summary (rescued by any leakage-safe regime?) ===")
    for name, r in out["models"].items():
        print(f"{name:<10} rescued={r['rescued_by_any_regime']}  cells={r['significant_cells_ci_lb_gt_0.5'] or 'NONE'}")


if __name__ == "__main__":
    main()
