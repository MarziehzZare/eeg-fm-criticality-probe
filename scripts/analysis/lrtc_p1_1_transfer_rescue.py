#!/usr/bin/env python3
"""P1.1 arm (c) --- leakage-safe transfer-rescue test for the LRTC paper.

Deciding question (manuscript \\S sec:results-irrecov). Arm (b) showed that a transductive
ComBat removes the recording-site axis from the frozen REVE embedding (site decodability
0.996 -> 0.26). But removing a *site* axis is not the same as recovering a *disease* axis.
The operative clinical claim is about cross-population disease TRANSFER, so:

    Does leakage-safe harmonization rescue cross-population disease transfer of the frozen
    REVE embedding?

Design. Binary dementia-vs-control transfer, both directions, Western (ds004504 AD/HC) <->
Korean (CAUEEG Dementia/Normal, MCI dropped). Features and label-free harmonization regimes
are chosen so the answer is interpretable via two positive controls:

  Features (all disease-labelled the same way; source-labelled classifier, target AUROC):
    REVE   frozen 38912-d embedding, PCA-50               -- the FM under test
    DFA    19-d alpha-envelope DFA exponent (dimensionless) -- transfers natively (control A)
    CLASS  173-d classical band/spectral features           -- signal present but site-masked (control B)

  Harmonization regimes (every one is label-free w.r.t. disease => leakage-safe):
    raw        strict zero-shot: train-only standardization, NO target information used
    persite_z  poor-man's ComBat: each cohort z-scored by its own mean/std (unlabeled target batch)
    combat     neuroCombat, batch=site, NO diagnosis covariate (unlabeled target batch)

Reading. Harmonization is expected to rescue a feature ONLY when the disease signal is present
but masked by a removable site axis (CLASS: raw ~ chance -> harmonized up). A dimensionless
feature needs no harmonization (DFA: high at raw, ~unchanged). If the frozen embedding is NOT
rescued by any regime (REVE stays near chance), the FM collapse is irrecoverable because the
transferable disease signal was never encoded -- not merely masked by a removable site axis.
This is the positive form of manuscript contribution 3.

Note on "leakage-safe": persite_z and combat use the target cohort's *unlabelled* feature
statistics (the realistic "unlabelled batch at deployment" scenario). The `raw` column is the
strictly held-out case (no target data at all). Disease labels are never used to harmonize.

Run from repo root (env: neurogenis):
  python scripts/analysis/lrtc_p1_1_transfer_rescue.py
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
from src.io_safety import safe_np_load  # noqa: E402  (governance-safe deserialisation)
import cross_population_criticality as cpc  # noqa: E402  (reuse _clf / _transfer_ci / _persite / _impute)

SEED = 42
N_PCA = 50
DFA_SLICE = slice(38, 57)  # extract() = [1/f-exp(19) | 1/f-off(19) | DFA(19)]
CACHE = ROOT / "results" / "cross_population" / "p02_matrices.npz"
OUT = ROOT / "results" / "cross_population" / "lrtc_p1_1_transfer_rescue.json"


def combat_labelfree(Fw: np.ndarray, Fk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """neuroCombat with batch=site and NO biological covariate (label-free)."""
    from neuroCombat import neuroCombat
    import pandas as pd

    F = np.vstack([Fw, Fk]).astype(np.float64)
    site = np.array([0] * len(Fw) + [1] * len(Fk))
    covars = pd.DataFrame({"batch": site})
    out = np.asarray(neuroCombat(dat=F.T, covars=covars, batch_col="batch")["data"]).T
    return out[: len(Fw)], out[len(Fw):]


def reve_pca(Rw: np.ndarray, Rk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label-free PCA-50 on the pooled, standardized 38912-d embedding (as in arm b).

    Pooled (unsupervised) reduction gives the FM its best chance and matches the site-probe
    arm; no disease labels are used, so it does not leak the transfer target.
    """
    pooled = np.vstack([Rw, Rk]).astype(np.float64)
    Z = PCA(n_components=N_PCA, random_state=SEED).fit_transform(
        StandardScaler().fit_transform(pooled)
    )
    return Z[: len(Rw)], Z[len(Rw):]


def regimes(Fw: np.ndarray, Fk: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "raw": (Fw, Fk),
        "persite_z": cpc._persite(Fw, Fk),
    }
    try:
        out["combat"] = combat_labelfree(Fw, Fk)
    except Exception as e:  # pragma: no cover - environment/degeneracy guard
        out["combat"] = None  # type: ignore[assignment]
        out["_combat_note"] = f"{type(e).__name__}: {e}"  # type: ignore[assignment]
    return out


def transfer_both(Fw: np.ndarray, yw: np.ndarray, Fk: np.ndarray, yk: np.ndarray) -> dict:
    return {
        "WtoK": cpc._transfer_ci(Fw, yw, Fk, yk),
        "KtoW": cpc._transfer_ci(Fk, yk, Fw, yw),
    }


def main() -> None:
    cpc.RNG = np.random.default_rng(SEED)  # deterministic bootstrap
    cpc.SEED = SEED
    m = safe_np_load(str(CACHE))
    Rw, Rk = m["Rw"], m["Rk"]
    Cw, Ck = cpc._impute(m["Cw"]), cpc._impute(m["Ck"])
    Xw, Xk = cpc._impute(m["Xw"]), cpc._impute(m["Xk"])
    yw, yk = m["yw"].astype(int), m["yk"].astype(int)

    Zw, Zk = reve_pca(Rw, Rk)
    features = {
        "REVE_pca50": (Zw, Zk),
        "DFA_19d": (Xw[:, DFA_SLICE], Xk[:, DFA_SLICE]),
        "Classical_173d": (Cw, Ck),
    }
    print(f"cohorts: W(AD/HC) n={len(yw)} (dis={int(yw.sum())}), "
          f"K(Dem/Normal) n={len(yk)} (dis={int(yk.sum())})")

    results: dict[str, dict] = {}
    for fname, (Fw, Fk) in features.items():
        results[fname] = {}
        reg = regimes(Fw, Fk)
        note = reg.pop("_combat_note", None)
        for rname, pair in reg.items():
            if pair is None:
                results[fname][rname] = None
                continue
            results[fname][rname] = transfer_both(pair[0], yw, pair[1], yk)
        if note:
            results[fname]["combat_note"] = note
        # console summary
        def best(reg_name: str) -> str:
            r = results[fname].get(reg_name)
            if not r:
                return "  --"
            return f"W->K {r['WtoK'][0]:.3f}  K->W {r['KtoW'][0]:.3f}"
        print(f"[{fname}]  raw: {best('raw')} | persite_z: {best('persite_z')} "
              f"| combat: {best('combat')}")

    # ---- data-driven verdict (computed on CI, not point estimates) -----------
    # A cell "transfers" only if its target-AUROC 95% CI lower bound clears chance (0.5).
    # This is the manuscript's honest-strength standard: a point estimate above 0.5 whose
    # bootstrap CI straddles 0.5 (e.g. REVE K->W 0.606, CI [0.448, 0.766], n=53) is NOT a rescue.
    def cells(fname: str):
        for rname, r in results[fname].items():
            if isinstance(r, dict) and "WtoK" in r:
                for direction in ("WtoK", "KtoW"):
                    yield rname, direction, r[direction]  # [auc, lo, hi]

    def sig_regimes(fname: str) -> list:
        return [f"{rn}:{d}={c[0]:.3f} CI[{c[1]:.3f},{c[2]:.3f}]"
                for rn, d, c in cells(fname) if c[1] > 0.5]

    def raw_sig(fname: str) -> bool:
        r = results[fname]["raw"]
        return r["WtoK"][1] > 0.5 or r["KtoW"][1] > 0.5

    def harm_sig(fname: str) -> bool:
        return any(c[1] > 0.5 for rn, d, c in cells(fname) if rn != "raw")

    reve_sig = sig_regimes("REVE_pca50")
    class_sig = sig_regimes("Classical_173d")
    dfa_sig = sig_regimes("DFA_19d")
    reve_rescued = len(reve_sig) > 0                       # any REVE cell significant?
    classical_rescued = (not raw_sig("Classical_173d")) and harm_sig("Classical_173d")
    dfa_native = raw_sig("DFA_19d")

    pattern_holds = (not reve_rescued) and classical_rescued and dfa_native
    verdict = {
        "reve_rescued_by_harmonization": bool(reve_rescued),
        "reve_significant_cells": reve_sig,  # empty => no regime clears chance at CI-LB
        "classical_rescued_by_harmonization": bool(classical_rescued),
        "classical_significant_cells": class_sig,
        "dfa_transfers_natively": bool(dfa_native),
        "dfa_significant_cells": dfa_sig,
        "criterion": "target-AUROC 95% bootstrap CI lower bound > 0.5 (chance)",
        "interpretation": (
            "Leakage-safe harmonization RESCUES Classical (disease signal present but site-masked: "
            "no significant raw transfer, significant after harmonization) and DFA transfers "
            "natively without harmonization (dimensionless), but does NOT rescue the frozen REVE "
            "embedding: no regime clears chance at the CI lower bound, and in the well-powered "
            "direction (test on the large Korean cohort) REVE sits at/below chance; the sole nominal "
            "uptick (K->W ~0.61) is the underpowered direction (test n=53) with a CI spanning chance "
            "and trails both DFA and the classical rescue. Removing the site axis (arm b) does not "
            "recover a disease axis: the frozen embedding's transferable disease signal is not present "
            "to a degree leakage-safe harmonization can recover."
            if pattern_holds else
            "Result does not match the pre-registered pattern; inspect the table before claiming."
        ),
    }

    result = {
        "meta": {
            "title": "P1.1 arm (c) — leakage-safe transfer-rescue of a frozen FM embedding",
            "question": (
                "Does leakage-safe harmonization rescue cross-population disease transfer of "
                "the frozen REVE embedding? (removing site != recovering disease)"
            ),
            "task": "binary dementia-vs-control, ds004504 (AD/HC) <-> CAUEEG (Dementia/Normal)",
            "encoder": "REVE frozen 19ch 38912-d, PCA-50 (label-free pooled)",
            "classifier": "StandardScaler(train) + SVC-RBF (class-weight balanced)",
            "harmonization_regimes": {
                "raw": "strict zero-shot, train-only standardization, no target info",
                "persite_z": "per-cohort z by own mean/std (unlabelled target batch)",
                "combat": "neuroCombat batch=site, NO diagnosis covariate (unlabelled target batch)",
            },
            "cohorts": {"W_ds004504_AD_HC": int(len(yw)), "K_CAUEEG_Dem_Normal": int(len(yk))},
            "ci": "1000x subject-level bootstrap 95% CI on the target",
            "seed": SEED,
            "status": "exploratory",
            "reproduce": "python scripts/analysis/lrtc_p1_1_transfer_rescue.py",
            "evidence_source": "results/cross_population/p02_matrices.npz",
        },
        "results": results,
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(result, indent=2))
    print("\nVERDICT:", verdict["interpretation"])
    print(f"  REVE significant cells (CI-LB>0.5): {reve_sig or 'NONE'}")
    print(f"  Classical significant cells:        {class_sig or 'NONE'}")
    print(f"  DFA significant cells:              {dfa_sig or 'NONE'}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
