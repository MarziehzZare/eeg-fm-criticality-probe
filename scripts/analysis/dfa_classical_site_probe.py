#!/usr/bin/env python3
"""Site-decodability of the DFA exponent and classical features (paper Sec 4.5).

The paper contrasts the frozen FM embedding (decodes recording site at 0.98-1.00)
against the "site-invariant" DFA exponent. But *transfers* disease signal across
sites is not the same property as *site is hard to decode from it*. This script
tests the second property directly: the identical two-way site probe (ds004504 vs
CAUEEG, StandardScaler + logistic, 5-fold, matched Gaussian null) applied to
  (a) the 19-d DFA vector used as the transfer feature in Table 5, and
  (b) the 173-d classical feature vector,
so the "site-invariant" claim rests on measured site-decodability, not inference.

Result: DFA decodes site at ~0.71 (well below the FMs' near-ceiling), so the
exponent is comparatively site-robust; classical decodes site at ~1.00 (at
ceiling, like the FMs), showing that site-decodability alone is NOT what
distinguishes the useful features from the FM embedding -- the discriminating
property is site-decodable AND disease-absent (FM) vs site-decodable AND
disease-recoverable (classical) vs site-modest AND disease-present (DFA).

Features come from the same cached matrices as the transfer analysis
(p02_matrices.npz: 57-d [1/f-exp 19 | 1/f-off 19 | DFA 19], and 173-d classical).
The ds004504 side is the binary AD/HC transfer cohort (n=53), so site n is
53/cohort here vs 88/cohort for the FM probe; the >=0.3 gap between DFA (0.71)
and the FMs (~1.00) is far larger than any sample-size sensitivity.

Run from repo root:
  /opt/anaconda3/envs/neurogenis/bin/python scripts/analysis/dfa_classical_site_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import cross_population_criticality as cpc  # noqa: E402  (_impute)
from src.io_safety import safe_np_load  # noqa: E402

CACHE = ROOT / "results" / "cross_population" / "p02_matrices.npz"
OUT = ROOT / "results" / "cross_population" / "dfa_classical_site_probe.json"
SEED = 42
DFA_SLICE = slice(38, 57)   # [1/f-exp 19 | 1/f-off 19 | DFA 19]
K_NULL = 20


def site_probe(F_w: np.ndarray, F_k: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    cap = min(len(F_w), len(F_k))

    def bal(X):
        return X if len(X) <= cap else X[rng.choice(len(X), cap, replace=False)]

    Fw, Fk = bal(F_w), bal(F_k)
    X = np.vstack([Fw, Fk])
    site = np.concatenate([np.zeros(len(Fw)), np.ones(len(Fk))])
    n, d = X.shape

    def probe(Z, fold_seed):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        skf = StratifiedKFold(5, shuffle=True, random_state=fold_seed)
        return float(np.mean([balanced_accuracy_score(site[te], clf.fit(Z[tr], site[tr]).predict(Z[te]))
                              for tr, te in skf.split(Z, site)]))

    acc = probe(X, SEED)
    nulls = [probe(np.random.default_rng(s).standard_normal((n, d)), s) for s in range(K_NULL)]
    nm = float(np.mean(nulls))
    return {"balanced_accuracy": acc, "chance": 0.5, "n_per_cohort": int(cap), "dim": int(d),
            "null_mean": nm, "null_std": float(np.std(nulls)), "excess": acc / nm}


def main() -> None:
    m = safe_np_load(str(CACHE))
    Xw, Xk = cpc._impute(m["Xw"]), cpc._impute(m["Xk"])
    Cw, Ck = cpc._impute(m["Cw"]), cpc._impute(m["Ck"])
    dfa = site_probe(Xw[:, DFA_SLICE], Xk[:, DFA_SLICE])
    classical = site_probe(Cw, Ck)
    print(f"DFA 19-d      : site balacc {dfa['balanced_accuracy']:.4f} "
          f"(null {dfa['null_mean']:.4f}, {dfa['excess']:.2f}x, n={dfa['n_per_cohort']}/cohort)")
    print(f"classical 173d: site balacc {classical['balanced_accuracy']:.4f} "
          f"(null {classical['null_mean']:.4f}, {classical['excess']:.2f}x, n={classical['n_per_cohort']}/cohort)")

    out = {
        "meta": {"title": "Site-decodability of DFA and classical features (ds004504 vs CAUEEG)",
                 "task": "2-way recording-site decode, StandardScaler + logistic, 5-fold, matched Gaussian null",
                 "cohorts": {"W": "ds004504 (AD/HC transfer cohort)", "K": "CAUEEG"},
                 "seed": SEED, "null_seeds": K_NULL,
                 "comparison": "FM 2-way site probe (lrtc_irrecoverability_p1_1_multifm_2way.json): 0.977-1.000",
                 "reproduce": "python scripts/analysis/dfa_classical_site_probe.py"},
        "dfa_19d": dfa,
        "classical_173d": classical,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
