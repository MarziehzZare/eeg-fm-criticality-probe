"""Rerun the DFA-vs-1/f residualization check (LRTC-not-a-shadow-of-aperiodic
claim, paper Sec. 4.3) at the full N=770 CAUEEG sample instead of the original
N=200, reusing the already-cached targets and embeddings (no raw EEG re-read).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))

import scripts.analysis.fm_lrtc_nonlinear_probe as p01  # noqa: E402
from src.io_safety import safe_np_load  # noqa: E402

CACHE = ROOT / "results/cross_population/p01_targets_N1187.npz"
OUT = ROOT / "results/cross_population/residualization_fullN.json"
PROBES = ["ridge_linear_full", "hist_gboost", "random_forest"]


def main() -> None:
    tcache = {k: list(v) for k, v in safe_np_load(str(CACHE))["data"].item().items()}
    enc = p01.load_embeddings()
    common = set(enc["REVE"]) & set(enc["CBraMod"]) & set(enc["BIOT"]) \
        & set(enc["LaBraM"]) & set(enc["classical"])
    serials = [s for s in tcache if s in common]
    n = len(serials)
    print(f"N={n}", flush=True)

    dfa = np.array([tcache[s][0] for s in serials])   # DFA_full
    beta = np.array([tcache[s][2] for s in serials])  # aperiodic exponent (1/f)

    r, p = stats.pearsonr(beta, dfa)
    r2_lin = LinearRegression().fit(beta.reshape(-1, 1), dfa).score(beta.reshape(-1, 1), dfa)
    resid = dfa - LinearRegression().fit(beta.reshape(-1, 1), dfa).predict(beta.reshape(-1, 1))
    print(f"r(beta,DFA)={r:+.4f} p={p:.4g}  beta explains {100*r2_lin:.2f}% of DFA variance", flush=True)

    X = {name: np.vstack([enc[name][s] for s in serials])
         for name in ("REVE", "CBraMod", "BIOT", "LaBraM", "classical")}

    out = {
        "meta": {"title": "DFA-vs-1/f residualization at full N (rerun of Sec 4.3)",
                 "n": n, "n_original_paper": 200,
                 "r_beta_dfa": r, "p_beta_dfa": p, "beta_explains_pct_dfa_var": 100 * r2_lin,
                 "probes": PROBES, "status": "exploratory (reviewer-requested full-N rerun)"},
        "raw_dfa_r2": {}, "residualized_dfa_r2": {},
    }
    for name in ("REVE", "CBraMod", "BIOT", "LaBraM", "classical"):
        d = X[name].shape[1]
        all_probes = p01.make_probes(d, n)
        probes = {k: v for k, v in all_probes.items() if k in PROBES}
        raw_best = max(p01.cv_r2(pfn, X[name], dfa) for pfn in probes.values())
        res_best = max(p01.cv_r2(pfn, X[name], resid) for pfn in probes.values())
        out["raw_dfa_r2"][name] = round(raw_best, 4)
        out["residualized_dfa_r2"][name] = round(res_best, 4)
        print(f"  {name:9s} raw_DFA_R2={raw_best:+.3f}  residualized_DFA_R2={res_best:+.3f}", flush=True)

    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
