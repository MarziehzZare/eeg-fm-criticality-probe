"""Rerun the CAUEEG encoding probe (Table 2 of the LRTC paper) at the full
available N instead of the original N=200 cap, using only the three probes
Table 2 actually reports (ridge_linear_full, hist_gboost, random_forest) so
this completes quickly by reusing the already-computed target cache.

Addresses reviewer point #2: N=200 on CAUEEG when N=1187 subjects/embeddings
are available is an easy rebuttal target for a negative ("not decodable")
result. This reruns at the full N with cached DFA/1f targets.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))

import scripts.analysis.fm_lrtc_nonlinear_probe as p01  # noqa: E402
from src.io_safety import safe_np_load  # noqa: E402

CACHE = ROOT / "results/cross_population/p01_targets_N1187.npz"
OUT = ROOT / "results/cross_population/multifm_lrtc_encoding_fullN.json"
PROBES = ["ridge_linear_full", "hist_gboost", "random_forest"]


def main() -> None:
    tcache = {k: list(v) for k, v in safe_np_load(str(CACHE))["data"].item().items()}
    enc = p01.load_embeddings()
    common = set(enc["REVE"]) & set(enc["CBraMod"]) & set(enc["BIOT"]) \
        & set(enc["LaBraM"]) & set(enc["classical"])
    serials = [s for s in tcache if s in common]
    print(f"[rerun] N={len(serials)} subjects with cached targets + all 5 encoders", flush=True)

    y_full = np.array([tcache[s][0] for s in serials])
    y_short = np.array([tcache[s][1] for s in serials])
    y_1f = np.array([tcache[s][2] for s in serials])
    y = {"DFA_full": y_full, "DFA_short": y_short, "1/f": y_1f}

    X = {name: np.vstack([enc[name][s] for s in serials])
         for name in ("REVE", "CBraMod", "BIOT", "LaBraM", "classical")}

    n = len(serials)
    out = {
        "meta": {
            "title": "CAUEEG encoding probe at full available N (rerun of Table 2, reviewer request)",
            "n": n, "n_original_paper": 200,
            "encoders": {k: int(X[k].shape[1]) for k in X},
            "targets": ["DFA_full", "DFA_short", "1/f"],
            "probes": PROBES,
            "cv": "5-fold KFold shuffle seed 0, R^2",
            "status": "exploratory (reviewer-requested full-N rerun)",
        },
        "cv_r2": {},
    }
    for name in ("REVE", "CBraMod", "BIOT", "LaBraM", "classical"):
        d = X[name].shape[1]
        all_probes = p01.make_probes(d, n)
        probes = {k: v for k, v in all_probes.items() if k in PROBES}
        out["cv_r2"][name] = {}
        for tname, yv in y.items():
            row = {p: round(p01.cv_r2(pfn, X[name], yv), 4) for p, pfn in probes.items()}
            out["cv_r2"][name][tname] = row
            best = max(row.values())
            print(f"  {name:9s} {tname:9s} | best={best:+.3f}  "
                  + "  ".join(f"{p}={v:+.3f}" for p, v in row.items()), flush=True)

    OUT.write_text(json.dumps(out, indent=2))
    print(f"[rerun] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
