#!/usr/bin/env python3
"""CAUEEG DFA_short re-probe at N=770 (confound-resolution check for paper Sec 4.1).

The paper reports that CBraMod/BIOT recover CAUEEG DFA (0.18/0.25 on the full
0.5-30s scale range) but this collapses on BrainLat (<=0.06). Because BrainLat
necessarily uses the short 0.5-2s DFA range, the non-replication could be a
change in target definition rather than a real cohort effect. This script rules
that out by re-probing CBraMod/BIOT on CAUEEG against DFA_short (the same 0.5-2s
range used on BrainLat). Result: 0.175/0.242, within noise of the full-range
0.178/0.251 -> CAUEEG recovery is scale-range-invariant, so the BrainLat collapse
is a genuine cohort effect.

Reuses the cached targets (p01_targets_N1187.npz, index 1 = DFA_short) and the
same encoding-probe machinery as fm_lrtc_nonlinear_probe.py; no raw EEG re-read.

Run from repo root:
  /opt/anaconda3/envs/neurogenis/bin/python scripts/analysis/caueeg_dfa_short_reprobe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import scripts.analysis.fm_lrtc_nonlinear_probe as p01  # noqa: E402
from src.io_safety import safe_np_load  # noqa: E402

CACHE = ROOT / "results/cross_population/p01_targets_N1187.npz"
OUT = ROOT / "results/cross_population/caueeg_dfa_short_reprobe_N770.json"
PROBES = ["ridge_linear_full", "hist_gboost", "random_forest"]


def main() -> None:
    tcache = {k: list(v) for k, v in safe_np_load(str(CACHE))["data"].item().items()}
    enc = p01.load_embeddings()
    common = set(enc["REVE"]) & set(enc["CBraMod"]) & set(enc["BIOT"]) \
        & set(enc["LaBraM"]) & set(enc["classical"])
    serials = [s for s in tcache if s in common]
    n = len(serials)
    dfa_short = np.array([tcache[s][1] for s in serials])  # index 1 = DFA_short target
    print(f"N={n}  DFA_short range [{dfa_short.min():.3f}, {dfa_short.max():.3f}]", flush=True)

    out = {}
    for name in ("CBraMod", "BIOT"):
        X = np.vstack([enc[name][s] for s in serials])
        probes = {k: v for k, v in p01.make_probes(X.shape[1], n).items() if k in PROBES}
        best = max(p01.cv_r2(fn, X, dfa_short) for fn in probes.values())
        out[name] = round(best, 4)
        print(f"  {name:8s} DFA_short best R2 = {best:.4f}", flush=True)

    result = {
        "meta": {"title": "CAUEEG DFA_short re-probe at N=770 (confound resolution, Sec 4.1)",
                 "n": n, "probes": PROBES,
                 "purpose": ("verify CBraMod/BIOT DFA recovery is scale-range-invariant, not an "
                             "artifact of comparing full-range (CAUEEG) vs short-range (BrainLat)"),
                 "reproduce": "python scripts/analysis/caueeg_dfa_short_reprobe.py"},
        "dfa_short_r2": out,
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
