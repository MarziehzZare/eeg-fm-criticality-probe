"""3-cohort cross-population transfer — does a 3rd cohort (BrainLat) power the claim?

The 2-cohort P0.2 result was underpowered (1/8 criticality cells survive Holm; W->K direction
n=53 non-significant). This adds BrainLat (Latin-American AD+bvFTD vs HC, N=79) as a 3rd population
and runs, for the criticality features (DFA and 1/f+DFA):
  - all 6 ordered PAIRWISE transfers among {W=Greek AD/HC, K=Korean dem/normal, B=BrainLat dem/HC}
  - 3 LEAVE-ONE-COHORT-OUT tests (train on the pooled other two, test on the held-out cohort)
Each with a point AUROC + repeated-resample CI + label-permutation null; Holm across the family.

Estimator matches P0.2: transfer_cell -> StandardScaler(train) + RBF-SVM (C=1, gamma=scale, balanced),
AUROC via decision_function. All zero-shot (train-only standardization; no per-site z of the test set).

Features are the shared 57-d 1/f+DFA vector (cpc.extract) so the 3 cohorts are directly comparable.
(FM/classical extension for BrainLat is a follow-up — needs REVE embeds + classical features on BrainLat.)

USAGE: /opt/anaconda3/envs/neurogenis/bin/python scripts/analysis/transfer_3cohort.py [--perms 5000]
STATUS: exploratory (P0.4 transfer-power test with the 3rd cohort).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import cross_population_transfer_stats as ts  # noqa: E402  (transfer_cell, holm, RNG)
from src.io_safety import safe_np_load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=300)
    ap.add_argument("--perms", type=int, default=5000)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/cross_population/transfer_3cohort.json")
    args = ap.parse_args()
    os.chdir(ROOT)

    m = safe_np_load("results/cross_population/p02_matrices.npz")
    b = safe_np_load("results/cross_population/brainlat_features.npz")
    coh = {
        "W": (m["Xw"], m["yw"]),                 # Greek AD/HC
        "K": (m["Xk"], m["yk"]),                 # Korean dementia/normal
        "B": (b["X"], b["y"]),                   # BrainLat AD+bvFTD/HC
    }
    fam = {"DFA": slice(38, 57), "1/f+DFA": slice(0, 57)}
    print(f"[3coh] N: W={len(coh['W'][1])} K={len(coh['K'][1])} B={len(coh['B'][1])}; "
          f"repeats={args.repeats} perms={args.perms}", flush=True)

    out = {"meta": {"cohorts": {k: int(len(v[1])) for k, v in coh.items()},
                    "repeats": args.repeats, "perms": args.perms,
                    "estimator": "StandardScaler(train)+RBF-SVM C=1 gamma=scale balanced (zero-shot)",
                    "status": "exploratory (3-cohort transfer power)"},
           "pairwise": {}, "loco": {}}
    above_p = {}

    for fname, sl in fam.items():
        # pairwise
        for tr in coh:
            for te in coh:
                if tr == te:
                    continue
                Xtr, ytr = coh[tr][0][:, sl], coh[tr][1]
                Xte, yte = coh[te][0][:, sl], coh[te][1]
                key = f"{fname}|{tr}->{te}"
                res = ts.transfer_cell(Xtr, ytr, Xte, yte, args.repeats, args.perms, ts.RNG)
                out["pairwise"][key] = res
                above_p[key] = res["p_above_chance"]
                ci = res["resample_ci95"]
                print(f"  {key:18s} AUROC={res['point_auroc']:.3f} CI[{ci[0]:.3f},{ci[1]:.3f}] "
                      f"p_above={res['p_above_chance']:.4f}", flush=True)
        # leave-one-cohort-out
        for held in coh:
            others = [c for c in coh if c != held]
            Xtr = np.vstack([coh[o][0][:, sl] for o in others])
            ytr = np.concatenate([coh[o][1] for o in others])
            Xte, yte = coh[held][0][:, sl], coh[held][1]
            key = f"{fname}|{'+'.join(others)}->{held}"
            res = ts.transfer_cell(Xtr, ytr, Xte, yte, args.repeats, args.perms, ts.RNG)
            out["loco"][key] = res
            above_p[key] = res["p_above_chance"]
            ci = res["resample_ci95"]
            print(f"  [LOCO] {key:20s} AUROC={res['point_auroc']:.3f} CI[{ci[0]:.3f},{ci[1]:.3f}] "
                  f"p_above={res['p_above_chance']:.4f}", flush=True)

    holm_res = ts.holm(above_p)
    for key, hv in holm_res.items():
        (out["pairwise"] if key in out["pairwise"] else out["loco"])[key].update(hv)
    n_sig = sum(v["reject_0.05"] for v in holm_res.values())
    out["meta"]["n_cells"] = len(above_p)
    out["meta"]["n_above_chance_holm_0.05"] = int(n_sig)

    args.out.write_text(json.dumps(out, indent=2))
    try:
        from src.runs import mirror_results_file
        mirror_results_file(args.out, stem="transfer_3cohort_results")  # no-op unless POC_RUN_DIR set
    except Exception:
        pass
    print(f"\n[3coh] {n_sig}/{len(above_p)} cells transfer above chance (Holm .05); wrote {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
