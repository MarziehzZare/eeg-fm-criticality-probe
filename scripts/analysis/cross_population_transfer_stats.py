"""P0.2 — honest error bars + permutation nulls for the cross-population transfer matrix.

The original `cross_population_criticality.py` reports transfer AUROC from a SINGLE fit with a
test-set bootstrap CI (`_transfer_ci`). That measures test-sample noise only, not
generalization variance, and gives no significance test. This script replaces it with:

  (1) REPEATED-RESAMPLE CI  — per cell, R repeats of {stratified bootstrap of the TRAIN
      population + bootstrap of the TEST population}, refit, score. Captures both training-
      sample and target-sample variability. Report median + 2.5/97.5 percentiles.
  (2) LABEL-PERMUTATION NULL — per cell, P refits on shuffled TRAIN labels scored on the
      true TEST labels. p_above = P(null >= obs), p_below = P(null <= obs) (add-one smoothed).
      Tests both "does this feature transfer ABOVE chance?" and "is the FM reversal BELOW
      chance?".
  (3) HOLM correction across the family of above-chance tests (one per cell).

Speed: RBF-SVM with a PRECOMPUTED kernel — the Gram matrix is built once per cell and reused
across all R resamples and P permutations (only labels / row-subsets change). Matches the
original estimator: StandardScaler -> SVC(rbf, C=1, gamma='scale', class_weight='balanced');
AUROC uses decision_function (monotone in predict_proba, so identical AUROC). The point AUROC
is cross-checked against the original make_pipeline estimator on the first cell.

Populations (binary disease vs control), harmonised 19ch/200Hz/CAR/1-45Hz/240s:
  Western = ds004504 (Greek) AD/HC ; Korean = CAUEEG Dementia/Normal.
Feature families: 1/f (38d), DFA (19d), 1/f+DFA (57d), REVE (38912d), classical (173d).
raw vs harmonised (label-free per-site z-score) for each; both transfer directions.

USAGE:
    /opt/anaconda3/envs/neurogenis/bin/python \
        scripts/analysis/cross_population_transfer_stats.py --repeats 300 --perms 1000 \
        --out results/cross_population/cross_population_transfer_stats.json

STATUS: exploratory (P0.2 hardening). Matrices cached to results/cross_population/p02_matrices.npz.
References: Hardstone 2012 (DFA); Donoghue 2020 (specparam); Holm 1979 (multiple comparisons).
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

from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.metrics.pairwise import rbf_kernel  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

import cross_population_criticality as cpc  # noqa: E402  (reuse loaders / _clf / _persite)
from src.io_safety import safe_np_load  # noqa: E402

RNG = np.random.default_rng(0)


def load_or_cache_matrices(cache: Path):
    if cache.exists():
        d = safe_np_load(str(cache))
        return {k: d[k] for k in d.files}
    Xw, yw, Rw, Cw, _ = cpc.load_western()
    Xk, yk, Rk, Ck, _ = cpc.load_korean()
    Xw, Xk = cpc._impute(Xw), cpc._impute(Xk)
    m = {"Xw": Xw, "yw": yw, "Rw": Rw, "Cw": Cw,
         "Xk": Xk, "yk": yk, "Rk": Rk, "Ck": Ck}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **m)
    return m


def _grams(Xtr: np.ndarray, Xte: np.ndarray):
    """StandardScaler(train) then RBF Gram matrices with gamma matching sklearn 'scale'."""
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    gamma = 1.0 / (A.shape[1] * A.var())          # 'scale' = 1/(n_features * X.var())
    return rbf_kernel(A, A, gamma=gamma), rbf_kernel(B, A, gamma=gamma)


def _svc_auc(Ktr, ytr, Kte, yte):
    clf = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(Ktr, ytr)
    return roc_auc_score(yte, clf.decision_function(Kte))


def _strat_boot(y, rng):
    return np.concatenate([np.where(y == c)[0][rng.integers(0, (y == c).sum(), (y == c).sum())]
                           for c in np.unique(y)])


def _boot_both(y, rng):
    while True:
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) > 1:
            return idx


def transfer_cell(Xtr, ytr, Xte, yte, repeats, perms, rng):
    Ktr, Kte = _grams(Xtr, Xte)
    obs = _svc_auc(Ktr, ytr, Kte, yte)

    rep = np.empty(repeats)
    for r in range(repeats):
        ti = _strat_boot(ytr, rng)
        ei = _boot_both(yte, rng)
        rep[r] = _svc_auc(Ktr[np.ix_(ti, ti)], ytr[ti], Kte[np.ix_(ei, ti)], yte[ei])

    nul = np.empty(perms)
    for p in range(perms):
        nul[p] = _svc_auc(Ktr, rng.permutation(ytr), Kte, yte)

    p_above = (1 + int((nul >= obs).sum())) / (perms + 1)
    p_below = (1 + int((nul <= obs).sum())) / (perms + 1)
    return {
        "point_auroc": round(float(obs), 4),
        "resample_median": round(float(np.median(rep)), 4),
        "resample_ci95": [round(float(np.percentile(rep, 2.5)), 4),
                          round(float(np.percentile(rep, 97.5)), 4)],
        "null_mean": round(float(nul.mean()), 4),
        "p_above_chance": round(p_above, 5),
        "p_below_chance": round(p_below, 5),
    }


def holm(pvals: dict[str, float]) -> dict[str, dict]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[k] = {"p_holm": round(adj, 5), "reject_0.05": bool(adj < 0.05)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=300)
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/cross_population/cross_population_transfer_stats.json")
    ap.add_argument("--cache", type=Path,
                    default=ROOT / "results/cross_population/p02_matrices.npz")
    args = ap.parse_args()
    os.chdir(ROOT)

    print("[P0.2] loading / caching matrices ...", flush=True)
    M = load_or_cache_matrices(args.cache)
    yw, yk = M["yw"], M["yk"]
    fam = {"1/f": slice(0, 38), "DFA": slice(38, 57), "1/f+DFA": slice(0, 57)}
    feats = {name: (M["Xw"][:, sl], M["Xk"][:, sl]) for name, sl in fam.items()}
    feats["REVE"] = (M["Rw"], M["Rk"])
    feats["classical"] = (M["Cw"], M["Ck"])
    print(f"[P0.2] N_western={len(yw)} N_korean={len(yk)}; repeats={args.repeats} perms={args.perms}",
          flush=True)

    # cross-check the precomputed-kernel estimator vs the original pipeline on one cell
    A, B = feats["DFA"]
    Ktr, Kte = _grams(A, B)
    pipe_auc = roc_auc_score(yk, cpc._clf().fit(A, yw).predict_proba(B)[:, 1])
    print(f"[P0.2] estimator check (DFA W->K): precomputed={_svc_auc(Ktr, yw, Kte, yk):.3f} "
          f"vs pipeline={pipe_auc:.3f}", flush=True)

    out = {"meta": {"n_western": int(len(yw)), "n_korean": int(len(yk)),
                    "repeats": args.repeats, "perms": args.perms,
                    "harmonisation": "19ch/200Hz/CAR/1-45Hz/240s; harm=label-free per-site z",
                    "estimator": "RBF-SVM (precomputed kernel), C=1, gamma=scale, balanced",
                    "status": "exploratory (P0.2)"},
           "cells": {}}
    above_p = {}

    for name, (A, B) in feats.items():
        for scheme in ("raw", "harm"):
            if scheme == "harm":
                Ah, Bh = cpc._persite(A, B)
            else:
                Ah, Bh = A, B
            for direction, (Xtr, ytr, Xte, yte) in {
                "WtoK": (Ah, yw, Bh, yk), "KtoW": (Bh, yk, Ah, yw)}.items():
                key = f"{name}|{scheme}|{direction}"
                res = transfer_cell(Xtr, ytr, Xte, yte, args.repeats, args.perms, RNG)
                out["cells"][key] = res
                above_p[key] = res["p_above_chance"]
                ci = res["resample_ci95"]
                print(f"  {key:28s} AUROC={res['point_auroc']:.3f} "
                      f"med={res['resample_median']:.3f} CI[{ci[0]:.3f},{ci[1]:.3f}] "
                      f"null={res['null_mean']:.3f} p_above={res['p_above_chance']:.4f} "
                      f"p_below={res['p_below_chance']:.4f}", flush=True)

    # Two corrections: (i) the PRE-SPECIFIED PRIMARY family = the criticality claim
    # (features DFA and 1/f+DFA); 1/f-alone and REVE are pre-declared null/negative
    # controls, NOT part of the claim family. (ii) a transparent all-20 Holm as well.
    primary = {k: p for k, p in above_p.items() if k.split("|")[0] in ("DFA", "1/f+DFA")}
    holm_all = holm(above_p)
    holm_prim = holm(primary)
    for key in out["cells"]:
        out["cells"][key]["p_holm_all20"] = holm_all[key]["p_holm"]
        out["cells"][key]["reject_holm_all20"] = holm_all[key]["reject_0.05"]
        if key in holm_prim:
            out["cells"][key]["p_holm_primary"] = holm_prim[key]["p_holm"]
            out["cells"][key]["reject_holm_primary"] = holm_prim[key]["reject_0.05"]
    n_prim = sum(v["reject_0.05"] for v in holm_prim.values())
    n_all = sum(v["reject_0.05"] for v in holm_all.values())
    out["meta"]["n_cells"] = len(above_p)
    out["meta"]["primary_family"] = sorted(primary)
    out["meta"]["n_primary_above_chance_holm"] = int(n_prim)
    out["meta"]["n_all20_above_chance_holm"] = int(n_all)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[P0.2] primary (criticality) family: {n_prim}/{len(primary)} above chance (Holm .05); "
          f"all-20: {n_all}/{len(above_p)}. wrote {args.out}", flush=True)
    print("[P0.2] primary-family cells:", flush=True)
    for k in sorted(primary):
        c = out["cells"][k]
        print(f"    {k:22s} AUROC={c['point_auroc']:.3f} CI{c['resample_ci95']} "
              f"p_raw={c['p_above_chance']:.4f} p_holm_primary={c['p_holm_primary']:.4f} "
              f"{'SIG' if c['reject_holm_primary'] else 'ns'}", flush=True)


if __name__ == "__main__":
    main()
