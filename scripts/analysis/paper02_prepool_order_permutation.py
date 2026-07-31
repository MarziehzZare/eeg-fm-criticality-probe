"""Control B: a subject-level shuffle reference for the pre-pool order effect.

The seed-level paired comparison used in the first pass mixes order sensitivity
with optimization variation (fold split, initialization, and mini-batch
trajectory). This script instead matches each shuffle realization across fixed
configurations and reports batch-size and configuration sensitivity explicitly.

Design:
  * Fold split and network initialization are held FIXED within a comparison, so the ordered
    run and every shuffled run differ only in the temporal order of the sequence.
  * The reference distribution is built from K independent subject-level shuffle
    realizations (each subject's sequence independently permuted), not from
    re-seeding the optimizer.
  * Test statistic  T = mean over configurations of the ordered out-of-fold R^2.
    Reference draw T_k = mean over configurations of the k-th shuffled R^2 (matched configs).
    tau_MC = (1 + #{k : T_k >= T}) / (K + 1)
  * The whole procedure is repeated at two batch sizes so the conclusion is reported with its
    hyperparameter sensitivity rather than conditioned on one choice.
  * Convergence is recorded (epochs to early stop, best validation loss) because differential
    under-training of the shuffled arm is one possible source of an apparent gap.

Inferential limit:
  tau_MC is a calibrated randomization p-value only under the sharp null that the
  chronological within-subject sequence is exchangeable with the sampled
  permutations. Real EEG violates that assumption through autocorrelation,
  non-stationarity, drift, artifact clustering, and position effects. The paper
  therefore reports tau_MC as a descriptive shuffle-reference tail proportion,
  not as an LRTC-specific p-value.

Reading (deliberately weak; none of these are proofs):
  large tau_MC at both batch sizes -> no order advantage was DETECTED by this probe at this
                                  granularity.  This is consistent with an order-insensitive
                                  proxy, but it is not evidence that no readout could detect one.
  tau_MC sensitive to batch size -> the detected effect is not stable under this hyperparameter.
                                  Optimization dynamics AND effective model capacity or
                                  regularization are both admissible explanations; the
                                  sensitivity should be reported rather than resolved by
                                  choosing one setting.
  small tau_MC at both batch sizes -> an order advantage was detected relative to
                                  this shuffle reference. Because preserving order also
                                  preserves non-stationarity, arousal drift, artifact clustering
                                  and local temporal continuity, this does NOT isolate LRTC.

USAGE:
  python scripts/analysis/paper02_prepool_order_permutation.py \
      [--n-perm 30] [--configs 2]
STATUS: POST HOC exploratory control for the Paper 02 revision (designed after earlier results
were seen; not a prespecified confirmatory test), consistent with the JSON metadata.

KNOWN HETEROGENEITY: the ordered statistic varies substantially across fold/initialization
configurations.  In the CBraMod bs=128 cell, three configurations trained to ordered
$R^2\approx0.28$ with a +0.08 to +0.10 increment over their matched nulls, while one collapsed in
BOTH arms (ordered 0.065, reference 0.056), i.e. a globally poor
optimization/split configuration rather than a selective loss of the order
effect. The aggregate statistic is well-defined as the mean across
configurations, but the effect must not be described as stable across
configurations. Per-configuration values are retained for reporting.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import paper02_prepool_data as po  # noqa: E402  (v2 targets + sequence loaders)
import paper02_sequence_probe as sp  # noqa: E402
from src.seed import set_seed  # noqa: E402

def run_model(name, sequences, y, batch_size, n_perm, config_seeds):
    """Matched-realization shuffle-reference benchmark.

    CRITICAL (corrected 2026-07-29): permutation k uses ONE subject-wise realization,
    evaluated under EVERY fold/initialization configuration, and those matched results are
    averaged.  The earlier version drew a different realization per configuration
    (shuffle_seed = 1000 + 97*k + c), which averaged away the between-realization variance
    component in the reference statistic and changes its cross-configuration
    covariance. The variance and tail can move in either direction, so those
    tail fractions are invalid rather than uniformly conservative or
    anti-conservative. The seed is now a function of k alone, so the observed
    and reference statistics are averaged over the same kind of object.
    """
    configs = list(config_seeds)
    ordered, ord_epochs, ord_val, ord_fold = [], [], [], []
    for c in configs:
        r2, ep, vl, folds = sp.cv_r2(
            sequences,
            y,
            batch_size=batch_size,
            config_seed=c,
            shuffle_seed=None,
        )
        ordered.append(r2); ord_epochs.append(ep); ord_val.append(vl); ord_fold.append(folds)
    t_obs = float(np.mean(ordered))

    null_matrix, null_epochs, null_val = [], [], []
    for k in range(n_perm):
        realization_seed = 1000 + k          # SAME realization across all configurations
        per_config = []
        for c in configs:
            r2, ep, vl, _ = sp.cv_r2(
                sequences,
                y,
                batch_size=batch_size,
                config_seed=c,
                shuffle_seed=realization_seed,
            )
            per_config.append(r2); null_epochs.append(ep); null_val.append(vl)
        null_matrix.append([float(v) for v in per_config])
    null_matrix = np.array(null_matrix)                       # (n_perm, n_configs)
    null = null_matrix.mean(axis=1)
    p = float((1 + np.sum(null >= t_obs)) / (n_perm + 1))
    return {
        "batch_size": batch_size,
        "n_configs": len(configs),
        "config_seeds": configs,
        "n_permutations": n_perm,
        "shuffle_seed_formula": "1000 + k (identical realization across configurations)",
        "ordered_r2_per_config": [round(v, 5) for v in ordered],
        "ordered_r2_mean": round(t_obs, 4),
        "shuffled_null_mean": round(float(null.mean()), 4),
        "shuffled_null_sd": round(float(null.std()), 4),
        "shuffled_null_q95": round(float(np.percentile(null, 95)), 4),
        "observed_minus_null_mean": round(t_obs - float(null.mean()), 4),
        "shuffle_reference_tail_fraction_one_sided": round(p, 4),
        "min_attainable_tail_fraction": round(1.0 / (n_perm + 1), 4),
        # full auditability: every individual permutation value is retained
        "null_means_all": [round(float(v), 5) for v in null],
        "null_matrix_per_config": [[round(float(v), 5) for v in row] for row in null_matrix],
        "convergence": {
            "ordered_mean_epochs": round(float(np.mean(ord_epochs)), 1),
            "shuffled_mean_epochs": round(float(np.mean(null_epochs)), 1),
            "ordered_mean_best_val_loss": round(float(np.mean(ord_val)), 5),
            "shuffled_mean_best_val_loss": round(float(np.mean(null_val)), 5),
            "ordered_per_fold_epochs": ord_fold,
            "ordered_epochs_per_config": [round(float(v), 1) for v in ord_epochs],
            "shuffled_epochs_all": [round(float(v), 1) for v in null_epochs],
            "shuffled_best_val_loss_all": [round(float(v), 5) for v in null_val],
            "note": ("comparable mean stopping epochs are consistent with, but do not prove, "
                     "equal effective optimization; per-fold values are retained so the "
                     "validation behaviour can be re-examined"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=30)
    ap.add_argument("--configs", type=int, default=2)
    ap.add_argument(
        "--config-seeds",
        nargs="+",
        type=int,
        help="explicit fold/init seeds; overrides --configs",
    )
    ap.add_argument("--primary-batch", type=int, default=128)
    ap.add_argument("--sensitivity-batch", type=int, default=32)
    ap.add_argument(
        "--sensitivity-models",
        nargs="*",
        default=["CBraMod", "BIOT"],
        help="models for batch-size sensitivity; pass the flag with no values to skip",
    )
    ap.add_argument("--models", nargs="+",
                    default=["REVE", "LaBraM", "BENDR", "CBraMod", "BIOT"],
                    help="models for the primary (batch-size) pass")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/paper02/prepool_order_permutation.json")
    args = ap.parse_args()
    set_seed(0)
    config_seeds = (
        list(args.config_seeds)
        if args.config_seeds is not None
        else list(range(args.configs))
    )
    if len(config_seeds) != len(set(config_seeds)):
        raise ValueError("configuration seeds must be unique")

    targets = po._v2_targets()
    sequences = po._load_sequences()
    out = {
        "meta": {
            "title": "Subject-level shuffle reference for the pre-pool order effect",
            "target": "v2 alpha-envelope DFA (mean 19ch)",
            "statistic": "mean ordered OOF R^2 over fixed fold/init configurations",
            "shuffle_reference": (
                "K independent subject-level sequence shuffles; for each draw ONE "
                "realization is evaluated under every fold/initialization configuration and "
                "those matched results are averaged, so observed and reference statistics "
                "average over the same kind of object"
            ),
            "tail_fraction": (
                "(1 + #{shuffle reference >= observed}) / (K + 1), one-sided"
            ),
            "rationale": ("replaces the seed-level paired t-test, which confounded the order "
                          "effect with optimization noise and was batch-size dependent"),
            "correction_history": (
                "An earlier version used shuffle_seed = 1000 + 97*k + c, giving each "
                "configuration a different realization within a permutation. "
                "That changed the cross-configuration covariance and made the "
                "tail fractions invalid for the intended statistic; the "
                "direction is not guaranteed. Superseded by this run."),
            "n_permutations": args.n_perm,
            "n_configs": len(config_seeds),
            "config_seeds": config_seeds,
            "interpretation_limits": [
                "The Monte Carlo tail fraction is a calibrated randomization p-value only "
                "under a sharp within-subject sequence-exchangeability null. Chronological "
                "EEG is not generally exchangeable, so it is reported descriptively.",
                "Order-sensitivity is not LRTC-specific: non-stationarity, arousal drift, "
                "artifact clustering and local temporal continuity are alternative explanations.",
                "CAUEEG only (n=199 for the four epoch-sequence models); provides no "
                "cross-population replication and does not alter cross-cohort conclusions.",
                "Designed after earlier results were seen; POST HOC / EXPLORATORY. "
                "Multiplicity-adjusted thresholds remain descriptive.",
                "Comparing the sequence-CNN R^2 with the mean-pooled ridge R^2 mixes readout "
                "architectures; only the ordered-minus-shuffled increment is interpretable.",
            ],
            "status": "POST HOC exploratory control for the Paper 02 revision",
        },
        "primary": {},
        "sensitivity": {},
    }

    for model in args.models:
        table = sequences.get(model)
        if not table:
            continue
        usable = sorted(set(table) & set(targets))
        p_min = min(table[s].shape[0] for s in usable)
        stack = np.stack([table[s][:p_min] for s in usable]).astype(np.float32)
        y = np.array([targets[s] for s in usable], dtype=float)
        t0 = time.time()
        row = run_model(model, stack, y, args.primary_batch, args.n_perm, config_seeds)
        row.update({"n": len(usable), "seq_len": int(p_min)})
        out["primary"][model] = row
        print(f"[bs={args.primary_batch}] {model:8s} n={len(usable):4d} "
              f"ordered={row['ordered_r2_mean']:+.3f} null={row['shuffled_null_mean']:+.3f}"
              f"±{row['shuffled_null_sd']:.3f} diff={row['observed_minus_null_mean']:+.3f} "
              f"tau_MC={row['shuffle_reference_tail_fraction_one_sided']:.3f} "
              f"[epochs ord/shuf {row['convergence']['ordered_mean_epochs']:.0f}/"
              f"{row['convergence']['shuffled_mean_epochs']:.0f}] ({time.time()-t0:.0f}s)",
              flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))

    for model in args.sensitivity_models:
        table = sequences.get(model)
        if not table:
            continue
        usable = sorted(set(table) & set(targets))
        p_min = min(table[s].shape[0] for s in usable)
        stack = np.stack([table[s][:p_min] for s in usable]).astype(np.float32)
        y = np.array([targets[s] for s in usable], dtype=float)
        row = run_model(model, stack, y, args.sensitivity_batch, args.n_perm, config_seeds)
        row.update({"n": len(usable), "seq_len": int(p_min)})
        out["sensitivity"][model] = row
        print(f"[bs={args.sensitivity_batch}] {model:8s} "
              f"ordered={row['ordered_r2_mean']:+.3f} null={row['shuffled_null_mean']:+.3f}"
              f"±{row['shuffled_null_sd']:.3f} diff={row['observed_minus_null_mean']:+.3f} "
              f"tau_MC={row['shuffle_reference_tail_fraction_one_sided']:.3f} "
              f"[epochs ord/shuf {row['convergence']['ordered_mean_epochs']:.0f}/"
              f"{row['convergence']['shuffled_mean_epochs']:.0f}]", flush=True)
        args.out.write_text(json.dumps(out, indent=2))

    print(f"\n[perm] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
