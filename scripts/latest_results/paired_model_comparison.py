"""Compute a conditional paired model versus baseline comparison.

The input contains fixed predictions for the same records. No model is fitted
here. The statistic is the difference in R squared on one shared denominator,
with a paired BCa bootstrap interval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def r_squared(y: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.sum((y - y.mean()) ** 2)
    if denominator <= 0:
        raise ValueError("target has zero variance")
    return float(1.0 - np.sum((y - prediction) ** 2) / denominator)


def delta_statistic(y: np.ndarray, model: np.ndarray, baseline: np.ndarray) -> float:
    denominator = np.sum((y - y.mean()) ** 2)
    return float((np.sum((y - baseline) ** 2) - np.sum((y - model) ** 2)) / denominator)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as data:
        y = np.asarray(data["y"], dtype=np.float64)
        model = np.asarray(data["model"], dtype=np.float64)
        baseline = np.asarray(data["baseline"], dtype=np.float64)
    if y.ndim != 1 or model.ndim != 1 or baseline.ndim != 1:
        raise ValueError("y, model, and baseline must be one-dimensional")
    if not (len(y) == len(model) == len(baseline)) or len(y) < 3:
        raise ValueError("arrays must have equal length and at least three records")
    if not np.isfinite(np.concatenate([y, model, baseline])).all():
        raise ValueError("input arrays contain non-finite values")

    delta = delta_statistic(y, model, baseline)

    def statistic(yy, mm, bb):
        return delta_statistic(yy, mm, bb)

    result = stats.bootstrap(
        (y, model, baseline), statistic, n_resamples=args.bootstrap,
        paired=True, method="BCa", confidence_level=0.95,
        random_state=np.random.default_rng(args.seed),
    )
    report = {
        "n": int(len(y)),
        "model_r2": r_squared(y, model),
        "baseline_r2": r_squared(y, baseline),
        "delta_r2": delta,
        "conditional_bca_95": [float(result.confidence_interval.low),
                                float(result.confidence_interval.high)],
        "bootstrap": int(args.bootstrap),
        "seed": int(args.seed),
        "note": "fixed predictions, paired records, no refitting uncertainty",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
