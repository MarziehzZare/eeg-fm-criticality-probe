"""Max-statistic correction for paired model versus baseline contrasts.

The input contains fixed predictions for the same records. Each declared
family uses the same paired bootstrap rows for all contrasts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def r2_rows(y: np.ndarray, prediction: np.ndarray, rows: np.ndarray) -> np.ndarray:
    yy = y[rows]
    pp = prediction[rows]
    centred = yy - yy.mean(axis=1, keepdims=True)
    denominator = np.sum(centred ** 2, axis=1)
    if np.any(denominator <= 0):
        raise ValueError("a bootstrap row has zero target variance")
    return 1.0 - np.sum((yy - pp) ** 2, axis=1) / denominator


def correct_family(name, contrasts, y, predictions, rows):
    observed = {}
    draws = {}
    for label, (high, low) in contrasts.items():
        observed[label] = float(r2_rows(y, predictions[high], np.arange(len(y))[None, :])[0]
                                 - r2_rows(y, predictions[low], np.arange(len(y))[None, :])[0])
        draws[label] = (r2_rows(y, predictions[high], rows)
                        - r2_rows(y, predictions[low], rows))
    deviations = np.vstack([draws[key] - observed[key] for key in contrasts])
    maximum = np.max(np.abs(deviations), axis=0)
    critical = float(np.quantile(maximum, 0.95))
    output = {}
    for label in contrasts:
        point = observed[label]
        uncorrected = np.quantile(draws[label], [0.025, 0.975])
        simultaneous = [point - critical, point + critical]
        output[label] = {
            "point": point,
            "uncorrected_ci95": [float(uncorrected[0]), float(uncorrected[1])],
            "simultaneous_ci95": [float(simultaneous[0]), float(simultaneous[1])],
            "simultaneous_excludes_zero": bool(abs(point) > critical),
            "familywise_p": float(np.mean(maximum >= abs(point))),
        }
    return {"family": name, "size": len(contrasts),
            "critical_value": critical, "contrasts": output}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        required = ["y", "model_reve", "model_labram", "model_cbramod",
                    "baseline_adjusted"]
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError("missing arrays: " + ", ".join(missing))
        y = np.asarray(data["y"], dtype=np.float64)
        predictions = {name: np.asarray(data[name], dtype=np.float64)
                       for name in data.files if name != "y"}
    n = len(y)
    if y.ndim != 1 or n < 3:
        raise ValueError("y must be one-dimensional with at least three records")
    if any(x.ndim != 1 or len(x) != n for x in predictions.values()):
        raise ValueError("all prediction arrays must be one-dimensional and match y")
    if not np.isfinite(np.concatenate([y, *predictions.values()])).all():
        raise ValueError("input arrays contain non-finite values")
    rows = np.random.default_rng(args.seed).integers(0, n, (args.bootstrap, n))
    adjusted = {f"adjusted: {model} minus band power":
                (f"model_{model}", "baseline_adjusted")
                for model in ("reve", "labram", "cbramod")}
    report = {
        "n": n, "bootstrap": args.bootstrap, "seed": args.seed,
        "note": "fixed predictions, paired max-statistic bootstrap, no refitting uncertainty",
        "families": [correct_family("adjusted_models", adjusted, y, predictions, rows)],
    }
    if all(name in predictions for name in ("baseline_unadjusted", "model_reve_unadjusted",
                                             "model_labram_unadjusted", "model_cbramod_unadjusted")):
        contrasts = dict(adjusted)
        contrasts.update({f"unadjusted: {model} minus band power":
                          (f"model_{model}_unadjusted", "baseline_unadjusted")
                          for model in ("reve", "labram", "cbramod")})
        report["families"].append(correct_family("adjusted_plus_unadjusted", contrasts,
                                                   y, predictions, rows))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
