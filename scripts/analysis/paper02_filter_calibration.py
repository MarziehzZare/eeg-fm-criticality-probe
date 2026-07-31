"""White-noise calibration of the Paper 02 alpha-envelope DFA lower bound.

Hardstone et al. recommend determining the filter's temporal integration
effect on matched filtered white noise.  This script simulates 1000 independent
240 s signals, applies the exact 51-tap 8--13 Hz FIR/Hilbert pipeline, and
compares candidate lower fitting bounds while retaining the same <=10%
upper-scale rule.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.lrtc import (  # noqa: E402
    LRTCConfig,
    admissible_scales,
    alpha_envelope,
    dfa,
)


SEED = 20260729
CANDIDATE_MIN_SECONDS = (0.5, 1.0, 1.5, 2.0, 2.5)


def run(n_simulations: int) -> Path:
    base = LRTCConfig()
    n_samples = int(240 * base.sfreq)
    trim = int(round(base.edge_trim_seconds * base.sfreq))
    rng = np.random.default_rng(SEED)
    estimates = {str(value): [] for value in CANDIDATE_MIN_SECONDS}

    for index in range(n_simulations):
        raw = rng.standard_normal(n_samples)
        envelope = alpha_envelope(raw, base)
        envelope = envelope[trim:-trim]
        calibration_config = LRTCConfig(
            fit_min_seconds=min(CANDIDATE_MIN_SECONDS),
            n_scales=30,
            min_valid_scales=8,
        )
        scales = admissible_scales(n_samples, calibration_config)
        result = dfa(
            envelope,
            base.sfreq,
            scales,
            overlap=base.overlap,
            min_clean_fraction_per_window=1.0,
            min_windows_per_scale=base.min_windows_per_scale,
            min_valid_scales=base.min_valid_scales,
        )
        for lower in CANDIDATE_MIN_SECONDS:
            selected = result.scales_seconds >= lower
            slope = np.polyfit(
                np.log(result.scales_seconds[selected]),
                np.log(result.fluctuations[selected]),
                1,
            )[0]
            estimates[str(lower)].append(float(slope))
        if (index + 1) % 100 == 0:
            print(f"simulated {index + 1}/{n_simulations}", flush=True)

    output = {
        "meta": {
            "n_simulations": n_simulations,
            "duration_seconds": 240,
            "seed": SEED,
            "expected_null_exponent": 0.5,
            "filter": "51-tap linear-phase FIR, 8-13 Hz, Hilbert amplitude",
            "upper_bound": "<=10% of post-trim analyzable duration",
            "calculation_scales": (
                "30 common log-spaced scales from 0.5 s to the upper bound; "
                "candidate fits use subsets of the same fluctuation function"
            ),
            "selection_rule": (
                "The primary 2 s lower bound was specified from Hardstone et "
                "al.; simulation quantifies rather than tunes its residual bias."
            ),
        },
        "candidate_lower_bounds": {},
    }
    for lower, values in estimates.items():
        values = np.asarray(values, dtype=float)
        standard_error = np.std(values, ddof=1) / np.sqrt(len(values))
        output["candidate_lower_bounds"][lower] = {
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)),
            "mean_ci95_normal": [
                float(np.mean(values) - 1.96 * standard_error),
                float(np.mean(values) + 1.96 * standard_error),
            ],
            "individual_interval_95": [
                float(x) for x in np.percentile(values, [2.5, 97.5])
            ],
            "bias_from_0.5": float(np.mean(values) - 0.5),
        }

    path = ROOT / "results/paper02/filter_calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2), flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-simulations", type=int, default=1000)
    args = parser.parse_args()
    run(args.n_simulations)


if __name__ == "__main__":
    main()
