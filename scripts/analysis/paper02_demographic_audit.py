"""Demographic and cohort-composition audit for revised Paper 02 targets.

This analysis is descriptive and associational.  It reports missingness before
modeling, summarizes age/sex/site by diagnosis, and estimates standardized OLS
associations with subject-mean DFA using a stratified subject bootstrap.
Medication information is not available in the current derived metadata and is
reported as unavailable rather than assumed absent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.io_safety import safe_np_load  # noqa: E402


SEED = 20260729


def _subject_target(exponent: np.ndarray) -> np.ndarray:
    count = np.isfinite(exponent).sum(axis=1)
    return np.divide(
        np.nansum(exponent, axis=1),
        count,
        out=np.full(len(exponent), np.nan),
        where=count > 0,
    )


def _normalize_sex(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"m", "male", "1"}:
        return "male"
    if normalized in {"f", "female", "2", "0"}:
        return "female"
    return "unknown"


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "mean": None, "sd": None, "median": None, "iqr": None}
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "median": float(np.median(values)),
        "iqr": [float(x) for x in np.percentile(values, [25, 75])],
    }


def _design_matrix(
    label: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    site: np.ndarray,
    alpha_snr: np.ndarray,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    complete = np.ones(len(label), dtype=bool)
    columns = [np.ones(len(label)), label.astype(float)]
    names = ["intercept", "disease_label"]

    if np.isfinite(age).mean() >= 0.80:
        complete &= np.isfinite(age)
        age_z = (age - np.nanmean(age)) / np.nanstd(age)
        columns.append(age_z)
        names.append("age_z")

    known_sex = sex != "unknown"
    if known_sex.mean() >= 0.80 and len(np.unique(sex[known_sex])) == 2:
        complete &= known_sex
        columns.append((sex == "male").astype(float))
        names.append("male_vs_female")

    site_levels = sorted(np.unique(site))
    if len(site_levels) > 1:
        for level in site_levels[1:]:
            columns.append((site == level).astype(float))
            names.append(f"site_{level}_vs_{site_levels[0]}")

    if np.isfinite(alpha_snr).mean() >= 0.80:
        complete &= np.isfinite(alpha_snr) & (alpha_snr > 0)
        log_snr = np.log(alpha_snr)
        log_snr_z = (log_snr - np.nanmean(log_snr)) / np.nanstd(log_snr)
        columns.append(log_snr_z)
        names.append("log_alpha_peak_snr_z")

    return np.column_stack(columns), names, complete


def _bootstrap_ols(
    target: np.ndarray,
    design: np.ndarray,
    names: list[str],
    complete: np.ndarray,
    strata: np.ndarray,
    n_bootstrap: int,
) -> dict:
    complete &= np.isfinite(target)
    y = target[complete]
    x = design[complete]
    strata = strata[complete]
    y = (y - y.mean()) / y.std(ddof=0)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]

    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(strata == level) for level in np.unique(strata)]
    draws = []
    for _ in range(n_bootstrap):
        sample = np.concatenate(
            [rng.choice(group, len(group), replace=True) for group in groups]
        )
        if np.linalg.matrix_rank(x[sample]) < x.shape[1]:
            continue
        draws.append(np.linalg.lstsq(x[sample], y[sample], rcond=None)[0])
    draws = np.asarray(draws)
    return {
        "n_complete": len(y),
        "outcome": "subject-mean DFA standardized within cohort",
        "coefficients": {
            name: {
                "beta": float(beta[index]),
                "ci95": [
                    float(np.percentile(draws[:, index], 2.5)),
                    float(np.percentile(draws[:, index], 97.5)),
                ],
            }
            for index, name in enumerate(names)
        },
        "n_bootstrap": len(draws),
        "scope": (
            "Associational adjustment only; coefficients do not identify "
            "causal disease effects."
        ),
    }


def audit(cohort: str, n_bootstrap: int) -> Path:
    source = ROOT / f"results/paper02/dfa_v2_{cohort}.npz"
    data = safe_np_load(str(source))
    eligible = np.asarray(data["eligible"], dtype=bool)
    label = np.asarray(data["label"], dtype=int)[eligible]
    diagnosis = np.asarray(data["diagnosis"]).astype(str)[eligible]
    site = np.asarray(data["site"]).astype(str)[eligible]
    age = np.asarray(data["age"], dtype=float)[eligible]
    sex = np.asarray([_normalize_sex(value) for value in data["sex"]])[eligible]
    target = _subject_target(np.asarray(data["exponent"], dtype=float))[eligible]
    alpha_snr = np.nanmedian(
        np.asarray(data["alpha_peak_snr"], dtype=float)[eligible],
        axis=1,
    )

    groups = {}
    for group in sorted(np.unique(diagnosis)):
        mask = diagnosis == group
        groups[group] = {
            "n": int(mask.sum()),
            "age": _summary(age[mask]),
            "sex_counts": {
                level: int(np.sum(sex[mask] == level))
                for level in ("female", "male", "unknown")
            },
            "site_counts": {
                level: int(np.sum(site[mask] == level))
                for level in sorted(np.unique(site))
            },
            "dfa": _summary(target[mask]),
            "alpha_peak_snr": _summary(alpha_snr[mask]),
        }

    design, names, complete = _design_matrix(label, age, sex, site, alpha_snr)
    strata = np.asarray([f"{lab}|{st}" for lab, st in zip(label, site)])
    result = {
        "meta": {
            "cohort": cohort,
            "n_eligible": len(target),
            "source": str(source.relative_to(ROOT)),
            "medication": "unavailable in current derived metadata",
            "status": "descriptive confound audit",
        },
        "missingness": {
            "age_n_missing": int(np.sum(~np.isfinite(age))),
            "sex_n_unknown": int(np.sum(sex == "unknown")),
            "dfa_n_missing": int(np.sum(~np.isfinite(target))),
            "alpha_peak_snr_n_missing": int(np.sum(~np.isfinite(alpha_snr))),
        },
        "by_diagnosis": groups,
        "adjusted_associations": _bootstrap_ols(
            target,
            design,
            names,
            complete,
            strata,
            n_bootstrap,
        ),
    }
    output = ROOT / f"results/paper02/demographic_audit_{cohort}.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("brainlat", "caueeg"), required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()
    audit(args.cohort, args.n_bootstrap)


if __name__ == "__main__":
    main()
