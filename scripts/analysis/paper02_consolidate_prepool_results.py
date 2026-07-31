"""Build the canonical five-model Paper-02 pre-pool evidence artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MODELS = {"REVE", "LaBraM", "BENDR", "CBraMod", "BIOT"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _tail_fraction(row: dict) -> float:
    if "shuffle_reference_tail_fraction_one_sided" in row:
        return float(row["shuffle_reference_tail_fraction_one_sided"])
    return float(row["permutation_p_one_sided"])


def tail_fraction_ledger(rows: dict[str, dict]) -> dict[str, dict]:
    """Rank the tested cells by tail fraction, WITHOUT any threshold decision.

    A step-down procedure such as Holm controls the family-wise error rate only when
    its inputs are valid p-values for a predeclared family.  Neither condition holds
    here: the tail fractions are uncalibrated for real chronological EEG (epochs are
    not exchangeable), and the cell set was assembled adaptively.  Applying Holm would
    grant these quantities an inferential status the artifact explicitly disclaims, so
    the ledger reports ordered magnitudes only.
    """
    ranked = sorted(
        ((name, _tail_fraction(row)) for name, row in rows.items()),
        key=lambda item: item[1],
    )
    return {
        name: {"tail_fraction": round(value, 4), "rank_by_tail_fraction": rank + 1}
        for rank, (name, value) in enumerate(ranked)
    }


def _validate_source(source: dict, path: Path) -> None:
    meta = source.get("meta", {})
    if meta.get("n_permutations") != 150 or meta.get("n_configs") != 4:
        raise ValueError(f"{path} is not the 4-configuration, K=150 matched run")
    reference_description = meta.get("null", meta.get("shuffle_reference", ""))
    if "one realization is evaluated under every" not in reference_description.lower():
        raise ValueError(f"{path} does not describe the matched-realization reference")


def _record_legacy_config_seeds(rows: dict[str, dict]) -> None:
    """Materialize the legacy script's ``range(n_configs)`` default in each row."""
    for row in rows.values():
        if "config_seeds" not in row:
            row["config_seeds"] = list(range(int(row["n_configs"])))


def _normalise_shuffle_fields(rows: dict[str, dict]) -> None:
    """Replace inferential legacy labels with descriptive shuffle terminology."""
    for row in rows.values():
        if "permutation_p_one_sided" in row:
            row["shuffle_reference_tail_fraction_one_sided"] = row.pop(
                "permutation_p_one_sided"
            )
        if "min_attainable_p" in row:
            row["min_attainable_tail_fraction"] = row.pop("min_attainable_p")
        row.pop("significant_at_0.05", None)
        row.pop("survives_bonferroni_5_models_0.01", None)


def _combine_configuration_sensitivity(
    primary_source: dict,
    extension_source: dict,
) -> dict[str, dict]:
    """Combine configs 0--3 and 4--11 over the shared first 30 permutations."""
    combined: dict[str, dict] = {}
    for batch_key, base_section, extension_section in (
        ("batch_128", primary_source["primary"], extension_source["primary"]),
        ("batch_32", primary_source["sensitivity"], extension_source["sensitivity"]),
    ):
        combined[batch_key] = {}
        for model in ("CBraMod", "BIOT"):
            base = base_section[model]
            extension = extension_section[model]
            if extension.get("config_seeds") != list(range(4, 12)):
                raise ValueError(
                    f"{model} {batch_key} extension must contain config seeds 4--11"
                )
            if extension["n_permutations"] != 30:
                raise ValueError(f"{model} {batch_key} extension must use K=30")

            # Verifiable compatibility invariants. NOTE: subject identity/order
            # CANNOT be checked here because the run artifacts store neither
            # subject ids nor an ordered-subject hash. Future runs should emit
            # `subject_ids_sha256`; until then this validates only the
            # dimensions and metadata actually recorded, and must not be
            # described as verifying subject order.
            for field in ("n", "seq_len", "batch_size"):
                if base.get(field) != extension.get(field):
                    raise ValueError(
                        f"{model} {batch_key}: {field} differs between base "
                        f"({base.get(field)}) and extension ({extension.get(field)})"
                    )
            if base.get("shuffle_seed_formula") != extension.get("shuffle_seed_formula"):
                raise ValueError(
                    f"{model} {batch_key}: shuffle seed formula differs between runs"
                )
            if base.get("config_seeds") != list(range(4)):
                raise ValueError(
                    f"{model} {batch_key} base must contain config seeds 0--3"
                )
            if len(base["null_matrix_per_config"]) < 30:
                raise ValueError(
                    f"{model} {batch_key} base has "
                    f"{len(base['null_matrix_per_config'])} realizations; needs >=30"
                )
            if len(extension["null_matrix_per_config"]) != 30:
                raise ValueError(
                    f"{model} {batch_key} extension must have exactly 30 realizations"
                )
            if any(len(r) != 4 for r in base["null_matrix_per_config"][:30]):
                raise ValueError(f"{model} {batch_key} base rows must have 4 configs")
            if any(len(r) != 8 for r in extension["null_matrix_per_config"]):
                raise ValueError(
                    f"{model} {batch_key} extension rows must have 8 configs"
                )

            ordered = [
                *map(float, base["ordered_r2_per_config"]),
                *map(float, extension["ordered_r2_per_config"]),
            ]
            base_null = base["null_matrix_per_config"][:30]
            extension_null = extension["null_matrix_per_config"]
            null_matrix = [
                [*map(float, base_row), *map(float, extension_row)]
                for base_row, extension_row in zip(base_null, extension_null)
            ]
            null_means = [float(np.mean(row)) for row in null_matrix]
            config_null_means = list(np.mean(np.asarray(null_matrix), axis=0))
            config_gaps = [
                float(observed - null_mean)
                for observed, null_mean in zip(ordered, config_null_means)
            ]
            observed_mean = float(np.mean(ordered))
            null_mean = float(np.mean(null_means))
            p_value = (1 + sum(value >= observed_mean for value in null_means)) / 31
            combined[batch_key][model] = {
                "role": (
                    "descriptive configuration-stability sensitivity; the "
                    "4-configuration K=150 analysis remains primary"
                ),
                "batch_size": int(base["batch_size"]),
                "config_seeds": list(range(12)),
                "n_configs": 12,
                "n_permutations": 30,
                "ordered_r2_per_config": [round(value, 5) for value in ordered],
                "config_null_means": [
                    round(float(value), 5) for value in config_null_means
                ],
                "config_order_increments": [
                    round(value, 5) for value in config_gaps
                ],
                # Each increment subtracts a reference mean estimated from only
                # K=30 shuffles, so it carries Monte Carlo integration error.
                # Report it so sign counts are not read as stability evidence.
                "config_increment_montecarlo_se": [
                    round(float(np.std(np.asarray(null_matrix)[:, i], ddof=1)
                               / np.sqrt(len(null_matrix))), 5)
                    for i in range(len(config_gaps))
                ],
                "positive_increment_configs": int(sum(value > 0 for value in config_gaps)),
                "positive_increment_configs_exceeding_montecarlo_interval": int(
                    sum(
                        gap - 1.96 * float(np.std(np.asarray(null_matrix)[:, i], ddof=1)
                                           / np.sqrt(len(null_matrix))) > 0
                        for i, gap in enumerate(config_gaps)
                    )
                ),
                "sign_count_caveat": (
                    "positive_increment_configs is a finite-K point count. Use "
                    "positive_increment_configs_exceeding_montecarlo_interval "
                    "and config_increment_montecarlo_se before describing "
                    "configuration stability."
                ),
                "ordered_r2_mean": round(observed_mean, 4),
                "shuffled_null_mean": round(null_mean, 4),
                "observed_minus_null_mean": round(observed_mean - null_mean, 4),
                "shuffle_reference_tail_fraction_descriptive": round(
                    float(p_value), 4
                ),
                "increment_median": round(float(np.median(config_gaps)), 4),
                "increment_min": round(float(np.min(config_gaps)), 4),
                "increment_max": round(float(np.max(config_gaps)), 4),
                "interpretation": (
                    "Configuration seeds are algorithmic conditions, not "
                    "independent biological replicates; this extension assesses "
                    "stability and is not a second confirmatory family."
                ),
            }
    return combined


def build(
    primary_path: Path,
    rest_path: Path,
    extension_path: Path | None = None,
) -> dict:
    primary_source = json.loads(primary_path.read_text())
    rest_source = json.loads(rest_path.read_text())
    _validate_source(primary_source, primary_path)
    _validate_source(rest_source, rest_path)

    overlap = set(primary_source["primary"]) & set(rest_source["primary"])
    if overlap:
        raise ValueError(f"duplicate primary models across sources: {sorted(overlap)}")
    primary_rows = {**primary_source["primary"], **rest_source["primary"]}
    _record_legacy_config_seeds(primary_rows)
    _normalise_shuffle_fields(primary_rows)
    if set(primary_rows) != EXPECTED_MODELS:
        raise ValueError(
            f"five-model family mismatch: got {sorted(primary_rows)}, "
            f"expected {sorted(EXPECTED_MODELS)}"
        )

    sensitivity_rows = primary_source.get("sensitivity", {})
    _record_legacy_config_seeds(sensitivity_rows)
    _normalise_shuffle_fields(sensitivity_rows)
    all_tested_cells = {
        **{f"{model}_bs128": row for model, row in primary_rows.items()},
        **{f"{model}_bs32": row for model, row in sensitivity_rows.items()},
    }

    artifact = {
        "meta": {
            "title": (
                "Canonical pre-pool ordered-vs-shuffled reference "
                "result (paper-02)"
            ),
            "supersedes": [
                (
                    "results/paper02/superseded/"
                    "prepool_order_permutation_2cfg_K30_INVALID.json"
                ),
                (
                    "results/paper02/superseded/"
                    "prepool_order_permutation_2cfg_K150_INVALID.json"
                ),
            ],
            "superseded_reason": (
                "different shuffle realizations across configurations changed "
                "the joint reference distribution, making its tail fractions "
                "invalid for the intended statistic; see superseded/MANIFEST.md"
            ),
            "consolidated_from": {
                "primary_cbramod_biot": {
                    "path": str(primary_path.relative_to(ROOT)),
                    "sha256": file_sha256(primary_path),
                },
                "primary_reve_labram_bendr": {
                    "path": str(rest_path.relative_to(ROOT)),
                    "sha256": file_sha256(rest_path),
                },
            },
            "generator": str(Path(__file__).resolve().relative_to(ROOT)),
            "shuffle_reference": (
                "one subject-wise shuffle realization per permutation, evaluated "
                "under every fold/init configuration (seed=1000+k)"
            ),
            "n_configs": 4,
            "config_seeds": [0, 1, 2, 3],
            "config_seed_provenance": (
                "The K=150 constituent artifacts predate the explicit "
                "config_seeds field. Their producing script used "
                "range(n_configs), so the recorded four configurations are "
                "seeds 0--3."
            ),
            "n_permutations": 150,
            "primary_batch_size": 128,
            "tail_fraction_note": (
                "The reported tail fraction is a Monte Carlo estimate at the "
                "K=150 resolution floor (minimum attainable 1/151=0.0066). "
                "It is a calibrated randomization p-value only under the sharp "
                "null that chronological within-subject epoch order is "
                "exchangeable with the sampled permutations. Real EEG violates "
                "that assumption through autocorrelation, non-stationarity, "
                "drift, artifact clustering, and position effects, so the paper "
                "treats it as a descriptive shuffle-reference statistic rather "
                "than an LRTC-specific inferential p-value."
            ),
            "multiplicity_note": (
                "All analyses are post hoc and NO family-wise correction is "
                "applied. A step-down procedure controls the family-wise error "
                "rate only when supplied valid p-values for a predeclared "
                "family; neither condition holds here. The tested cells were "
                "not predeclared: they accumulated through repeated-seed "
                "comparisons, a withdrawn permutation construction, batch-size "
                "sensitivity, model selection, corrected reruns, and a "
                "configuration extension applied only to CBraMod and BIOT. Any "
                "ledger over the surviving cells would therefore omit the "
                "adaptive path that produced them, so this accounting must not "
                "be described as complete. The tail fractions are reported as "
                "ordered magnitudes only."
            ),
            "interpretation_limits": [
                (
                    "Order-sensitivity does not isolate LRTC: non-stationarity, "
                    "arousal drift, artifact clustering and local temporal "
                    "continuity remain alternatives."
                ),
                (
                    "Chronological EEG epochs are not exchangeable. The "
                    "shuffle-reference tail proportion therefore does not test "
                    "a weak null of no LRTC information."
                ),
                (
                    "CAUEEG only; n=199 for the four epoch-sequence models, n=396 "
                    "for REVE. No cross-population replication."
                ),
                "POST HOC: designed after earlier results were seen.",
                (
                    "BIOT is not declared null: batch 128 detects no advantage "
                    "while batch 32 is positive but optimization-sensitive; the "
                    "epoch ratio is a diagnostic, not proof of an artifact."
                ),
                (
                    "The 12-configuration extension gives positive CBraMod "
                    "point increments in 10/12 batch-128 and 12/12 batch-32 "
                    "configurations, but each increment subtracts a reference "
                    "mean estimated from only 30 shuffles. Sign counts are "
                    "therefore finite-K point counts, not evidence that every "
                    "underlying configuration effect is positive; see "
                    "montecarlo_se per configuration. Magnitudes remain "
                    "heterogeneous and the configurations are algorithmic "
                    "conditions, not biological replicates."
                ),
                (
                    "LaBraM is uninformative rather than negative because both "
                    "arms perform below a mean predictor."
                ),
            ],
            "status": (
                "POST HOC exploratory; canonical evidence for the paper-02 "
                "pre-pool control"
            ),
        },
        "primary_bs128": primary_rows,
        "sensitivity_bs32": sensitivity_rows,
        "tail_fraction_ledger_all_7_tested_cells": tail_fraction_ledger(
            all_tested_cells
        ),
    }
    if extension_path is not None:
        extension_source = json.loads(extension_path.read_text())
        artifact["meta"]["configuration_sensitivity_source"] = {
            "path": str(extension_path.relative_to(ROOT)),
            "sha256": file_sha256(extension_path),
        }
        artifact["configuration_sensitivity_12cfg_K30"] = (
            _combine_configuration_sensitivity(primary_source, extension_source)
        )
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--primary",
        type=Path,
        default=ROOT / "results/paper02/prepool_order_permutation_matched.json",
    )
    parser.add_argument(
        "--rest",
        type=Path,
        default=ROOT / "results/paper02/prepool_order_permutation_matched_rest.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/paper02/prepool_order_permutation_5model.json",
    )
    parser.add_argument(
        "--config-extension",
        type=Path,
        help=(
            "optional K=30 artifact for config seeds 4--11; when supplied, "
            "adds the descriptive 12-configuration sensitivity"
        ),
    )
    args = parser.parse_args()
    extension = args.config_extension.resolve() if args.config_extension else None
    artifact = build(args.primary.resolve(), args.rest.resolve(), extension)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"[paper02-consolidate] wrote {args.out}")


if __name__ == "__main__":
    main()
