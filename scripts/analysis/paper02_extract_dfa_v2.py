"""Extract the fixed revised Paper 02 alpha-envelope DFA target.

This script replaces the scattered legacy Butterworth/0.5--30 s target
implementations with :mod:`src.lrtc`.  It applies the same 240 s estimand to
CAUEEG, BrainLat, and ds004504 whenever the corresponding raw data are present.

Every subject is written to the output, including subjects that fail automatic
QC.  The ``eligible`` field is an explicit analysis flag, not an implicit
deletion:

* at least 15/19 channels return a finite estimate;
* median retained-sample fraction is at least 0.80; and
* median log-log DFA fit R2 across finite channels is at least 0.95.

Automatic artifact detection is an initial reproducible pass.  Before a final
confirmatory analysis, rejected recordings and a random sample of retained
recordings should also receive blinded visual review.

Usage
-----
    python scripts/analysis/paper02_extract_dfa_v2.py --cohort brainlat
    python scripts/analysis/paper02_extract_dfa_v2.py --cohort caueeg
    python scripts/analysis/paper02_extract_dfa_v2.py --cohort all
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.lrtc import (  # noqa: E402
    LRTCConfig,
    alpha_peak_snr,
    alpha_envelope,
    estimate_envelope_dfa,
    interpolate_masked_samples,
    robust_artifact_mask,
)


COMMON = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "Fz",
    "Cz",
    "Pz",
]
CAUEEG_ROOT = ROOT / "data/raw/caueeg-dataset"
BRAINLAT_ROOT = ROOT / "data/raw/brainlat"
DS004504_ROOT = ROOT / "data/raw/ds004504"
OUTPUT_ROOT = ROOT / "results/paper02"
N_SECONDS = 240.0


def _norm_channel(name: str) -> str:
    return (
        name.upper()
        .replace("-AVG", "")
        .replace("-REF", "")
        .replace("EEG ", "")
        .strip()
    )


def _common_channel_order(names: list[str]) -> list[int] | None:
    canonical = {channel.upper(): channel for channel in COMMON}
    found: dict[str, int] = {}
    for index, name in enumerate(names):
        normalized = _norm_channel(name)
        if normalized in canonical:
            found[canonical[normalized]] = index
    return [found[channel] for channel in COMMON] if len(found) == len(COMMON) else None


def _brainlat_channel_order() -> list[str]:
    """Map standard 10-20 channels to nearest BioSemi-128 positions."""

    import mne

    biosemi = mne.channels.make_standard_montage("biosemi128").get_positions()["ch_pos"]
    standard = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    renamed = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    return [
        min(
            biosemi,
            key=lambda candidate: float(
                np.linalg.norm(
                    biosemi[candidate] - standard[renamed.get(channel, channel)]
                )
            ),
        )
        for channel in COMMON
    ]


def _prepare_data(raw, indices: list[int], config: LRTCConfig) -> tuple[np.ndarray, np.ndarray]:
    """Resample, crop, detect artifacts, interpolate, and common-average reference."""

    if raw.info["sfreq"] != config.sfreq:
        raw.resample(config.sfreq, verbose="ERROR")
    required = int(round(N_SECONDS * config.sfreq))
    data = raw.get_data()[indices, :required]
    if data.shape != (len(COMMON), required):
        raise ValueError(
            f"need {len(COMMON)} channels x {required} samples; got {data.shape}"
        )

    clean = robust_artifact_mask(data, config.sfreq)
    interpolated = interpolate_masked_samples(data, clean)
    car = interpolated - interpolated.mean(axis=0, keepdims=True)
    return car, clean


def _iter_caueeg():
    import mne

    manifest_path = CAUEEG_ROOT / "dementia.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    items = [
        item
        for split in ("train_split", "validation_split", "test_split")
        for item in manifest[split]
        if item["class_name"] in ("Normal", "Dementia")
    ]
    for item in items:
        serial = item["serial"]
        path = CAUEEG_ROOT / "signal/edf" / f"{serial}.edf"
        if not path.exists():
            continue
        metadata = {
            "label": 1 if item["class_name"] == "Dementia" else 0,
            "diagnosis": item["class_name"],
            "site": "CAUEEG",
            "age": item.get("age", np.nan),
            "sex": item.get("sex", item.get("gender", "")),
        }
        try:
            raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
            raw.crop(tmax=min(N_SECONDS, (raw.n_times - 1) / raw.info["sfreq"]))
            raw.load_data(verbose="ERROR")
        except Exception as error:
            yield serial, None, {
                **metadata,
                "error": f"{type(error).__name__}: {error}",
            }
            continue
        order = _common_channel_order(raw.ch_names)
        if order is None:
            yield serial, None, {
                **metadata,
                "error": "missing common channels",
            }
            continue
        yield serial, (raw, order), metadata


def _iter_brainlat():
    import mne

    groups = {"1_AD": (1, "AD"), "2_bvFTD": (1, "bvFTD"), "5_HC": (0, "HC")}
    biosemi_names = _brainlat_channel_order()
    for path_string in sorted(
        glob.glob(str(BRAINLAT_ROOT / "**/*_eeg.set"), recursive=True)
    ):
        path = Path(path_string)
        group = next((part for part in path.parts if part in groups), None)
        if group is None:
            continue
        serial = next(
            (part for part in path.parts if part.startswith("sub-")),
            path.stem,
        )
        site = next((part for part in path.parts if part in ("AR", "CL")), "unknown")
        label, diagnosis = groups[group]
        metadata = {
            "label": label,
            "diagnosis": diagnosis,
            "site": site,
            "age": np.nan,
            "sex": "",
        }
        try:
            raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
        except Exception as error:
            yield serial, None, {
                **metadata,
                "error": f"{type(error).__name__}: {error}",
            }
            continue
        channel_index = {name: index for index, name in enumerate(raw.ch_names)}
        if any(name not in channel_index for name in biosemi_names):
            yield serial, None, {
                **metadata,
                "error": "missing mapped BioSemi channels",
            }
            continue
        yield serial, (raw, [channel_index[name] for name in biosemi_names]), metadata


def _iter_ds004504():
    import mne
    import pandas as pd

    participants_path = DS004504_ROOT / "participants.tsv"
    participants = (
        pd.read_csv(participants_path, sep="\t").set_index("participant_id")
        if participants_path.exists()
        else None
    )
    paths = sorted(
        glob.glob(
            str(DS004504_ROOT / "sub-*/eeg/*task-eyesclosed_eeg.set")
        )
    )
    for path_string in paths:
        path = Path(path_string)
        serial = path.name.split("_")[0]
        row = participants.loc[serial] if participants is not None else {}
        diagnosis = str(
            row.get("Group", row.get("group", row.get("diagnosis", "unknown")))
        )
        diagnosis_upper = diagnosis.upper()
        if diagnosis_upper in {"C", "HC", "CONTROL", "HEALTHY"}:
            label = 0
        elif diagnosis_upper in {
            "A",
            "AD",
            "ALZHEIMER",
            "F",
            "FTD",
            "BVFTD",
        }:
            label = 1
        else:
            label = -1
        metadata = {
            "label": label,
            "diagnosis": diagnosis,
            "site": "ds004504",
            "age": row.get("Age", row.get("age", np.nan)),
            "sex": row.get("Gender", row.get("sex", "")),
        }
        if label < 0:
            metadata["error"] = f"unrecognized diagnosis label: {diagnosis}"
            yield serial, None, metadata
            continue
        try:
            raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
        except Exception as error:
            yield serial, None, {
                **metadata,
                "error": f"{type(error).__name__}: {error}",
            }
            continue
        order = _common_channel_order(raw.ch_names)
        if order is None:
            metadata["error"] = "missing common channels"
            yield serial, None, metadata
        else:
            yield serial, (raw, order), metadata


ITERATORS = {
    "caueeg": _iter_caueeg,
    "brainlat": _iter_brainlat,
    "ds004504": _iter_ds004504,
}


def _subject_eligible(
    exponent: np.ndarray,
    fit_r2: np.ndarray,
    retained_fraction: np.ndarray,
) -> bool:
    finite = np.isfinite(exponent) & np.isfinite(fit_r2)
    return bool(
        finite.sum() >= 15
        and np.nanmedian(retained_fraction) >= 0.80
        and np.nanmedian(fit_r2[finite]) >= 0.95
    )


def extract_cohort(
    cohort: str,
    *,
    max_subjects: int | None = None,
    config: LRTCConfig | None = None,
    split_half: bool = False,
) -> Path:
    config = config or LRTCConfig()
    records: list[dict] = []
    iterator = ITERATORS[cohort]()
    for index, (serial, payload, metadata) in enumerate(iterator):
        record = {"serial": serial, **metadata}
        try:
            if payload is None:
                raise ValueError(record.get("error", "unreadable input"))
            raw, order = payload
            data, clean = _prepare_data(raw, order, config)
            envelope = alpha_envelope(data, config, clean_mask=clean)
            result = estimate_envelope_dfa(envelope, config, clean_mask=clean)
            spectral_snr = alpha_peak_snr(data, config.sfreq)
            record.update(
                exponent=result.exponent,
                fit_r2=result.fit_r2,
                retained_fraction=result.clean_sample_fraction,
                n_valid_scales=result.n_valid_scales,
                alpha_peak_snr=spectral_snr,
                eligible=_subject_eligible(
                    result.exponent,
                    result.fit_r2,
                    result.clean_sample_fraction,
                ),
                error="",
            )
            if split_half:
                midpoint = data.shape[1] // 2
                first = estimate_envelope_dfa(
                    envelope[:, :midpoint],
                    config,
                    clean_mask=clean[:, :midpoint],
                )
                second = estimate_envelope_dfa(
                    envelope[:, midpoint:],
                    config,
                    clean_mask=clean[:, midpoint:],
                )
                record.update(
                    half1_exponent=first.exponent,
                    half2_exponent=second.exponent,
                )
        except Exception as error:
            record.update(
                exponent=np.full(len(COMMON), np.nan),
                fit_r2=np.full(len(COMMON), np.nan),
                retained_fraction=np.zeros(len(COMMON)),
                n_valid_scales=np.zeros(len(COMMON), dtype=int),
                alpha_peak_snr=np.full(len(COMMON), np.nan),
                eligible=False,
                error=f"{type(error).__name__}: {error}",
            )
            if split_half:
                record.update(
                    half1_exponent=np.full(len(COMMON), np.nan),
                    half2_exponent=np.full(len(COMMON), np.nan),
                )
        records.append(record)
        if len(records) % 10 == 0:
            print(
                f"[{cohort}] processed {len(records)}; "
                f"eligible={sum(item['eligible'] for item in records)}",
                flush=True,
            )
        if max_subjects is not None and index + 1 >= max_subjects:
            break

    if not records:
        raise FileNotFoundError(f"no raw records found for {cohort}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_ROOT / f"dfa_v2_{cohort}.npz"
    np.savez_compressed(
        output,
        serial=np.asarray([item["serial"] for item in records]),
        label=np.asarray([item["label"] for item in records], dtype=int),
        diagnosis=np.asarray([item["diagnosis"] for item in records]),
        site=np.asarray([item["site"] for item in records]),
        age=np.asarray([item["age"] for item in records], dtype=float),
        sex=np.asarray([item["sex"] for item in records]),
        exponent=np.vstack([item["exponent"] for item in records]),
        fit_r2=np.vstack([item["fit_r2"] for item in records]),
        retained_fraction=np.vstack(
            [item["retained_fraction"] for item in records]
        ),
        n_valid_scales=np.vstack([item["n_valid_scales"] for item in records]),
        alpha_peak_snr=np.vstack([item["alpha_peak_snr"] for item in records]),
        eligible=np.asarray([item["eligible"] for item in records], dtype=bool),
        error=np.asarray([item["error"] for item in records]),
        channels=np.asarray(COMMON),
        config=np.asarray(asdict(config), dtype=object),
        **(
            {
                "half1_exponent": np.vstack(
                    [item["half1_exponent"] for item in records]
                ),
                "half2_exponent": np.vstack(
                    [item["half2_exponent"] for item in records]
                ),
            }
            if split_half
            else {}
        ),
    )

    eligible = np.asarray([item["eligible"] for item in records], dtype=bool)
    exponent = np.vstack([item["exponent"] for item in records])
    fit_r2 = np.vstack([item["fit_r2"] for item in records])
    retained = np.vstack([item["retained_fraction"] for item in records])
    spectral_snr = np.vstack([item["alpha_peak_snr"] for item in records])
    summary = {
        "cohort": cohort,
        "n_total": len(records),
        "n_eligible": int(eligible.sum()),
        "n_failed": int(sum(bool(item["error"]) for item in records)),
        "class_counts_eligible": {
            str(label): int(
                sum(item["eligible"] and item["label"] == label for item in records)
            )
            for label in (0, 1)
        },
        "median_exponent_eligible": (
            float(np.nanmedian(exponent[eligible])) if eligible.any() else None
        ),
        "median_fit_r2_eligible": (
            float(np.nanmedian(fit_r2[eligible])) if eligible.any() else None
        ),
        "median_retained_fraction_eligible": (
            float(np.nanmedian(retained[eligible])) if eligible.any() else None
        ),
        "median_alpha_peak_snr_eligible": (
            float(np.nanmedian(spectral_snr[eligible])) if eligible.any() else None
        ),
        "config": asdict(config),
        "output": str(output.relative_to(ROOT)),
        "status": (
            "automatic QC complete; blinded visual review required before "
            "final inference"
        ),
    }
    if split_half:
        half1 = np.nanmean(
            np.vstack([item["half1_exponent"] for item in records]), axis=1
        )
        half2 = np.nanmean(
            np.vstack([item["half2_exponent"] for item in records]), axis=1
        )
        reliable = eligible & np.isfinite(half1) & np.isfinite(half2)
        half_r = (
            float(np.corrcoef(half1[reliable], half2[reliable])[0, 1])
            if reliable.sum() >= 3
            else None
        )
        summary["split_half"] = {
            "n": int(reliable.sum()),
            "pearson_r": half_r,
            "half_to_half_r2": half_r**2 if half_r is not None else None,
            "spearman_brown": (
                2 * half_r / (1 + half_r)
                if half_r is not None and half_r > -1
                else None
            ),
            "important_scope": (
                "Each 120 s half uses 2 s to <=10% of its own analyzable "
                "duration (~11.8 s); this assesses short-to-mid-scale target "
                "repeatability, not an exact reliability ceiling for the "
                "full 2-23.8 s estimand."
            ),
        }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort",
        choices=[*ITERATORS, "all"],
        required=True,
    )
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--split-half", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)

    cohorts = list(ITERATORS) if args.cohort == "all" else [args.cohort]
    failures = {}
    for cohort in cohorts:
        try:
            extract_cohort(
                cohort,
                max_subjects=args.max_subjects,
                split_half=args.split_half,
            )
        except FileNotFoundError as error:
            failures[cohort] = str(error)
            print(f"[{cohort}] unavailable: {error}", flush=True)
    if failures and args.cohort != "all":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
