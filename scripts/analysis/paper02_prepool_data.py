"""Load the fixed Paper-02 v2 target and pre-pool sequence caches.

This module contains data plumbing only.  It intentionally performs no
hypothesis test and makes no claim that chronological order isolates LRTC.
The sequence caches are generated locally from third-party EEG and frozen
model weights; neither data nor derived embeddings are distributed here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import paper02_primary_encoding_v2 as pe  # noqa: E402
from src.io_safety import safe_np_load  # noqa: E402


FRONT_END = {
    "REVE": "raw-waveform",
    "LaBraM": "raw-patch",
    "BENDR": "raw-wav2vec",
    "CBraMod": "patch-attention",
    "BIOT": "STFT-token",
}


def _v2_targets() -> dict[str, float]:
    """Return eligible CAUEEG subject-level v2 DFA targets by serial."""

    target = pe._load_target("caueeg")
    eligible = target["eligible"]
    serials = target["serial"][eligible]
    values = target["target"][eligible]
    return {
        serial: float(value)
        for serial, value in zip(serials, values)
        if np.isfinite(value)
    }


def _load_sequences() -> dict[str, dict[str, np.ndarray]]:
    """Return ``model -> serial -> [position, dimension]`` sequences."""

    sequences: dict[str, dict[str, np.ndarray]] = {}
    reve = safe_np_load(
        str(ROOT / "results/cross_population/reve_prepool_tokens.npz")
    )
    reve_serials = [str(value) for value in reve["serials"]]
    sequences["REVE"] = {
        serial: np.asarray(sequence, dtype=np.float32)
        for serial, sequence in zip(reve_serials, np.asarray(reve["seqs"]))
    }

    multifm = safe_np_load(
        str(ROOT / "results/cross_population/multifm_prepool_seqs.npz")
    )["data"].item()
    for model, table in multifm["seqs"].items():
        sequences[model] = {
            str(serial): np.asarray(sequence, dtype=np.float32)
            for serial, sequence in table.items()
        }
    return sequences
