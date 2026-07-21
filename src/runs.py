"""Helpers for entrypoints that opt-in to the optional `run_experiment.py` wrapper.

When a script is invoked via `make run` (or directly through
`scripts/run_experiment.py`), the wrapper sets the `POC_RUN_DIR` env var
and later collects:

  * any file in that directory matching ``*results*.json``    -> metrics_json
  * an optional ``.inputs.jsonl`` (one JSON object per line)  -> data_fingerprints
  * any ``*checkpoint*.pt`` and ``*.png``                     -> checkpoint / figures

These helpers let an entrypoint participate in that contract with one or
two lines, while remaining fully back-compatible when run directly (no
POC_RUN_DIR set => no-op).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable


def get_run_dir() -> Path | None:
    """Return the run directory if POC_RUN_DIR is set, else None."""
    rd = os.environ.get("POC_RUN_DIR")
    if not rd:
        return None
    p = Path(rd)
    p.mkdir(parents=True, exist_ok=True)
    return p


def emit_results_json(payload: dict[str, Any], stem: str) -> Path | None:
    """Mirror a results payload into POC_RUN_DIR as ``<stem>_results.json``.

    No-op when POC_RUN_DIR is unset. Returns the written path (or None).
    The wrapper picks up the first file matching ``*results*.json``, so
    any ``stem`` ending in ``_results`` works.
    """
    run_dir = get_run_dir()
    if run_dir is None:
        return None
    target = (
        run_dir / f"{stem}.json"
        if stem.endswith("_results")
        else run_dir / f"{stem}_results.json"
    )
    target.write_text(json.dumps(payload, indent=2, default=str))
    return target


def mirror_results_file(src: Path, stem: str | None = None) -> Path | None:
    """Copy an existing results JSON into POC_RUN_DIR.

    Use when the script already wrote its canonical metrics file to its
    legacy location and we only want to expose it to the wrapper.
    """
    run_dir = get_run_dir()
    if run_dir is None:
        return None
    src = Path(src)
    if not src.exists():
        return None
    name = (
        f"{stem}.json"
        if stem and stem.endswith("_results")
        else (f"{stem}_results.json" if stem else src.name)
    )
    target = run_dir / name
    shutil.copyfile(src, target)
    return target


def record_inputs(inputs: Iterable[tuple[str | Path, str]]) -> Path | None:
    """Append fingerprintable input paths to ``<run_dir>/.inputs.jsonl``.

    `inputs` is an iterable of ``(path, role)`` pairs. The wrapper hashes
    each path and records the SHA-256 + size in the manifest. Paths that
    don't exist are silently skipped by the wrapper.
    """
    run_dir = get_run_dir()
    if run_dir is None:
        return None
    target = run_dir / ".inputs.jsonl"
    with target.open("a") as f:
        for path, role in inputs:
            f.write(json.dumps({"path": str(path), "role": role}) + "\n")
    return target
