"""Safe deserialization wrappers for pickle and pickle-backed numpy loads.

Pickle is unsafe in the general case (arbitrary code execution on load).
This module restricts pickle deserialization to a whitelist of trusted
project directories and emits a SHA-256 audit-trail log line for every
load — closing the path-traversal vector and providing FDA-style
provenance without changing on-disk formats.

Usage:
    from src.io_safety import safe_pickle_load, safe_np_load_pickle

    data = safe_pickle_load("data/processed/chbmit_processed.pkl")
    cache = safe_np_load_pickle("data/embeddings/foo.npz")  # context manager
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

logger = logging.getLogger(__name__)

# Project root is two levels above this file (src/io_safety.py → poc/).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories from which pickle deserialization is permitted. Any path
# outside this set is rejected — this prevents loading adversarial
# pickles smuggled in via environment variables, config injection, or
# symlinks pointing into /tmp.
_DEFAULT_ALLOWED_ROOTS: tuple[Path, ...] = (
    _PROJECT_ROOT / "data",
    _PROJECT_ROOT / "results",
    _PROJECT_ROOT / "models",
)


def _resolve_and_validate(
    path: str | Path,
    allowed_roots: tuple[Path, ...] | None = None,
) -> Path:
    """Resolve `path` to an absolute path and check it is under an allowed root.

    Raises:
        FileNotFoundError: path does not exist.
        PermissionError: path is outside all allowed roots.
    """
    roots = allowed_roots if allowed_roots is not None else _DEFAULT_ALLOWED_ROOTS
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Pickle source not found: {resolved}")
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return resolved
        except ValueError:
            continue
    raise PermissionError(
        f"Refusing to deserialize pickle outside allowed roots: {resolved} "
        f"(allowed: {[str(r) for r in roots]})"
    )


def _sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_pickle_load(
    path: str | Path,
    allowed_roots: tuple[Path, ...] | None = None,
) -> Any:
    """Load a pickle file with path validation and SHA-256 audit logging.

    The pickle format itself remains arbitrary-code-execution-capable;
    this wrapper hardens the call by:
      1. Rejecting paths outside the project's data/results/models tree.
      2. Logging a SHA-256 digest of the file (audit trail).

    For untrusted inputs, do not use pickle at all — convert the
    upstream producer to numpy/safetensors/json.
    """
    resolved = _resolve_and_validate(path, allowed_roots)
    digest = _sha256_of_file(resolved)
    logger.info(
        "pickle_load path=%s sha256=%s size_bytes=%d",
        resolved,
        digest,
        resolved.stat().st_size,
    )
    with resolved.open("rb") as f:
        return pickle.load(f)  # noqa: S301 — guarded by path whitelist + audit log


def safe_np_load(
    path: str | Path,
    allow_pickle: bool = True,
    mmap_mode: str | None = None,
    allowed_roots: tuple[Path, ...] | None = None,
) -> Any:
    """Drop-in replacement for `np.load(path, allow_pickle=True)`.

    Performs path whitelisting + SHA-256 audit logging, then defers to
    `numpy.load`. Returns whatever `np.load` returns — an `NpzFile` for
    `.npz` archives or an `ndarray` for `.npy` files. Both forms are
    safe to use as a context manager (`with safe_np_load(...) as f:`)
    or directly (`data = safe_np_load(...)["foo"]`).

    Pickle deserialization is the default for backward compatibility
    with existing call sites, but explicit `allow_pickle=False` is
    honored when the caller can guarantee a no-object cache.
    """
    resolved = _resolve_and_validate(path, allowed_roots)
    digest = _sha256_of_file(resolved)
    logger.info(
        "np_load path=%s sha256=%s allow_pickle=%s mmap=%s size_bytes=%d",
        resolved,
        digest,
        allow_pickle,
        mmap_mode,
        resolved.stat().st_size,
    )
    return np.load(resolved, allow_pickle=allow_pickle, mmap_mode=mmap_mode)


# Back-compat: the context-manager form remains available as a thin
# wrapper around safe_np_load for code that prefers explicit close.
@contextmanager
def safe_np_load_pickle(
    path: str | Path,
    allowed_roots: tuple[Path, ...] | None = None,
    mmap_mode: str | None = None,
) -> Iterator[np.lib.npyio.NpzFile]:
    """Context-manager form of `safe_np_load(..., allow_pickle=True)`."""
    npz = safe_np_load(
        path,
        allow_pickle=True,
        mmap_mode=mmap_mode,
        allowed_roots=allowed_roots,
    )
    try:
        yield npz
    finally:
        if hasattr(npz, "close"):
            npz.close()
