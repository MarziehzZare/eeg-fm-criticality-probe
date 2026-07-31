"""Reproducibility seed helper for training scripts.

Any script that calls `loss.backward()` (i.e., trains weights — LoRA,
linear probe, MLP probe, full fine-tune) must call `set_seed()` at the
top of `main()` so that re-runs yield deterministic numbers and the
results JSON / claim registry remains reproducible.

This requirement is enforced by `scripts/ci/check_unsafe_pickle.py`
(rule F3) — adding `.backward()` without importing `set_seed` will
fail `make check`.

Default seed is 42, matching the project convention.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int | None = None) -> int:
    """Seed Python, NumPy, and PyTorch (CPU + accelerator) RNGs.

    If `seed` is None, reads the `SEED` environment variable (default 42).
    Returns the seed actually used so callers can record it in manifests.

    Notes:
      - cuDNN deterministic + benchmark=False is set so conv kernels
        don't pick non-deterministic implementations on CUDA.
      - MPS (Apple Silicon) does not yet expose a deterministic mode;
        we seed `torch.mps.manual_seed` and accept residual non-determinism
        for ops without deterministic kernels.
      - PYTHONHASHSEED is set so hash-based shuffles in stdlib are stable.
    """
    if seed is None:
        seed = int(os.environ.get("SEED", "42"))
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # torch.mps.manual_seed is the canonical entry point; fall back
        # to torch.manual_seed (which seeds MPS implicitly) if unavailable.
        if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
            torch.mps.manual_seed(seed)
    return seed
