"""REVE-Base wrapper for NeuroGenis PoC.

REVE (El Ouahidi et al., NeurIPS 2025) — 69.4M params, pretrained on
60,000+ hours of EEG from 92 datasets and 25,000 subjects.
4D positional encoding natively handles any electrode configuration.

HuggingFace: brain-bzh/reve-base
Paper: https://arxiv.org/abs/2510.21585

Usage:
    wrapper = REVEWrapper.load(ch_names=["Fp1", "F3", ...])
    embeddings = wrapper.extract(epochs)  # (n_epochs, n_ch, 800) -> (n_epochs, d)
    wrapper.free()
"""

import logging
from pathlib import Path
from typing import Optional

import mne
import numpy as np
import torch

from src.embeddings import capture_hook

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# HuggingFace model ID — single source of truth
REVE_HUB_ID = "brain-bzh/reve-base"

# Standard 10-20 channel ordering used by TUH/CHB-MIT.
# Used as default when ch_names is not provided explicitly.
REVE_STANDARD_CH_21: list[str] = [
    "Fp1", "Fp2", "F3",  "F4",  "C3",  "C4",  "P3",  "P4",
    "O1",  "O2",  "F7",  "F8",  "T3",  "T4",  "T5",  "T6",
    "Fz",  "Cz",  "Pz",  "A1",  "A2",
]

# Alias map: legacy 10-20 names → MNE standard_1020 montage names
_CH_ALIAS: dict[str, str] = {
    "T3": "T7", "T4": "T8",
    "T5": "P7", "T6": "P8",
    "A1": "M1", "A2": "M2",
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _resolve_ch_name(ch: str, pos_map: dict) -> str:
    """Resolve a channel name to one REVE's position bank recognises.

    Handles three cases in order:
      1. Exact match after legacy alias mapping (T3→T7, etc.)
      2. Compound differential names (e.g. 'Fpz-Cz' → try 'Fpz', then 'Cz')
      3. Returns the original name unchanged if nothing matches (REVE will
         fall back to a learned default position for unknown channels).
    """
    # 1. Legacy alias
    candidate = _CH_ALIAS.get(ch, ch)
    if candidate in pos_map:
        return candidate

    # 2. Compound differential: 'Fpz-Cz' → try active electrode first
    if "-" in ch:
        parts = ch.split("-")
        for part in parts:
            alias = _CH_ALIAS.get(part, part)
            if alias in pos_map:
                logger.debug(f"REVE: compound channel '{ch}' resolved to '{alias}'")
                return alias

    # 3. No match
    return ch


def _build_chs_info(ch_names: list[str]) -> list[dict]:
    """Build chs_info list with standard 10-20 positions for REVE (braindecode API)."""
    montage = mne.channels.make_standard_montage("standard_1020")
    pos_map = montage.get_positions()["ch_pos"]

    chs_info = []
    for ch in ch_names:
        resolved = _resolve_ch_name(ch, pos_map)
        if resolved in pos_map:
            chs_info.append({"ch_name": resolved, "loc": list(pos_map[resolved]) + [0] * 9})
        else:
            logger.debug(f"REVE: channel '{ch}' not in standard_1020 montage — omitting loc")
            chs_info.append({"ch_name": ch})

    return chs_info


def _build_pos_tensor(ch_names: list[str]) -> torch.Tensor:
    """Build (1, n_channels, 3) electrode position tensor for the HuggingFace REVE forward pass.

    Uses MNE standard_1020 montage. Legacy aliases (T3→T7 etc.) are resolved.
    Unknown channels get position (0, 0, 0).
    """
    montage = mne.channels.make_standard_montage("standard_1020")
    pos_map = montage.get_positions()["ch_pos"]

    coords = []
    for ch in ch_names:
        resolved = _resolve_ch_name(ch, pos_map)
        if resolved in pos_map:
            coords.append(pos_map[resolved])
        else:
            logger.debug(f"REVE pos: '{ch}' unknown — using (0,0,0)")
            coords.append([0.0, 0.0, 0.0])

    pos = torch.from_numpy(np.array(coords, dtype=np.float32)).unsqueeze(0)  # (1, n_ch, 3)
    return pos


def _select_device(preferred: Optional[str] = None) -> torch.device:
    """Return the best available device.

    Args:
        preferred: 'mps', 'cuda', or 'cpu'. If None, auto-selects
                   MPS > CUDA > CPU in that order.
    """
    if preferred is not None:
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _pool_features(feat: torch.Tensor) -> torch.Tensor:
    """Mean-pool REVE output to a 1-D embedding per sample.

    REVE can return different shapes depending on the braindecode version:
        [B, T, D]       — mean over T  → [B, D]
        [B, C, T, D]    — mean over C, T → [B, D]
        [B, ...]        — flatten       → [B, -1]
    """
    if feat.dim() == 3:
        return feat.mean(dim=1)
    if feat.dim() == 4:
        return feat.mean(dim=(1, 2))
    # Fallback: flatten all non-batch dims
    return feat.reshape(feat.size(0), -1)


# ── Public API ────────────────────────────────────────────────────────────────

class REVEWrapper:
    """Thin wrapper around REVE-Base for frozen embedding extraction.

    Attributes:
        model:      REVE nn.Module in eval mode with frozen weights.
        device:     torch.device the model lives on.
        ch_names:   Channel names the model was instantiated with.
        n_times:    Epoch length in samples (default 800 = 4s at 200 Hz).
        sfreq:      Sampling frequency (default 200 Hz).
        batch_size: Default inference batch size.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        ch_names: list[str],
        n_times: int,
        sfreq: int,
        batch_size: int,
        pos_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        self.model = model
        self.device = device
        self.ch_names = ch_names
        self.pos_tensor = pos_tensor  # (1, n_ch, 3) electrode positions for HF REVE
        self.n_times = n_times
        self.sfreq = sfreq
        self.batch_size = batch_size

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        ch_names: Optional[list[str]] = None,
        n_chans: Optional[int] = None,
        n_times: int = 800,
        sfreq: int = 200,
        device: Optional[str] = None,
        batch_size: int = 4,
    ) -> "REVEWrapper":
        """Load REVE-Base from HuggingFace with correct channel positions.

        Provide either ch_names (explicit list, recommended) or n_chans
        (uses first n_chans entries from REVE_STANDARD_CH_21).

        Args:
            ch_names:   Ordered list of EEG channel names matching your data.
                        Takes precedence over n_chans if both are given.
            n_chans:    Number of channels — only used when ch_names is None.
                        Defaults to 21 if neither argument is provided.
            n_times:    Epoch length in samples (default 800 = 4s at 200 Hz).
            sfreq:      Sampling frequency in Hz (default 200).
            device:     'mps', 'cuda', or 'cpu'. Auto-selects if None.
            batch_size: Inference batch size (default 4 — conservative for
                        16 GB RAM; use 2 if you hit memory errors).

        Returns:
            REVEWrapper instance ready for extract().

        Raises:
            ImportError: if braindecode is not installed.
        """
        from transformers import AutoModel  # noqa: PLC0415

        # Resolve channel names
        if ch_names is None:
            n = n_chans if n_chans is not None else 21
            ch_names = REVE_STANDARD_CH_21[:n]
            logger.info(
                f"REVE: ch_names not provided — using first {n} channels "
                f"from REVE_STANDARD_CH_21"
            )

        n_chans_actual = len(ch_names)
        dev = _select_device(device)

        logger.info(
            f"Loading REVE-Base ({REVE_HUB_ID}) — "
            f"{n_chans_actual} channels, {n_times} samples, {sfreq} Hz, device={dev}"
        )

        model = AutoModel.from_pretrained(REVE_HUB_ID, trust_remote_code=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        model = model.to(dev)

        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"REVE loaded: {n_params / 1e6:.1f}M params on {dev}")

        pos_tensor = _build_pos_tensor(ch_names).to(dev)  # (1, n_ch, 3)

        return cls(
            model=model,
            device=dev,
            ch_names=ch_names,
            n_times=n_times,
            sfreq=sfreq,
            batch_size=batch_size,
            pos_tensor=pos_tensor,
        )

    # ── Embedding extraction ──────────────────────────────────────────────────

    def extract(
        self,
        epochs: np.ndarray,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Extract frozen embeddings from a batch of EEG epochs.

        Hooks the last LayerNorm in REVE, runs inference in batches,
        and mean-pools the output to one vector per epoch.

        Args:
            epochs:     Float32 array of shape (n_epochs, n_channels, n_times).
                        n_channels must match the ch_names used at load time.
            batch_size: Override the default batch size set at load time.

        Returns:
            Float32 array of shape (n_epochs, embedding_dim).

        Raises:
            ValueError:  if epochs has wrong number of channels.
            RuntimeError: if no embeddings are captured (hook misconfigured).
        """
        if epochs.shape[1] != len(self.ch_names):
            raise ValueError(
                f"REVE expects {len(self.ch_names)} channels "
                f"(loaded with {self.ch_names[:3]}...), "
                f"but got {epochs.shape[1]}. "
                f"Re-instantiate with the correct ch_names."
            )

        bs = batch_size if batch_size is not None else self.batch_size

        all_embeds: list[np.ndarray] = []
        for start in range(0, len(epochs), bs):
            batch = torch.FloatTensor(epochs[start: start + bs]).to(self.device)
            b = batch.shape[0]
            # Expand pos from (1, n_ch, 3) to (batch, n_ch, 3)
            pos = self.pos_tensor.expand(b, -1, -1)
            with torch.no_grad():
                feat = self.model(batch, pos, return_output=True)
            # return_output=True → list of layer activations; take the last (deepest) layer
            if isinstance(feat, (list, tuple)):
                feat = feat[-1]
            # feat shape: (batch, n_ch * n_patches, embed_dim) — mean-pool to (batch, embed_dim)
            feat = _pool_features(feat)
            all_embeds.append(feat.cpu().numpy())

            if start > 0 and start % 200 == 0:
                logger.info(f"  REVE: {start}/{len(epochs)} epochs")

        if not all_embeds:
            raise RuntimeError("REVE: no embeddings produced.")

        return np.concatenate(all_embeds, axis=0)

    # ── Memory management ─────────────────────────────────────────────────────

    def free(self) -> None:
        """Delete the model from memory and flush the device cache.

        Call this after embedding extraction is complete to free VRAM/RAM
        before loading another model.
        """
        del self.model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("REVE model freed from memory.")
