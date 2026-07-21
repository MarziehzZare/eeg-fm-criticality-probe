"""BIOT wrapper for PoC evaluation — braindecode from_pretrained.

BIOT (Yang et al., NeurIPS 2023) — Biosignal Transformer for cross-domain
biosignal analysis. 3.2M params, 256-d embeddings.

Pretrained on TUH + SHHS + CHB-MIT + IIIC (six datasets, 18 channels, 200Hz).
Loaded via braindecode: BIOT.from_pretrained('braindecode/biot-pretrained-six-datasets-18chs')

Channel constraint: pretrained on 18 channels. Data with >18ch must be truncated.
Embedding extraction: hook encoder.transformer → (batch, seq, 256) → mean pool.
"""

import logging

import numpy as np
import torch

from src.embeddings import capture_hook

logger = logging.getLogger(__name__)

BIOT_SFREQ = 200  # pretrained sampling frequency
BIOT_EMBED_DIM = 256  # transformer embedding dimension
BIOT_N_CHANNELS = 18  # pretrained channel count

# Standard 18-channel subset (matches BIOT six-dataset pretrained order)
BIOT_CHANNELS_18 = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6', 'O1',
]


def load_pretrained_biot(device: str = "cpu") -> torch.nn.Module:
    """Load BIOT via braindecode from_pretrained.

    Args:
        device: 'cpu' or 'mps'

    Returns:
        BIOT model in eval mode with frozen parameters
    """
    from braindecode.models import BIOT

    logger.info("Loading BIOT from braindecode (six-datasets-18chs)...")
    model = BIOT.from_pretrained('braindecode/biot-pretrained-six-datasets-18chs')

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"BIOT loaded: {n_params / 1e6:.1f}M params on {device}")

    return model


def extract_biot_epoch_embeds(
    model: torch.nn.Module,
    epochs: np.ndarray,
    batch_size: int = 8,
    device: str = "cpu",
) -> np.ndarray:
    """Extract frozen embeddings from BIOT.

    Hooks encoder.transformer (LinearAttentionTransformer) which outputs
    (batch, seq_len, 256). Mean-pools over seq_len → (batch, 256).

    BIOT pretrained on 18 channels — truncates input to first 18 if >18.

    Args:
        model: Loaded BIOT model (braindecode)
        epochs: (n_epochs, n_channels, n_times) at 200Hz
        batch_size: Batch size for inference
        device: Device for inference

    Returns:
        (n_epochs, 256) array of frozen embeddings
    """
    model.eval()
    n_epochs, n_channels, n_times = epochs.shape

    # Handle channel count mismatch with BIOT's pretrained 18 channels
    if n_channels > BIOT_N_CHANNELS:
        logger.info(f"Truncating {n_channels} → {BIOT_N_CHANNELS} channels (BIOT limit)")
        epochs = epochs[:, :BIOT_N_CHANNELS, :]
        n_channels = BIOT_N_CHANNELS
    elif n_channels < BIOT_N_CHANNELS:
        pad_width = BIOT_N_CHANNELS - n_channels
        logger.warning(
            f"BIOT requires {BIOT_N_CHANNELS} channels but got {n_channels}. "
            f"Zero-padding {pad_width} channels. Results may be degraded."
        )
        padding = np.zeros((n_epochs, pad_width, n_times), dtype=epochs.dtype)
        epochs = np.concatenate([epochs, padding], axis=1)
        n_channels = BIOT_N_CHANNELS

    # Hook the transformer output for embeddings
    all_embeds = []
    with capture_hook(model.encoder.transformer) as captured:
        for start in range(0, n_epochs, batch_size):
            batch = torch.FloatTensor(epochs[start:start + batch_size]).to(device)

            with torch.no_grad():
                _ = model(batch)

                if "f" in captured:
                    feat = captured["f"]  # (batch, seq_len, 256)
                    embed = feat.mean(dim=1)  # (batch, 256)
                    all_embeds.append(embed.cpu().numpy())
                    captured.clear()

            if start % 100 == 0 and start > 0:
                logger.info(f"  BIOT embeddings: {start}/{n_epochs}")

    if not all_embeds:
        raise RuntimeError("BIOT: no embeddings captured — check hook target (model.encoder.transformer)")
    result = np.concatenate(all_embeds, axis=0)
    logger.info(f"BIOT embeddings: {result.shape}")
    return result
