"""BENDR wrapper for PoC evaluation — braindecode from_pretrained.

BENDR (Kostas et al., 2021, Frontiers in Human Neuroscience) — wav2vec 2.0-style
contrastive learning for EEG. 157M params, pretrained on TUH at 256Hz, 20 channels.
Loaded via braindecode: BENDR.from_pretrained('braindecode/braindecode-bendr')

Channel handling: pretrained on 20ch. Data >20ch is truncated to first 20.

Embedding extraction: hook encoder output → (batch, 512, T') → mean pool → 512-d.
Tested all hook points (encoder, contextualizer.norm, output_layer, last transformer
layer, concat). All give ~0.55 AUROC on CHB-MIT (near-chance). This is a legitimate
architectural limitation: BENDR's wav2vec 2.0 design produces representations that
require fine-tuning for task-specific discrimination. The 96x temporal downsampling
in the encoder (stride 3*2^5) also limits temporal resolution.
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F

from src.embeddings import capture_hook

logger = logging.getLogger(__name__)

BENDR_SFREQ = 256  # pretrained sampling frequency
BENDR_EMBED_DIM = 512  # encoder output dimension
BENDR_PRETRAINED_CHANS = 20  # pretrained channel count


def load_pretrained_bendr(
    n_times: int = 1024,
    device: str = "cpu",
) -> torch.nn.Module:
    """Load BENDR via braindecode from_pretrained.

    Always loads with n_chans=20 (pretrained). Data with more channels
    must be truncated to 20 before extraction.

    Args:
        n_times: Number of time samples (at 256Hz — resample before calling)
        device: 'cpu' or 'mps'

    Returns:
        BENDR model in eval mode with frozen parameters
    """
    from braindecode.models import BENDR

    logger.info(
        f"Loading BENDR from braindecode ({BENDR_PRETRAINED_CHANS}ch pretrained)..."
    )
    import torch.nn as nn

    model = BENDR.from_pretrained(
        "braindecode/braindecode-bendr",
        n_outputs=2,
        n_chans=BENDR_PRETRAINED_CHANS,
        n_times=n_times,
        sfreq=BENDR_SFREQ,
        activation=nn.GELU,  # braindecode 1.3.2 bug: str activation not callable
    )

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"BENDR loaded: {n_params / 1e6:.1f}M params on {device}")

    return model


def extract_bendr_epoch_embeds(
    model: torch.nn.Module,
    epochs: np.ndarray,
    sfreq: float = 200.0,
    batch_size: int = 4,
    device: str = "cpu",
) -> np.ndarray:
    """Extract frozen embeddings from BENDR.

    Truncates to 20 channels (pretrained), resamples 200→256Hz,
    hooks encoder output → (batch, 512, T') → mean pool → (batch, 512).

    Args:
        model: Loaded BENDR model (braindecode)
        epochs: (n_epochs, n_channels, n_times) at sfreq Hz
        sfreq: Sampling frequency of input data
        batch_size: Batch size for inference
        device: Device for inference

    Returns:
        (n_epochs, 512) array of frozen embeddings
    """
    model.eval()
    n_epochs, n_channels, n_times = epochs.shape

    # Match to 20 channels (BENDR Conv1d encoder requires exactly 20)
    if n_channels > BENDR_PRETRAINED_CHANS:
        logger.info(
            f"Truncating {n_channels} → {BENDR_PRETRAINED_CHANS} channels (BENDR pretrained)"
        )
        epochs = epochs[:, :BENDR_PRETRAINED_CHANS, :]
        n_channels = BENDR_PRETRAINED_CHANS
    elif n_channels < BENDR_PRETRAINED_CHANS:
        pad_count = BENDR_PRETRAINED_CHANS - n_channels
        logger.info(
            f"Zero-padding {n_channels} → {BENDR_PRETRAINED_CHANS} channels (+{pad_count} dead channels)"
        )
        padding = np.zeros((n_epochs, pad_count, n_times), dtype=epochs.dtype)
        epochs = np.concatenate([epochs, padding], axis=1)
        n_channels = BENDR_PRETRAINED_CHANS

    # Resample 200Hz → 256Hz (BENDR's native rate)
    if abs(sfreq - BENDR_SFREQ) > 1:
        new_n_times = int(n_times * BENDR_SFREQ / sfreq)
        logger.info(
            f"Resampling {sfreq}Hz → {BENDR_SFREQ}Hz ({n_times} → {new_n_times} samples)"
        )
        resampled = np.zeros((n_epochs, n_channels, new_n_times), dtype=np.float32)
        chunk = 64
        for i in range(0, n_epochs, chunk):
            t = torch.FloatTensor(epochs[i : i + chunk])
            t_r = F.interpolate(t, size=new_n_times, mode="linear", align_corners=False)
            resampled[i : i + chunk] = t_r.numpy()
        epochs = resampled
        n_times = new_n_times

    # Hook encoder output (Conv1d stack → (batch, 512, T'))
    all_embeds = []
    with capture_hook(model.encoder) as captured:
        for start in range(0, n_epochs, batch_size):
            batch = torch.FloatTensor(epochs[start : start + batch_size]).to(device)

            with torch.no_grad():
                _ = model(batch)

                if "f" in captured:
                    feat = captured["f"]  # (batch, 512, T')
                    embed = feat.mean(dim=2)  # (batch, 512)
                    all_embeds.append(embed.cpu().numpy())
                    captured.clear()

            if start % 100 == 0 and start > 0:
                logger.info(f"  BENDR embeddings: {start}/{n_epochs}")

    if not all_embeds:
        raise RuntimeError(
            "BENDR: no embeddings captured — check hook target (model.encoder)"
        )
    result = np.concatenate(all_embeds, axis=0)
    logger.info(f"BENDR embeddings: {result.shape}")
    return result
