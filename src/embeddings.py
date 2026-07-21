"""Foundation model embedding extraction.

Extracts frozen embeddings from LaBraM (primary) and optionally EEGMamba.
Handles channel/time dimension mismatch via position embedding interpolation.
Designed for Mac M3 16GB with MPS backend.
"""

import logging
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@contextmanager
def capture_hook(
    module: torch.nn.Module,
    key: str = "f",
) -> Generator[dict, None, None]:
    """Context manager that registers a forward hook and yields a capture dict.

    Usage:
        with capture_hook(model.some_layer) as captured:
            model(batch)
            feat = captured["f"]  # output tensor from that layer

    The hook is guaranteed to be removed on exit, even if an exception occurs.

    Args:
        module: The nn.Module whose output should be captured.
        key: Key used to store the captured output in the returned dict.

    Yields:
        dict that will contain {key: output_tensor} after each forward pass.
    """
    captured: dict = {}

    def _hook(module: torch.nn.Module, _input: tuple, output: torch.Tensor) -> None:
        captured[key] = output

    handle = module.register_forward_hook(_hook)
    try:
        yield captured
    finally:
        handle.remove()


# LaBraM pretrained config (from HuggingFace)
LABRAM_PRETRAINED_N_CHANS = 128
LABRAM_PRETRAINED_N_TIMES = 3000
LABRAM_PRETRAINED_PATCH_SIZE = 200


def get_device(preferred: str = "mps") -> torch.device:
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_labram_adapted(n_chans: int, n_times: int, config: dict, chs_info=None) -> torch.nn.Module:
    """Load pretrained LaBraM and adapt position embeddings for our data dimensions.

    The pretrained model expects 128 channels × 3000 samples.
    We interpolate position embeddings to match our actual data shape.
    All transformer weights (attention, FFN) transfer directly.
    """
    from braindecode.models import Labram

    model_name = config["models"]["labram"]["name"]
    logger.info(f"Loading pretrained LaBraM from {model_name}...")

    # Load pretrained model with original dimensions
    pretrained = Labram.from_pretrained(model_name)
    pretrained_state = pretrained.state_dict()

    # Create a new model with OUR dimensions
    labram_kwargs = dict(
        n_chans=n_chans,
        n_times=n_times,
        sfreq=200,
        n_outputs=2,  # binary classification
        patch_size=LABRAM_PRETRAINED_PATCH_SIZE,
        embed_dim=200,
        num_layers=12,
        num_heads=10,
    )
    if chs_info is not None:
        labram_kwargs["chs_info"] = chs_info
    target_model = Labram(**labram_kwargs)
    target_state = target_model.state_dict()

    # Compute expected token counts
    pretrained_n_temporal = (
        LABRAM_PRETRAINED_N_TIMES // LABRAM_PRETRAINED_PATCH_SIZE
    )  # 15
    target_n_temporal = n_times // LABRAM_PRETRAINED_PATCH_SIZE  # e.g., 4

    adapted_state = OrderedDict()
    skipped = []

    for key, pretrained_param in pretrained_state.items():
        if key not in target_state:
            skipped.append(key)
            continue

        target_param = target_state[key]

        if pretrained_param.shape == target_param.shape:
            adapted_state[key] = pretrained_param
        elif "position_embedding" in key:
            # Interpolate channel position embeddings
            # pretrained: [1, 129, 200] (128 channels + 1 CLS)
            # target: [1, n_chans+1, 200]
            logger.info(
                f"Interpolating {key}: {pretrained_param.shape} -> {target_param.shape}"
            )
            p = pretrained_param.permute(0, 2, 1)  # [1, 200, 129]
            p_interp = F.interpolate(
                p, size=target_param.shape[1], mode="linear", align_corners=False
            )
            adapted_state[key] = p_interp.permute(0, 2, 1)  # [1, n_chans+1, 200]
        elif "temporal_embedding" in key:
            # Interpolate temporal position embeddings
            # pretrained: [1, 16, 200] (15 patches + 1 CLS)
            # target: [1, target_n_temporal+1, 200]
            logger.info(
                f"Interpolating {key}: {pretrained_param.shape} -> {target_param.shape}"
            )
            p = pretrained_param.permute(0, 2, 1)  # [1, 200, 16]
            p_interp = F.interpolate(
                p, size=target_param.shape[1], mode="linear", align_corners=False
            )
            adapted_state[key] = p_interp.permute(0, 2, 1)
        elif "head" in key or "fc_norm" in key:
            # Classification head — will be replaced anyway, use target init
            adapted_state[key] = target_param
        else:
            # Shape mismatch we can't handle — use target initialization
            logger.warning(
                f"Shape mismatch for {key}: {pretrained_param.shape} vs {target_param.shape}, using random init"
            )
            adapted_state[key] = target_param

    # Load adapted weights
    missing = set(target_state.keys()) - set(adapted_state.keys())
    for key in missing:
        adapted_state[key] = target_state[key]

    target_model.load_state_dict(adapted_state)
    target_model.eval()

    n_params = sum(p.numel() for p in target_model.parameters())
    logger.info(
        f"LaBraM adapted: {n_params/1e6:.1f}M params ({len(skipped)} keys skipped)"
    )

    return target_model


def extract_labram_embeddings(
    model: torch.nn.Module,
    processed_data: dict,
    config: dict,
) -> dict:
    """Extract frozen embeddings from LaBraM for all subjects."""
    device = get_device(config["models"]["labram"]["device"])
    model = model.to(device)
    model.eval()  # Explicit eval mode for safety
    batch_size = config["models"]["labram"]["batch_size"]

    # Determine the model's expected channel count from data
    sample = next(iter(processed_data.values()))
    n_chans = sample["n_channels"]
    logger.info(
        f"Extracting embeddings for {len(processed_data)} subjects ({n_chans} channels)"
    )

    embeddings = {}

    # Resolve the target module for the capture hook. Prefer LaBraM's named
    # 'fc_norm' layer; otherwise fall back to the last LayerNorm in the model.
    target_module: torch.nn.Module | None = None
    target_name: str | None = None
    for name, module in model.named_modules():
        if name == "fc_norm":
            target_module = module
            target_name = name
            break
    if target_module is None:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                target_module = module
                target_name = name
    if target_module is None:
        raise RuntimeError("No LayerNorm found in model for embedding extraction hook")
    logger.info(f"Hook target: '{target_name}'")

    total_subjects = len(processed_data)
    with capture_hook(target_module, key="features") as captured:
        for idx, (sid, data) in enumerate(processed_data.items()):
            if idx % 20 == 0:
                logger.info(f"Extracting embeddings: {idx}/{total_subjects}")

            epochs = data["epochs"]  # [n_epochs, n_channels, n_times]
            epoch_embeds = []

            for start in range(0, len(epochs), batch_size):
                batch = epochs[start : start + batch_size]
                batch_tensor = torch.FloatTensor(batch).to(device)

                with torch.no_grad():
                    try:
                        _ = model(batch_tensor)

                        if "features" in captured:
                            features = captured["features"]
                            if features.dim() == 3:
                                feat = features.mean(dim=1)  # mean pool over sequence
                            elif features.dim() == 2:
                                feat = features
                            else:
                                feat = features.reshape(features.size(0), -1)
                            epoch_embeds.append(feat.cpu().numpy())
                            captured.clear()
                    except Exception as e:
                        logger.warning(f"Error on subject {sid}, batch {start}: {e}")
                        break

                if device.type == "mps" and idx % 10 == 0:
                    torch.mps.empty_cache()

            if not epoch_embeds:
                continue

            all_embeds = np.concatenate(epoch_embeds, axis=0)
            subject_embed = all_embeds.mean(axis=0)

            embeddings[sid] = {
                "embedding": subject_embed,
                "epoch_embeddings": all_embeds,
                "label": data["label"],
                "n_epochs": len(all_embeds),
            }

    if embeddings:
        dim = embeddings[next(iter(embeddings))]["embedding"].shape[0]
        logger.info(
            f"Extracted embeddings for {len(embeddings)}/{total_subjects} subjects, dim={dim}"
        )
    else:
        logger.error("No embeddings extracted!")

    return embeddings


def extract_all_embeddings(processed_data: dict, config: dict) -> dict:
    """Main entry point: load adapted model and extract embeddings."""
    sample = next(iter(processed_data.values()))
    n_chans = sample["n_channels"]
    n_times = sample["epochs"].shape[2]

    model = load_labram_adapted(n_chans, n_times, config)
    return extract_labram_embeddings(model, processed_data, config)
