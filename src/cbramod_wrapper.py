"""CBraMod wrapper — model architecture adapted from the upstream CBraMod repo.

CBraMod (Wang et al., 2025, ICLR) — Criss-Cross Brain Foundation Model.
Pure PyTorch, no CUDA-only dependencies. Runs on CPU/MPS natively.

Architecture: Criss-cross attention (split spatial + temporal attention heads),
time-domain conv + spectral FFT patch embedding, 12-layer transformer.

-----------------------------------------------------------------------------
THIRD-PARTY ATTRIBUTION
The CBraMod model architecture classes below (CBraMod, CrissCrossTransformer*,
CBraModPatchEmbedding, and related nn.Modules) are adapted from the upstream
CBraMod reference implementation:
    https://github.com/wjq-learning/CBraMod
    Copyright (c) 2025 Jiquan Wang
    Licensed under the MIT License.
The upstream MIT license text is reproduced in this repository's NOTICE file.
Pretrained weights are downloaded at runtime from the Hugging Face Hub
(weighting666/CBraMod) and are NOT redistributed here.
-----------------------------------------------------------------------------
"""

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ── Criss-Cross Transformer (from CBraMod repo) ──


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class CrissCrossTransformerEncoderLayer(nn.Module):
    """Criss-cross attention: splits embedding into spatial and temporal halves."""

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.gelu,
        layer_norm_eps=1e-5,
        batch_first=True,
        norm_first=True,
        bias=True,
    ):
        super().__init__()
        # Spatial attention on first half of embedding
        self.self_attn_s = nn.MultiheadAttention(
            d_model // 2,
            nhead // 2,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
        )
        # Temporal attention on second half
        self.self_attn_t = nn.MultiheadAttention(
            d_model // 2,
            nhead // 2,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation
        self.norm_first = norm_first

    def forward(self, src, src_mask=None):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, attn_mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        half = patch_size // 2

        # Spatial: attention across channels for each time patch
        xs = x[:, :, :, :half]
        xs = xs.transpose(1, 2).contiguous().view(bz * patch_num, ch_num, half)
        xs = self.self_attn_s(xs, xs, xs, need_weights=False)[0]
        xs = xs.view(bz, patch_num, ch_num, half).transpose(1, 2)

        # Temporal: attention across time patches for each channel
        xt = x[:, :, :, half:]
        xt = xt.contiguous().view(bz * ch_num, patch_num, half)
        xt = self.self_attn_t(xt, xt, xt, need_weights=False)[0]
        xt = xt.view(bz, ch_num, patch_num, half)

        x = torch.cat((xs, xt), dim=3)
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class CrissCrossTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = norm

    def forward(self, src, mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


# ── CBraMod Model ──


class CBraMod(nn.Module):
    """CBraMod: Criss-Cross Brain Foundation Model.

    Input: [batch, n_channels, n_patches, patch_size=200]
    Output: [batch, n_channels, n_patches, d_model]
    """

    def __init__(
        self,
        in_dim=200,
        out_dim=200,
        d_model=200,
        dim_feedforward=800,
        seq_len=30,
        n_layer=12,
        nhead=8,
    ):
        super().__init__()
        self.patch_embedding = CBraModPatchEmbedding(in_dim, out_dim, d_model, seq_len)
        encoder_layer = CrissCrossTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
            activation=F.gelu,
        )
        self.encoder = CrissCrossTransformerEncoder(encoder_layer, num_layers=n_layer)
        self.proj_out = nn.Sequential(nn.Linear(d_model, out_dim))

    def forward(self, x, mask=None):
        patch_emb = self.patch_embedding(x, mask)
        feats = self.encoder(patch_emb)
        out = self.proj_out(feats)
        return out


class CBraModPatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(
                d_model,
                d_model,
                kernel_size=(19, 7),
                stride=(1, 1),
                padding=(9, 3),
                groups=d_model,
            ),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
        self.proj_in = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(25, 25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(25, 25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(101, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        mask_x = x if mask is None else x.clone()
        if mask is not None:
            mask_x[mask == 1] = self.mask_encoding

        mask_x_flat = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(mask_x_flat)
        patch_emb = (
            patch_emb.permute(0, 2, 1, 3)
            .contiguous()
            .view(bz, ch_num, patch_num, self.d_model)
        )

        # Spectral embedding
        spec_input = mask_x_flat.view(bz * ch_num * patch_num, patch_size)
        spectral = torch.fft.rfft(spec_input, dim=-1, norm="forward")
        spectral = torch.abs(spectral).view(bz, ch_num, patch_num, 101)
        spectral_emb = self.spectral_proj(spectral)
        patch_emb = patch_emb + spectral_emb

        # Positional encoding
        pos_emb = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        pos_emb = pos_emb.permute(0, 2, 3, 1)
        patch_emb = patch_emb + pos_emb

        return patch_emb


# ── Loading and Extraction ──


def load_pretrained_cbramod(device: str = "cpu") -> CBraMod:
    """Load pretrained CBraMod weights from HuggingFace."""
    from huggingface_hub import hf_hub_download

    logger.info("Downloading CBraMod pretrained weights...")
    weight_path = hf_hub_download("weighting666/CBraMod", "pretrained_weights.pth")

    # Safety: weights from pinned HuggingFace repo (weighting666/CBraMod).
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)

    model = CBraMod()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(
            f"Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    if unexpected:
        logger.warning(
            f"Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
        )

    loaded = len(state_dict) - len(unexpected)
    logger.info(f"Loaded {loaded}/{len(state_dict)} pretrained parameters")

    model.eval()
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"CBraMod: {n_params / 1e6:.1f}M params on {device}")

    return model


def extract_cbramod_epoch_embeds(model, epochs, batch_size=8, device="cpu"):
    """Extract CBraMod embeddings from epoch data."""
    patch_size = 200
    n_epochs, n_channels, n_times = epochs.shape
    n_patches = n_times // patch_size
    if n_patches < 1:
        padded = np.zeros((n_epochs, n_channels, patch_size), dtype=epochs.dtype)
        padded[:, :, :n_times] = epochs
        epochs = padded
        n_patches = 1

    trimmed = epochs[:, :, : n_patches * patch_size]
    reshaped = trimmed.reshape(n_epochs, n_channels, n_patches, patch_size)

    all_embeds = []
    for start in range(0, n_epochs, batch_size):
        batch = torch.FloatTensor(reshaped[start : start + batch_size]).to(device)
        with torch.no_grad():
            out = model(batch)  # [batch, ch, patches, d_model]
            feat = out.mean(dim=(1, 2))  # Mean pool -> [batch, d_model]
            all_embeds.append(feat.cpu().numpy())

    return np.concatenate(all_embeds)
