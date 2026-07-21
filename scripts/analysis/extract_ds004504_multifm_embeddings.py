#!/usr/bin/env python3
"""Extract LaBraM / CBraMod / BIOT / BENDR subject-mean embeddings on ds004504.

Closes the first piece of the multi-FM transfer/site-dominance gap: today only
REVE has cached subject-mean embeddings for ds004504 (data/embeddings/
ds004504_3way_reve_19ch_subjects.npz), so lrtc_p1_1_irrecoverability.py's site
decoding can only test REVE. This produces the same-schema files for the four
other models so that script can loop over all five.

Reuses the exact preprocessed cache (data/processed/alzheimer_processed.pkl,
19ch, 200Hz) and per-model extraction code already validated in
scripts/eval/run_ds004504_3way.py (LaBraM, CBraMod) plus the BIOT/BENDR
wrappers used elsewhere in the repo (src/biot_wrapper.py, src/bendr_wrapper.py).
Each model's per-epoch embeddings are subject-mean-pooled to match the schema
of ds004504_3way_reve_19ch_subjects.npz: {sid: {"embedding": array, "label": int,
"n_epochs": int}}.

Run from repo root:
  /opt/anaconda3/envs/neurogenis/bin/python scripts/analysis/extract_ds004504_multifm_embeddings.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import mne
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.embeddings import get_device, load_labram_adapted  # noqa: E402
from src.cbramod_wrapper import load_pretrained_cbramod, extract_cbramod_epoch_embeds  # noqa: E402
from src.biot_wrapper import load_pretrained_biot, extract_biot_epoch_embeds  # noqa: E402
from src.bendr_wrapper import load_pretrained_bendr, extract_bendr_epoch_embeds  # noqa: E402
from src.io_safety import safe_pickle_load  # noqa: E402

mne.set_log_level("WARNING")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%H:%M:%S")
logger = logging.getLogger("ds004504_multifm_embed")

CACHE = ROOT / "data" / "processed" / "alzheimer_processed.pkl"
OUT_DIR = ROOT / "data" / "embeddings"
LABEL_MAP = {"A": 0, "C": 1, "F": 2}  # AD / Control / FTD — arbitrary int codes, only used as a covariate
TARGET_SFREQ = 200

# Same 19-channel 10-20 order used for the REVE extraction in run_ds004504_3way.py —
# alzheimer_processed.pkl's epochs are stored in this channel order.
STANDARD_CH = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
]


def _chs_info(n_ch: int) -> list[dict]:
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_map = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    info = []
    for ch in STANDARD_CH[:n_ch]:
        lookup = ch_map.get(ch, ch)
        try:
            pos = montage.get_positions()["ch_pos"][lookup]
            info.append({"ch_name": ch, "loc": list(pos) + [0] * 9})
        except (ValueError, KeyError):
            info.append({"ch_name": ch})
    return info


def _extract_labram_epoch_embeds(model: torch.nn.Module, epochs: np.ndarray, ch_names: list[str]) -> np.ndarray:
    """Same hook-based extraction as run_ds004504_3way.py's _extract_labram_embeds,
    plus explicit ch_names (newer braindecode requires channel identity at forward time)."""
    device = get_device("mps")
    model = model.to(device)
    captured: dict = {}

    for name, module in model.named_modules():
        if name == "fc_norm":
            hook = module.register_forward_hook(lambda m, inp, out: captured.update({"f": out}))
            break
    else:
        last_ln = None
        for _, module in model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                last_ln = module
        hook = last_ln.register_forward_hook(lambda m, inp, out: captured.update({"f": out}))

    all_embeds = []
    for start in range(0, len(epochs), 16):
        batch = torch.FloatTensor(epochs[start:start + 16]).to(device)
        with torch.no_grad():
            _ = model(batch, ch_names=ch_names)
            if "f" in captured:
                feat = captured["f"]
                if feat.dim() == 3:
                    feat = feat.mean(dim=1)
                all_embeds.append(feat.cpu().numpy())
                captured.clear()
        if device.type == "mps" and start % 200 == 0:
            torch.mps.empty_cache()

    hook.remove()
    return np.concatenate(all_embeds)


def _subject_pool(per_epoch: dict[str, np.ndarray], labels: dict[str, int]) -> dict:
    return {
        sid: {
            "embedding": emb.mean(axis=0).astype(np.float32),
            "label": int(labels[sid]),
            "n_epochs": int(len(emb)),
        }
        for sid, emb in per_epoch.items()
    }


def main() -> None:
    if not CACHE.exists():
        logger.error(f"Preprocessed cache not found: {CACHE}\nRun scripts/eval/run_alzheimer_eval.py first.")
        sys.exit(1)

    logger.info("Loading preprocessed ds004504 data...")
    all_data = safe_pickle_load(CACHE)
    subj_ids = sorted(all_data.keys())
    labels = {sid: LABEL_MAP[all_data[sid]["group"]] for sid in subj_ids}
    n_ch = next(iter(all_data.values()))["epochs"].shape[1]
    logger.info(f"{len(subj_ids)} subjects, {n_ch} channels, {TARGET_SFREQ}Hz")

    # ── LaBraM ────────────────────────────────────────────────────────────
    logger.info("\n== LaBraM ==")
    n_times = next(iter(all_data.values()))["epochs"].shape[2]
    ch_names = STANDARD_CH[:n_ch]
    config = {"models": {"labram": {"name": "braindecode/labram-pretrained", "device": "mps", "batch_size": 16}}}
    labram = load_labram_adapted(n_chans=n_ch, n_times=n_times, config=config, chs_info=_chs_info(n_ch))
    per_epoch = {sid: _extract_labram_epoch_embeds(labram, all_data[sid]["epochs"], ch_names) for sid in subj_ids}
    out = _subject_pool(per_epoch, labels)
    np.savez_compressed(OUT_DIR / "ds004504_3way_labram_19ch_subjects.npz", data=out)
    logger.info(f"wrote ds004504_3way_labram_19ch_subjects.npz ({len(out)} subjects, "
                f"dim={next(iter(out.values()))['embedding'].shape})")
    del labram
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # ── CBraMod ───────────────────────────────────────────────────────────
    logger.info("\n== CBraMod ==")
    cbramod = load_pretrained_cbramod("cpu")
    per_epoch = {sid: extract_cbramod_epoch_embeds(cbramod, all_data[sid]["epochs"]) for sid in subj_ids}
    out = _subject_pool(per_epoch, labels)
    np.savez_compressed(OUT_DIR / "ds004504_3way_cbramod_19ch_subjects.npz", data=out)
    logger.info(f"wrote ds004504_3way_cbramod_19ch_subjects.npz ({len(out)} subjects, "
                f"dim={next(iter(out.values()))['embedding'].shape})")
    del cbramod

    # ── BIOT ──────────────────────────────────────────────────────────────
    logger.info("\n== BIOT ==")
    biot = load_pretrained_biot("cpu")
    per_epoch = {sid: extract_biot_epoch_embeds(biot, all_data[sid]["epochs"], device="cpu")
                 for sid in subj_ids}
    out = _subject_pool(per_epoch, labels)
    np.savez_compressed(OUT_DIR / "ds004504_3way_biot_19ch_subjects.npz", data=out)
    logger.info(f"wrote ds004504_3way_biot_19ch_subjects.npz ({len(out)} subjects, "
                f"dim={next(iter(out.values()))['embedding'].shape})")
    del biot

    # ── BENDR ─────────────────────────────────────────────────────────────
    logger.info("\n== BENDR ==")
    bendr = load_pretrained_bendr(n_times=n_times, device="cpu")
    per_epoch = {sid: extract_bendr_epoch_embeds(bendr, all_data[sid]["epochs"], sfreq=TARGET_SFREQ, device="cpu")
                 for sid in subj_ids}
    out = _subject_pool(per_epoch, labels)
    np.savez_compressed(OUT_DIR / "ds004504_3way_bendr_19ch_subjects.npz", data=out)
    logger.info(f"wrote ds004504_3way_bendr_19ch_subjects.npz ({len(out)} subjects, "
                f"dim={next(iter(out.values()))['embedding'].shape})")
    del bendr

    logger.info("\nDone. All four ds004504_3way_<model>_19ch_subjects.npz files written to data/embeddings/.")


if __name__ == "__main__":
    main()
