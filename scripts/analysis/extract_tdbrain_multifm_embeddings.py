#!/usr/bin/env python3
"""Extract LaBraM / CBraMod / BIOT / BENDR subject-mean embeddings on TDBRAIN.

Second piece of the multi-FM transfer/site-dominance gap (after ds004504):
today only REVE (data/embeddings/tdbrain_mdd_healthy_reve_subjects.npz) has
cached subject-mean embeddings for TDBRAIN. This produces the same-schema
files for the four other models.

Note: TDBRAIN is excluded from every REVE-embedding analysis in the paper
(TDBRAIN is in REVE's own pretraining corpus; El Ouahidi et al., NeurIPS 2025, Appendix B).
That leak is REVE-specific and unconfirmed for LaBraM/CBraMod/BIOT/BENDR --
each has its own pretraining corpus, not yet individually audited. These
embeddings are produced so that question can be answered, not to reinstate
TDBRAIN into the existing REVE-based claims.

Reads raw BDF directly via MNE (the freshly-downloaded TDBRAIN V3.1 release
ships BIDS/BDF, not the CSV derivatives the older tdbrain_mdd_healthy_reve
pipeline used) and applies the identical preprocessing recipe documented in
scripts/eval/run_tdbrain_mdd_eval.py: 19-ch subset (REVE_CH_19), common-average
reference, notch 50 Hz, bandpass 0.5-70 Hz, resample 500->200 Hz, 4s epochs,
peak-to-peak >500uV rejection. No participant labels are used or needed --
this release's participants.tsv (diagnosis metadata) is not present in the
new download, and the site-decoding use case only needs subject identity,
not diagnosis.

Run from repo root:
  /opt/anaconda3/envs/neurogenis/bin/python scripts/analysis/extract_tdbrain_multifm_embeddings.py [--cap 300]
"""
from __future__ import annotations

import argparse
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

mne.set_log_level("WARNING")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%H:%M:%S")
logger = logging.getLogger("tdbrain_multifm_embed")

TDBRAIN_ROOT = ROOT / "data" / "raw" / "tdbrain" / "TDBRAIN_Dataset_V3_1"
OUT_DIR = ROOT / "data" / "embeddings"

REVE_CH_19 = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8",
    "Fz", "Cz", "Pz",
]
SFREQ_RAW = 500.0
SFREQ_TARGET = 200.0
EPOCH_DURATION = 4.0  # seconds
REJECT_UV = 500.0
SEED = 42


def load_subject_epochs(sub_dir: Path) -> np.ndarray | None:
    """Load, preprocess, epoch one TDBRAIN subject's restEC recording. (n_epochs, 19, 800) float32."""
    bdf_path = sub_dir / "ses-1" / "eeg" / f"{sub_dir.name}_ses-1_task-restEC_eeg.bdf"
    if not bdf_path.exists():
        return None
    try:
        raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose="ERROR")
    except Exception as exc:
        logger.debug(f"  {sub_dir.name}: BDF load failed: {exc}")
        return None

    available = [ch for ch in REVE_CH_19 if ch in raw.ch_names]
    if len(available) < len(REVE_CH_19):
        return None
    raw.pick(available)
    raw.reorder_channels(available)

    raw.set_eeg_reference("average", projection=False, verbose=False)
    raw.notch_filter(50.0, verbose=False)
    raw.filter(l_freq=0.5, h_freq=70.0, verbose=False)
    if raw.info["sfreq"] != SFREQ_TARGET:
        raw.resample(SFREQ_TARGET, verbose=False)

    n_samples = int(EPOCH_DURATION * SFREQ_TARGET)
    data = raw.get_data(units="uV")
    n_epochs = data.shape[1] // n_samples
    if n_epochs < 5:
        return None

    epochs = data[:, : n_epochs * n_samples].reshape(n_epochs, len(available), n_samples)
    peak_to_peak = epochs.max(axis=2) - epochs.min(axis=2)
    good = (peak_to_peak < REJECT_UV).all(axis=1)
    epochs = epochs[good]
    if len(epochs) < 3:
        return None
    return (epochs * 1e-6).astype(np.float32)  # back to volts for the model wrappers


def _extract_labram_epoch_embeds(model: torch.nn.Module, epochs: np.ndarray, ch_names: list[str]) -> np.ndarray:
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


def _chs_info(ch_names: list[str]) -> list[dict]:
    montage = mne.channels.make_standard_montage("standard_1020")
    positions = montage.get_positions()["ch_pos"]
    info = []
    for ch in ch_names:
        try:
            pos = positions[ch]
            info.append({"ch_name": ch, "loc": list(pos) + [0] * 9})
        except KeyError:
            info.append({"ch_name": ch})
    return info


def _subject_pool(per_epoch: dict[str, np.ndarray]) -> dict:
    return {
        sid: {"embedding": emb.mean(axis=0).astype(np.float32), "n_epochs": int(len(emb))}
        for sid, emb in per_epoch.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=300, help="max subjects (matches paper Table 1's TDBRAIN N=300)")
    args = ap.parse_args()

    sub_dirs = sorted(TDBRAIN_ROOT.glob("sub-*"))
    rng = np.random.default_rng(SEED)
    if len(sub_dirs) > args.cap:
        idx = rng.choice(len(sub_dirs), size=args.cap, replace=False)
        sub_dirs = [sub_dirs[i] for i in sorted(idx)]
    logger.info(f"Candidate subjects: {len(sub_dirs)} (capped at {args.cap})")

    logger.info("Loading + preprocessing restEC epochs (BDF -> 19ch/200Hz/CAR/4s)...")
    all_epochs: dict[str, np.ndarray] = {}
    for i, sd in enumerate(sub_dirs):
        ep = load_subject_epochs(sd)
        if ep is not None:
            all_epochs[sd.name] = ep
        if (i + 1) % 50 == 0:
            logger.info(f"  {i + 1}/{len(sub_dirs)} scanned, {len(all_epochs)} usable so far")
    subj_ids = sorted(all_epochs.keys())
    n_ch = len(REVE_CH_19)
    n_times = next(iter(all_epochs.values())).shape[2]
    logger.info(f"Usable subjects: {len(subj_ids)}, {n_ch} channels, {n_times} samples/epoch @ {SFREQ_TARGET}Hz")

    # ── LaBraM ────────────────────────────────────────────────────────────
    logger.info("\n== LaBraM ==")
    config = {"models": {"labram": {"name": "braindecode/labram-pretrained", "device": "mps", "batch_size": 16}}}
    labram = load_labram_adapted(n_chans=n_ch, n_times=n_times, config=config, chs_info=_chs_info(REVE_CH_19))
    per_epoch = {sid: _extract_labram_epoch_embeds(labram, all_epochs[sid], REVE_CH_19) for sid in subj_ids}
    out = _subject_pool(per_epoch)
    np.savez_compressed(OUT_DIR / "tdbrain_multifm_labram_19ch_subjects.npz", data=out)
    logger.info(f"wrote tdbrain_multifm_labram_19ch_subjects.npz ({len(out)} subjects, "
                f"dim={next(iter(out.values()))['embedding'].shape})")
    del labram
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # ── CBraMod ───────────────────────────────────────────────────────────
    logger.info("\n== CBraMod ==")
    cbramod = load_pretrained_cbramod("cpu")
    per_epoch = {sid: extract_cbramod_epoch_embeds(cbramod, all_epochs[sid]) for sid in subj_ids}
    out = _subject_pool(per_epoch)
    np.savez_compressed(OUT_DIR / "tdbrain_multifm_cbramod_19ch_subjects.npz", data=out)
    logger.info(f"wrote tdbrain_multifm_cbramod_19ch_subjects.npz ({len(out)} subjects, "
                f"dim={next(iter(out.values()))['embedding'].shape})")
    del cbramod

    # ── BIOT ──────────────────────────────────────────────────────────────
    logger.info("\n== BIOT ==")
    biot = load_pretrained_biot("cpu")
    per_epoch = {sid: extract_biot_epoch_embeds(biot, all_epochs[sid], device="cpu") for sid in subj_ids}
    out = _subject_pool(per_epoch)
    np.savez_compressed(OUT_DIR / "tdbrain_multifm_biot_19ch_subjects.npz", data=out)
    logger.info(f"wrote tdbrain_multifm_biot_19ch_subjects.npz ({len(out)} subjects, "
                f"dim={next(iter(out.values()))['embedding'].shape})")
    del biot

    # ── BENDR ─────────────────────────────────────────────────────────────
    logger.info("\n== BENDR ==")
    bendr = load_pretrained_bendr(n_times=n_times, device="cpu")
    per_epoch = {sid: extract_bendr_epoch_embeds(bendr, all_epochs[sid], sfreq=SFREQ_TARGET, device="cpu")
                 for sid in subj_ids}
    out = _subject_pool(per_epoch)
    np.savez_compressed(OUT_DIR / "tdbrain_multifm_bendr_19ch_subjects.npz", data=out)
    logger.info(f"wrote tdbrain_multifm_bendr_19ch_subjects.npz ({len(out)} subjects, "
                f"dim={next(iter(out.values()))['embedding'].shape})")
    del bendr

    logger.info("\nDone. All four tdbrain_multifm_<model>_19ch_subjects.npz files written to data/embeddings/.")


if __name__ == "__main__":
    main()
