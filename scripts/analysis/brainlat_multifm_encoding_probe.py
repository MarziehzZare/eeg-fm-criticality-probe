"""Generate BrainLat multi-model embeddings used by revised Paper-02 analyses.

This helper harmonizes the available BrainLat recordings to 19 channels,
200 Hz, common-average reference, and a 240-second analysis window, then
extracts frozen LaBraM, CBraMod, BIOT, and BENDR representations.  The current
v2 estimand and fixed readout are evaluated by
``paper02_primary_encoding_v2.py``; the historical exploratory probe retained
in this file must not be substituted for that analysis.

Usage:
    python scripts/analysis/brainlat_multifm_encoding_probe.py --extract-only
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts" / "download"))
warnings.filterwarnings("ignore")

import fm_lrtc_nonlinear_probe as fm          # noqa: E402  make_probes, cv_r2, _car_filt, SF
import ingest_brainlat as ib                  # noqa: E402  select_19ch, DEMENTIA_GROUPS, CONTROL_GROUPS
from src.io_safety import safe_np_load        # noqa: E402
from src.seed import set_seed                 # noqa: E402

SF = fm.SF
COMMON = fm.COMMON
RAW = ROOT / "data" / "raw" / "brainlat"
DUR = int(240 * SF)
EPOCH = int(4 * SF)                            # 800 samples
DEV = "cpu"                                    # deterministic; N=79 is small
PROBES = ["ridge_linear_full", "hist_gboost", "random_forest"]
P05_CACHE = ROOT / "results/cross_population/brainlat_reve_encoding.npz"

LABRAM_CONFIG = {"models": {"labram": {"name": "braindecode/labram-pretrained",
                 "embed_dim": 200, "patch_size": 200, "num_layers": 12,
                 "num_heads": 10, "device": DEV, "batch_size": 32}}}


def _harmonise(fp: str):
    import mne
    r = mne.io.read_raw_eeglab(fp, preload=True, verbose="ERROR")
    if r.info["sfreq"] != SF:
        r.resample(SF, verbose="ERROR")
    d = ib.select_19ch(r)
    if d is None or d.shape[1] < DUR:
        return None
    return fm._car_filt(d[:, :DUR])


def _epochs_4s(sig: np.ndarray) -> np.ndarray:
    n = sig.shape[1] // EPOCH
    return np.stack([sig[:, i * EPOCH:(i + 1) * EPOCH] for i in range(n)], 0).astype(np.float32)


# ── LaBraM fc_norm hook (mirrors src/embeddings.extract_labram_embeddings) ──────
def _labram_hook(model):
    captured: dict = {}
    target = None
    for name, mod in model.named_modules():
        if name == "fc_norm":
            target = mod
            break
    if target is None:
        import torch.nn as nn
        for _n, mod in model.named_modules():
            if isinstance(mod, nn.LayerNorm):
                target = mod
    if target is None:
        raise RuntimeError("no fc_norm/LayerNorm in LaBraM")
    target.register_forward_hook(lambda m, i, o: captured.__setitem__("f", o))
    return captured


def _labram_subject_embed(model, captured, epochs, device) -> np.ndarray:
    import torch
    outs = []
    for s in range(0, len(epochs), 32):
        captured.clear()
        with torch.no_grad():
            model(torch.from_numpy(epochs[s:s + 32].astype(np.float32)).to(device))
        f = captured["f"]
        if f.dim() == 3:
            f = f.mean(1)
        outs.append(f.cpu().numpy())
    return np.concatenate(outs).mean(0)


def extract(cache: Path):
    if cache.exists():
        d = safe_np_load(str(cache))["data"].item()
        return d
    import torch  # noqa: F401
    from src.biot_wrapper import extract_biot_epoch_embeds, load_pretrained_biot
    from src.cbramod_wrapper import (extract_cbramod_epoch_embeds,
                                     load_pretrained_cbramod)
    from src.bendr_wrapper import extract_bendr_epoch_embeds, load_pretrained_bendr
    from src.embeddings import load_labram_adapted

    p05 = safe_np_load(str(P05_CACHE))
    serials = list(p05["serials"])
    idx = {s: i for i, s in enumerate(serials)}
    sets = {next((p for p in Path(fp).parts if p.startswith("sub-")), Path(fp).stem): fp
            for fp in sorted(glob.glob(str(RAW / "**" / "*_eeg.set"), recursive=True))}

    print("[multifm] loading models (cpu) ...", flush=True)
    import mne
    cbramod = load_pretrained_cbramod(device=DEV)
    biot = load_pretrained_biot(device=DEV)
    bendr = load_pretrained_bendr(n_times=1024, device=DEV)
    # LaBraM (braindecode) needs channel positions; build chs_info from standard_1020
    # (COMMON uses legacy T3/T4/T5/T6 -> remap to the 10-20 names LaBraM knows).
    remap = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    lab_info = mne.create_info([remap.get(c, c) for c in COMMON], SF, "eeg")
    lab_info.set_montage("standard_1020", on_missing="ignore")
    labram = load_labram_adapted(n_chans=19, n_times=EPOCH, config=LABRAM_CONFIG,
                                 chs_info=lab_info["chs"])
    labram.to(DEV).eval()
    lab_cap = _labram_hook(labram)

    emb = {m: {} for m in ("LaBraM", "CBraMod", "BIOT", "BENDR")}
    n_done = 0
    for s in serials:
        fp = sets.get(s)
        if fp is None:
            print(f"  [skip {s}] no .set", flush=True)
            continue
        try:
            sig = _harmonise(fp)
            if sig is None:
                print(f"  [skip {s}] harmonise None", flush=True)
                continue
            eps = _epochs_4s(sig)                                   # (n_ep, 19, 800)
            emb["CBraMod"][s] = extract_cbramod_epoch_embeds(cbramod, eps, batch_size=8, device=DEV).mean(0)
            emb["BIOT"][s] = extract_biot_epoch_embeds(biot, eps, batch_size=8, device=DEV).mean(0)
            emb["BENDR"][s] = np.asarray(extract_bendr_epoch_embeds(
                bendr, eps, sfreq=SF, batch_size=4, device=DEV)).mean(0)
            emb["LaBraM"][s] = _labram_subject_embed(labram, lab_cap, eps, DEV)
        except Exception as e:
            print(f"  [skip {s}] {type(e).__name__}: {e}", flush=True)
            continue
        n_done += 1
        if n_done % 10 == 0:
            print(f"  ... {n_done} subjects embedded", flush=True)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, data={**emb, "_serials": serials})
    out = {**emb, "_serials": serials}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, data=out)
    print(f"[multifm] embedded {n_done} subjects for {list(emb)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--cache", type=Path,
                    default=ROOT / "results/cross_population/brainlat_multifm_embeds.npz")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/cross_population/brainlat_multifm_encoding_probe.json")
    args = ap.parse_args()
    os.chdir(ROOT)
    set_seed(0)

    emb = extract(args.cache)
    if args.extract_only:
        print("[multifm] extract-only done.", flush=True)
        return

    # targets + REVE + classical from the P0.5 cache, aligned by serial
    p05 = safe_np_load(str(P05_CACHE))
    serials = list(p05["serials"])
    y_all = {"DFA_full": p05["dfa_full"], "DFA_short": p05["dfa_short"], "1/f": p05["f1"]}
    X_ref = {"REVE_pooled": p05["pooled"], "classical": p05["classical"]}

    # keep only subjects embedded by every added FM
    added = ["LaBraM", "CBraMod", "BIOT", "BENDR"]
    keep = [i for i, s in enumerate(serials) if all(s in emb[m] for m in added)]
    ser = [serials[i] for i in keep]
    n = len(ser)
    X = {m: np.vstack([emb[m][s] for s in ser]) for m in added}
    for k, v in X_ref.items():
        X[k] = v[keep]
    y = {t: v[keep] for t, v in y_all.items()}
    print(f"[multifm] N={n}; dims={ {k: X[k].shape[1] for k in X} }", flush=True)
    print("[multifm] embedding std (scaling sanity; ~0 => saturated/broken input):", flush=True)
    for m in X:
        print(f"    {m:11s} std={X[m].std():.3e}", flush=True)

    order = ["REVE_pooled", "LaBraM", "BENDR", "CBraMod", "BIOT", "classical"]
    fronts = {"REVE_pooled": "raw-waveform", "LaBraM": "raw-patch", "BENDR": "raw-wav2vec",
              "CBraMod": "spectral-input", "BIOT": "spectral-input", "classical": "hand-computed"}
    out = {"meta": {"title": "Multi-FM encoding replication on BrainLat (raw-vs-spectral split)",
                    "cohort": "BrainLat (Latin-American AD+bvFTD/HC, OOD)", "n": n,
                    "front_end": fronts, "probes": PROBES, "cv": "5-fold KFold shuffle seed 0, R^2",
                    "dims": {k: int(X[k].shape[1]) for k in X},
                    "embed_std": {k: float(X[k].std()) for k in X},
                    "reading": ("raw-waveform FMs ~0 on all targets AND spectral FMs recover 1/f while "
                                "classical recovers => input-representation split replicates on a 2nd OOD cohort."),
                    "status": "exploratory (P0.6 multi-FM encoding replication)"},
           "cv_r2": {}}
    for m in order:
        probes = {k: v for k, v in fm.make_probes(X[m].shape[1], n).items() if k in PROBES}
        out["cv_r2"][m] = {}
        for tname, yv in y.items():
            row = {p: round(fm.cv_r2(pfn, X[m], yv), 4) for p, pfn in probes.items()}
            out["cv_r2"][m][tname] = row
            best = max(row["hist_gboost"], row["random_forest"])
            print(f"  {m:11s} ({fronts[m]:14s}) {tname:9s} | "
                  + "  ".join(f"{p}={row[p]:+.3f}" for p in PROBES) + f"   best_nl={best:+.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    try:
        from src.runs import mirror_results_file
        mirror_results_file(args.out, stem="brainlat_multifm_encoding_results")  # no-op unless POC_RUN_DIR set
    except Exception:
        pass
    print(f"[multifm] positive control = classical; wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
