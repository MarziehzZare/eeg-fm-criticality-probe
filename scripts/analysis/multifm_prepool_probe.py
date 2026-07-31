"""Build multi-model pre-pool sequence caches for the Paper-02 v2 order probe.

For v2 reproduction, invoke this script with ``--extract-only``.  It extracts
LaBraM, CBraMod, BIOT, and BENDR per-epoch sequences over the first 240 seconds.
The canonical matched-realization analysis is implemented in
``paper02_prepool_order_permutation.py``.

The historical exploratory readout retained below is not the v2 inferential
analysis.  In particular, an ordered-versus-shuffled difference is descriptive:
preserving chronology also preserves non-stationarity, arousal drift, artifact
clustering, autocorrelation, and recording-position effects, so it does not
isolate LRTC.

Usage:
    python scripts/analysis/multifm_prepool_probe.py --extract-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import brainlat_multifm_encoding_probe as bm   # noqa: E402  _labram_hook, LABRAM_CONFIG, DEV
import extra_fm_lrtc_probe as ex               # noqa: E402  load_caueeg_epochs
import fm_lrtc_nonlinear_probe as fm           # noqa: E402  SF, COMMON, CAUEEG_ROOT
import reve_prepool_token_probe as pp          # noqa: E402  cnn_cv_r2, ridge_cv_r2
from src.io_safety import safe_np_load         # noqa: E402
from src.seed import set_seed                  # noqa: E402

SF = fm.SF
COMMON = fm.COMMON
DEV = bm.DEV
P0 = 60                                         # first 240 s = 60 x 4 s epochs (matches DFA target window)


def _labram_epoch_seq(model, captured, epochs, device) -> np.ndarray:
    """Per-epoch LaBraM embeddings (n_ep, 200) — like bm._labram_subject_embed but NOT meaned."""
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
    return np.concatenate(outs)


def extract(cache: Path):
    if cache.exists():
        return safe_np_load(str(cache))["data"].item()
    import mne  # noqa: F401
    from src.biot_wrapper import extract_biot_epoch_embeds, load_pretrained_biot
    from src.cbramod_wrapper import (extract_cbramod_epoch_embeds,
                                     load_pretrained_cbramod)
    from src.bendr_wrapper import extract_bendr_epoch_embeds, load_pretrained_bendr
    from src.embeddings import load_labram_adapted

    tcache = safe_np_load("results/cross_population/p01_targets.npz")["data"].item()
    serials = list(tcache.keys())

    print("[prepool-mf] loading models (cpu) ...", flush=True)
    cbramod = load_pretrained_cbramod(device=DEV)
    biot = load_pretrained_biot(device=DEV)
    bendr = load_pretrained_bendr(n_times=1024, device=DEV)
    import mne
    remap = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    lab_info = mne.create_info([remap.get(c, c) for c in COMMON], SF, "eeg")
    lab_info.set_montage("standard_1020", on_missing="ignore")
    labram = load_labram_adapted(n_chans=19, n_times=int(4 * SF), config=bm.LABRAM_CONFIG,
                                 chs_info=lab_info["chs"])
    labram.to(DEV).eval()
    lab_cap = bm._labram_hook(labram)

    seqs = {m: {} for m in ("LaBraM", "CBraMod", "BIOT", "BENDR")}
    dfa, ser = {}, []
    n_done = 0
    for s in serials:
        eps = ex.load_caueeg_epochs(s, uv_scale=False)          # (n_ep, 19, 800) first 240 s
        if eps is None or eps.shape[0] < P0:
            continue
        eps = eps[:P0]
        try:
            seqs["CBraMod"][s] = extract_cbramod_epoch_embeds(cbramod, eps, batch_size=8, device=DEV)[:P0]
            seqs["BIOT"][s] = extract_biot_epoch_embeds(biot, eps, batch_size=8, device=DEV)[:P0]
            seqs["BENDR"][s] = np.asarray(extract_bendr_epoch_embeds(
                bendr, eps, sfreq=SF, batch_size=4, device=DEV))[:P0]
            seqs["LaBraM"][s] = _labram_epoch_seq(labram, lab_cap, eps, DEV)[:P0]
        except Exception as e:
            print(f"  [skip {s}] {type(e).__name__}: {e}", flush=True)
            continue
        dfa[s] = float(tcache[s][0]); ser.append(s)
        n_done += 1
        if n_done % 20 == 0:
            print(f"  ... {n_done} subjects", flush=True)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, data={"seqs": seqs, "dfa": dfa, "serials": ser})
    out = {"seqs": seqs, "dfa": dfa, "serials": ser}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, data=out)
    print(f"[prepool-mf] extracted {len(ser)} subjects", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--seeds", type=int, default=5, help="repeated CNN seeds for ordered/shuffled mean+/-std")
    ap.add_argument("--cache", type=Path,
                    default=ROOT / "results/cross_population/multifm_prepool_seqs.npz")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/cross_population/multifm_prepool_probe.json")
    args = ap.parse_args()
    os.chdir(ROOT)
    set_seed(0)

    data = extract(args.cache)
    if args.extract_only:
        print("[prepool-mf] extract-only done.", flush=True)
        return
    seqs, dfa_d, ser = data["seqs"], data["dfa"], list(data["serials"])
    n = len(ser)
    y = np.array([dfa_d[s] for s in ser])

    # classical pooled positive control on the same subjects
    multi = safe_np_load("data/embeddings/caueeg_3way_multiepoch_19ch_all.npz")["data"].item()

    def classical_vec(s):
        v = multi["classical"]["subject_mean_embeds"][s]
        v = v.item() if getattr(v, "shape", None) == () else v
        return np.asarray(v["embedding"] if isinstance(v, dict) else v).ravel()
    classical = np.vstack([classical_vec(s) for s in ser])

    fronts = {"LaBraM": "raw-patch", "BENDR": "raw-wav2vec",
              "CBraMod": "spectral-input", "BIOT": "spectral-input"}
    print(f"[prepool-mf] N={n}; DFA mean={y.mean():.3f} std={y.std():.3f}; P0={P0}; device={DEV}", flush=True)

    seeds = list(range(args.seeds))
    out = {"meta": {"title": "Multi-FM pre-pool ordered-vs-shuffled probe (CAUEEG) — anti-pooling homework",
                    "n": n, "seq_len": P0, "target": "DFA_full", "n_seeds": len(seeds),
                    "front_end": fronts, "device": str(DEV),
                    "reading": ("ordered>>shuffled => model encodes across-epoch LRTC pooling discards; "
                                "ordered~shuffled (gap within noise) while classical recovers => genuinely "
                                "not represented / order-independent proxy (not a pooling artifact). REVE "
                                "reported separately (finer token-level test)."),
                    "note_reve": ("REVE covered by reve_prepool_token_probe.json (token-level P=256, N up to "
                                  "300): ordered -0.02 ~ shuffled -0.02 ~ pooled ~0, classical 0.30."),
                    "hardening": (f"ordered/shuffled repeated over {len(seeds)} seeds (varying fold split, "
                                  "shuffle permutation, CNN init/batch order together); mean+/-std reported. "
                                  "gap_mean/gap_std quantify whether ordered>shuffled survives seed noise."),
                    "status": "exploratory (P0.7 multi-FM anti-pooling, repeated-seed hardened)"},
           "prepool": {}}
    for m in ("LaBraM", "BENDR", "CBraMod", "BIOT"):
        S = np.stack([seqs[m][s] for s in ser]).astype(np.float32)   # (N, P0, d)
        ordered = np.array([pp.cnn_cv_r2(S, y, shuffle_order=False, seed=sd) for sd in seeds])
        shuffled = np.array([pp.cnn_cv_r2(S, y, shuffle_order=True, seed=sd) for sd in seeds])
        gap = ordered - shuffled
        row = {"cnn_ordered_mean": round(float(ordered.mean()), 4),
               "cnn_ordered_std": round(float(ordered.std()), 4),
               "cnn_shuffled_mean": round(float(shuffled.mean()), 4),
               "cnn_shuffled_std": round(float(shuffled.std()), 4),
               "order_gap_mean": round(float(gap.mean()), 4),
               "order_gap_std": round(float(gap.std()), 4),
               "cnn_ordered_seeds": [round(float(v), 4) for v in ordered],
               "cnn_shuffled_seeds": [round(float(v), 4) for v in shuffled],
               "pooled_ridge": round(pp.ridge_cv_r2(S.mean(1), y), 4),
               "classical_ridge": round(pp.ridge_cv_r2(classical, y), 4)}
        out["prepool"][m] = row
        print(f"  {m:9s} ({fronts[m]:14s}) | ordered={row['cnn_ordered_mean']:+.3f}"
              f"+/-{row['cnn_ordered_std']:.3f}  shuffled={row['cnn_shuffled_mean']:+.3f}"
              f"+/-{row['cnn_shuffled_std']:.3f}  gap={row['order_gap_mean']:+.3f}"
              f"+/-{row['order_gap_std']:.3f}  pooled={row['pooled_ridge']:+.3f}  "
              f"classical={row['classical_ridge']:+.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    try:
        from src.runs import mirror_results_file
        mirror_results_file(args.out, stem="multifm_prepool_results")  # no-op unless POC_RUN_DIR set
    except Exception:
        pass
    print(f"[prepool-mf] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
