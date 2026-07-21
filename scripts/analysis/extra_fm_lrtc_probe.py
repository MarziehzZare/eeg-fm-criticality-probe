"""Extend the LRTC encoding probe to more EEG foundation models: BENDR and LEAD.

Adds two more architectures to the four already tested (REVE, CBraMod, LaBraM, BIOT):
  - BENDR  : wav2vec-style contrastive FM (braindecode), 512-d pooled
  - LEAD    : recent Alzheimer's-focused EEG FM (cloned repo under models/LEAD)

Same probe as P0.1: on CAUEEG (cached DFA/1f targets), can a frozen embedding recover the
per-subject DFA_full (0.5-30s), DFA_short (0.5-2s), or 1/f scalar? (linear + gboost + RF CV R^2).
classical is the positive control (must recover DFA/1f, else the run is uninterpretable).

SCALING SAFEGUARD (the µV/Volt lesson): inputs are fed in Volts (EDF get_data, the project
convention that worked for REVE). We ALSO report each model's embedding std across subjects — if a
model's embeddings barely vary (saturated) OR it recovers nothing including 1/f while classical does,
that flags a possible input-scale mismatch rather than genuine blindness, and we retry in µV.

USAGE: /opt/anaconda3/envs/neurogenis/bin/python scripts/analysis/extra_fm_lrtc_probe.py [--max-subjects N] [--uv]
STATUS: exploratory (model-generality extension of the mechanism arm).
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

import mne  # noqa: E402

import fm_lrtc_nonlinear_probe as p01  # noqa: E402
from src.io_safety import safe_np_load  # noqa: E402

mne.set_log_level("ERROR")
SF = p01.SF


def epochs_4s(sig):
    n = sig.shape[1] // 800
    return np.stack([sig[:, i * 800:(i + 1) * 800] for i in range(n)], axis=0).astype(np.float32)


def load_caueeg_epochs(serial, uv_scale):
    fp = f"{p01.CAUEEG_ROOT}/signal/edf/{serial}.edf"
    if not os.path.exists(fp):
        return None
    r = mne.io.read_raw_edf(fp, preload=False, verbose="ERROR")
    r.crop(tmax=min(240.0, (r.n_times - 1) / r.info["sfreq"]))
    r.load_data(verbose="ERROR")
    oc = p01._order(r.ch_names)
    if oc is None:
        return None
    if r.info["sfreq"] != SF:
        r.resample(SF, verbose="ERROR")
    d = r.get_data()[oc][:, :int(240 * SF)]
    if d.shape[1] < int(240 * SF):
        return None
    sig = p01._car_filt(d)
    if uv_scale:
        sig = sig * 1e6                     # Volts -> microVolts
    return epochs_4s(sig)


def build(max_subjects, uv, cache: Path):
    tgt_path = str(ROOT / "results/cross_population/p01_targets.npz")
    tcache = {k: list(v) for k, v in safe_np_load(tgt_path)["data"].item().items()}
    serials_all = list(tcache.keys())[:max_subjects]
    multi = safe_np_load("data/embeddings/caueeg_3way_multiepoch_19ch_all.npz")["data"].item()

    def classical_vec(s):
        v = multi["classical"]["subject_mean_embeds"][s]
        v = v.item() if getattr(v, "shape", None) == () else v
        return np.asarray(v["embedding"] if isinstance(v, dict) else v).ravel()

    emb_cache = {}
    if cache.exists():
        emb_cache = {k: {kk: np.asarray(vv) for kk, vv in v.items()}
                     for k, v in safe_np_load(str(cache))["data"].item().items()}

    # load models
    from src.bendr_wrapper import extract_bendr_epoch_embeds, load_pretrained_bendr
    bendr = load_pretrained_bendr(n_times=1024, device="cpu")
    lead = None
    try:
        from src.lead_wrapper import load_lead_wrapper
        lead = load_lead_wrapper(n_channels=19, seq_len=800, channel_names=p01.COMMON)
    except Exception as e:
        print(f"[extra] LEAD unavailable ({type(e).__name__}: {e}); BENDR + classical only", flush=True)

    ser, DL, DS, ONEF = [], [], [], []
    Xb, Xl, Xc = [], [], []
    n_new = 0
    for s in serials_all:
        if s not in multi["classical"]["subject_mean_embeds"]:
            continue
        rec = emb_cache.get(s, {})
        need = ("BENDR" not in rec) or (lead is not None and "LEAD" not in rec)
        eps = load_caueeg_epochs(s, uv) if need else None
        if need and eps is None:
            continue
        try:
            if "BENDR" not in rec:
                rec["BENDR"] = np.asarray(extract_bendr_epoch_embeds(
                    bendr, eps, sfreq=SF, batch_size=4, device="cpu")).mean(0)
            if lead is not None and "LEAD" not in rec:
                rec["LEAD"] = np.asarray(lead.extract(eps, sfreq=SF)).mean(0)
        except Exception as e:
            print(f"  [skip {s}] {type(e).__name__}: {e}", flush=True)
            continue
        emb_cache[s] = rec
        ser.append(s)
        DL.append(tcache[s][0]); DS.append(tcache[s][1]); ONEF.append(tcache[s][2])
        Xb.append(rec["BENDR"])
        if lead is not None:
            Xl.append(rec["LEAD"])
        Xc.append(classical_vec(s))
        n_new += 1
        if n_new % 20 == 0:
            print(f"  ... {n_new} subjects embedded", flush=True)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, data=emb_cache)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, data=emb_cache)

    X = {"BENDR": np.vstack(Xb), "classical": np.vstack(Xc)}
    if Xl:
        X["LEAD"] = np.vstack(Xl)
    y = {"DFA_full": np.array(DL), "DFA_short": np.array(DS), "1/f": np.array(ONEF)}
    return ser, X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-subjects", type=int, default=200)
    ap.add_argument("--uv", action="store_true", help="feed inputs in microVolts (scaling retry)")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/cross_population/extra_fm_lrtc_probe.json")
    ap.add_argument("--cache", type=Path,
                    default=ROOT / "results/cross_population/extra_fm_embeds.npz")
    args = ap.parse_args()
    os.chdir(ROOT)

    print(f"[extra] building (max={args.max_subjects}, units={'uV' if args.uv else 'V'}) ...", flush=True)
    ser, X, y = build(args.max_subjects, args.uv, args.cache)
    n = len(ser)
    PROBES = ["ridge_linear_full", "hist_gboost", "random_forest"]
    order = [m for m in ("BENDR", "LEAD", "classical") if m in X]
    print(f"[extra] N={n}; dims={ {k: X[k].shape[1] for k in order} }", flush=True)
    print("[extra] embedding std across subjects (scaling sanity; ~0 => saturated/broken input):", flush=True)
    for m in order:
        print(f"    {m:9s} std={X[m].std():.3e}", flush=True)

    out = {"meta": {"title": "Extra EEG-FM LRTC probe (BENDR, LEAD) on CAUEEG",
                    "n": n, "units": "uV" if args.uv else "V", "probes": PROBES,
                    "dims": {k: int(X[k].shape[1]) for k in order},
                    "embed_std": {m: float(X[m].std()) for m in order},
                    "status": "exploratory (model-generality extension)"},
           "cv_r2": {}}
    for m in order:
        probes = {k: v for k, v in p01.make_probes(X[m].shape[1], n).items() if k in PROBES}
        out["cv_r2"][m] = {}
        for tname, yt in y.items():
            row = {p: round(p01.cv_r2(pfn, X[m], yt), 4) for p, pfn in probes.items()}
            out["cv_r2"][m][tname] = row
            print(f"  {m:9s} {tname:9s} | " + "  ".join(f"{p}={row[p]:+.3f}" for p in PROBES), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    try:
        from src.runs import mirror_results_file
        mirror_results_file(args.out, stem="extra_fm_lrtc_probe_results")  # no-op unless POC_RUN_DIR set
    except Exception:
        pass
    print(f"[extra] positive control = classical; wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
