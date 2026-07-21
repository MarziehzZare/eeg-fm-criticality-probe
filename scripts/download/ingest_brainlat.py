"""Ingest BrainLat (Latin-American AD/bvFTD/HC EEG) as the 3rd cross-population cohort.

BrainLat (Prado et al., Scientific Data 10:889, 2023). Synapse "EEG data" folder = syn53972960.
Confirmed structure (via metadata walk + one-subject probe):
  EEG data/<GROUP>/<SITE>/sub-XXXXX/eeg/s*_sub-XXXXX_rs-HEP_eeg.set   (EEGLAB .set + .fdt)
  GROUP in {1_AD, 2_bvFTD, 3_PD, 4_MS, 5_HC}; SITE in {AR (Argentina), CL (Chile)}.
  Counts: AD 35, bvFTD 19, PD 29, MS 33, HC 46. 128-ch Biosemi ABCD labels, 512 Hz, ~312 s
  resting-state (clears the 240 s / 0.5-30 s DFA bar). Confirmed OOD for REVE.

Dementia-vs-control task (matches ds004504 / CAUEEG): AD + bvFTD = 1 (N=54), HC = 0 (N=46);
PD and MS are skipped. Labels + site come from the folder PATH (no participants.tsv needed).
Channels: 128-ch Biosemi ABCD -> the 19-ch 10-20 subset by NEAREST 3D electrode position (MNE
biosemi128 vs standard_1020 montages), NOT by label (Biosemi 'C3' != 10-20 C3).

ACCESS (user completes the DUA + auth; Claude never handles the token):
  Put the Synapse PAT in ~/.synapseConfig ([authentication] authtoken = ...), chmod 600, then
  `synapse login` prints your username. This script reads that config via login(silent=True).

USAGE:
  python scripts/download/ingest_brainlat.py --discover            # inspect remote structure (metadata)
  python scripts/download/ingest_brainlat.py --download            # pull EEG for AD/bvFTD/HC (skip PD/MS)
  python scripts/download/ingest_brainlat.py --extract             # harmonize + features -> brainlat_features.npz

STATUS: exploratory (3rd-cohort ingest for the cross-population transfer/probe pipeline).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
warnings.filterwarnings("ignore")

import cross_population_criticality as cpc  # noqa: E402  (reuse extract, _car_filt, _impute, SF, COMMON)

RAW = ROOT / "data" / "raw" / "brainlat"
SYN_EEG = "syn53972960"
SF = cpc.SF
COMMON = cpc.COMMON                                          # 19-ch 10-20 (old T3/T4/T5/T6 names)
DEMENTIA_GROUPS = {"1_AD", "2_bvFTD"}                        # label 1
CONTROL_GROUPS = {"5_HC"}                                    # label 0  (skip 3_PD, 4_MS)


def _biosemi_to_1020() -> dict[str, str]:
    """COMMON 10-20 name -> nearest Biosemi-128 ABCD label (by 3D electrode position)."""
    import mne
    bio = mne.channels.make_standard_montage("biosemi128").get_positions()["ch_pos"]
    std = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    remap10 = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}   # COMMON -> standard_1020 name
    out = {}
    for ch in COMMON:
        p = std[remap10.get(ch, ch)]
        out[ch] = min(bio, key=lambda b: float(np.linalg.norm(bio[b] - p)))
    return out


BIO_MAP = None  # lazily built (needs mne)


def select_19ch(raw) -> np.ndarray | None:
    """128-ch Biosemi .set -> [19, T] in COMMON order via nearest-position Biosemi labels."""
    global BIO_MAP
    if BIO_MAP is None:
        BIO_MAP = _biosemi_to_1020()
    idx = {c: i for i, c in enumerate(raw.ch_names)}
    order = []
    for ch in COMMON:
        b = BIO_MAP[ch]
        if b not in idx:
            return None
        order.append(idx[b])
    return raw.get_data()[order]


# ── Synapse download (EEG only, AD/bvFTD/HC; skip PD/MS) ─────────────────────────
def _login():
    try:
        import synapseclient
    except ImportError:
        sys.exit("synapseclient not installed. `pip install synapseclient`.")
    return synapseclient.login(silent=True)


def download():
    import synapseutils
    syn = _login()
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"[brainlat] syncing {SYN_EEG} (AD/bvFTD/HC only) -> {RAW}", flush=True)
    n = 0
    for dirpath, _dirs, files in synapseutils.walk(syn, SYN_EEG):
        path = dirpath[0]
        grp = path.split("/")[1] if "/" in path else ""
        if grp not in (DEMENTIA_GROUPS | CONTROL_GROUPS):
            continue
        for fname, fid in files:
            if fname.lower().endswith((".set", ".fdt")):
                # mirror the group/site/subject subpath locally
                rel = path.replace("EEG data/", "").replace("/", os.sep)
                dest = RAW / rel
                dest.mkdir(parents=True, exist_ok=True)
                syn.get(fid, downloadLocation=str(dest), ifcollision="keep.local")
                n += 1
                if n % 20 == 0:
                    print(f"  ... {n} files", flush=True)
    print(f"[brainlat] download complete ({n} files)", flush=True)


def discover():
    """Metadata-only remote walk (no download) — group/site/format sanity check."""
    import collections
    import synapseutils
    syn = _login()
    grp, site, ext = collections.Counter(), collections.Counter(), collections.Counter()
    for dirpath, _dirs, files in synapseutils.walk(syn, SYN_EEG):
        parts = dirpath[0].split("/")
        for fname, _fid in files:
            e = "." + fname.split(".")[-1].lower() if "." in fname else "noext"
            ext[e] += 1
            if fname.lower().endswith(".set") and len(parts) > 2:
                grp[parts[1]] += 1
                site[parts[2]] += 1
    print("groups(.set):", dict(grp), "\nsites(.set):", dict(site), "\nextensions:", dict(ext))


# ── harmonize + extract features (labels/site from PATH) ────────────────────────
def extract_features(max_subjects: int | None):
    import mne
    mne.set_log_level("ERROR")
    sets = sorted(glob.glob(str(RAW / "**" / "*_eeg.set"), recursive=True))
    if not sets:
        sys.exit(f"No .set under {RAW} — run --download first.")
    X, y, site, serials = [], [], [], []
    for fp in sets:
        parts = Path(fp).parts
        grp = next((p for p in parts if p in (DEMENTIA_GROUPS | CONTROL_GROUPS)), None)
        st = next((p for p in parts if p in ("AR", "CL")), "NA")
        if grp is None:
            continue
        lab = 1 if grp in DEMENTIA_GROUPS else 0
        sid = next((p for p in parts if p.startswith("sub-")), Path(fp).stem)
        try:
            r = mne.io.read_raw_eeglab(fp, preload=True, verbose="ERROR")
            if r.info["sfreq"] != SF:
                r.resample(SF, verbose="ERROR")
            d = select_19ch(r)
            if d is None or d.shape[1] < int(240 * SF):
                print(f"  [skip {sid}] short/missing-ch ({None if d is None else d.shape[1]})", flush=True)
                continue
            e, o, fa = cpc.extract(cpc._car_filt(d[:, :int(240 * SF)]))
        except Exception as ex:
            print(f"  [skip {sid}] {type(ex).__name__}: {ex}", flush=True)
            continue
        X.append(np.concatenate([e, o, fa])); y.append(lab); site.append(st); serials.append(sid)
        if len(serials) % 10 == 0:
            print(f"  ... harmonized {len(serials)}", flush=True)
        if max_subjects and len(serials) >= max_subjects:
            break
    if not serials:
        sys.exit("No subjects harmonized — check channel mapping / download.")
    Xarr = cpc._impute(np.array(X))
    outp = ROOT / "results/cross_population/brainlat_features.npz"
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(outp, X=Xarr, y=np.array(y), site=np.array(site), serials=np.array(serials))
    n1 = int(np.sum(y))
    import collections
    print(f"[brainlat] N={len(serials)} ({n1} dementia / {len(serials)-n1} HC); "
          f"sites={dict(collections.Counter(site))}; features 57d. wrote {outp}")
    print("  NEXT: 3-cohort leave-one-cohort-out transfer (add BrainLat as cohort 3, site as covariate) "
          "+ BrainLat REVE embeds for the pre-pool probe.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--max-subjects", type=int, default=None)
    args = ap.parse_args()
    os.chdir(ROOT)
    if args.discover:
        discover()
    if args.download:
        download()
    if args.extract:
        extract_features(args.max_subjects)
    if not (args.discover or args.download or args.extract):
        ap.print_help()


if __name__ == "__main__":
    main()
