#!/usr/bin/env python3
"""Short-range DFA surrogate controls on BrainLat (paper Sec 4.1 / Methods).

The 0.5-2 s DFA range used on BrainLat is only ~0.6 decades and the 8-13 Hz
bandpass imposes autocorrelation on the order of 0.2 s, so a reviewer can ask
whether the "short-range exponent" is a genuine scaling measurement or a filter
artifact. This script answers with three controls computed on the same BrainLat
alpha envelopes used for the encoding probe:

  (1) short-range log-log FIT R^2 -- is the log-log relation actually linear over
      0.5-2 s (is it a well-defined slope at all)?
  (2) filtered-white-noise NULL -- run the identical 8-13 Hz bandpass + Hilbert
      envelope + DFA_short on white noise. This isolates what the FILTER alone
      contributes. If the real exponent >> the noise-null exponent, the short-range
      measurement is not merely a filter artifact.
  (3) temporally-shuffled surrogate -- shuffle the envelope in time (destroys all
      temporal correlation). DFA_short should collapse toward 0.5, confirming the
      real value reflects temporal structure, not the amplitude distribution.

Reuses the exact BrainLat loader (_harmonise) and DFA machinery
(SHORT_SCALES, _dfa) already used for Table 2.

Run from repo root:
  /opt/anaconda3/envs/neurogenis/bin/python scripts/analysis/brainlat_dfa_short_surrogate.py
"""
from __future__ import annotations

import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(ROOT / "scripts" / "download"))
warnings.filterwarnings("ignore")

import fm_lrtc_nonlinear_probe as fm  # noqa: E402  (_dfa, SHORT_SCALES, SF, _car_filt)
import ingest_brainlat as ib          # noqa: E402  (select_19ch, from scripts/download)

RAW = ROOT / "data" / "raw" / "brainlat"
OUT = ROOT / "results" / "cross_population" / "brainlat_dfa_short_surrogate.json"
SF = fm.SF
DUR = 240 * int(SF)
SEED = 42


def _harmonise(fp: str):
    import mne
    try:
        r = mne.io.read_raw_eeglab(fp, preload=True, verbose="ERROR")
    except (FileNotFoundError, Exception):  # missing .fdt sidecar on this machine
        return None
    if r.info["sfreq"] != SF:
        r.resample(SF, verbose="ERROR")
    d = ib.select_19ch(r)
    if d is None or d.shape[1] < DUR:
        return None
    return fm._car_filt(d[:, :DUR])


def _alpha_env(x: np.ndarray) -> np.ndarray:
    ba = butter(4, [8 / (SF / 2), 13 / (SF / 2)], btype="band")
    return np.abs(hilbert(filtfilt(ba[0], ba[1], x)))


def _phase_randomise(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """FT surrogate: preserve power spectrum, randomise phases (destroys nonlinear structure)."""
    X = np.fft.rfft(x)
    ph = rng.uniform(0, 2 * np.pi, len(X))
    ph[0] = 0.0
    Xs = np.abs(X) * np.exp(1j * ph)
    return np.fft.irfft(Xs, n=len(x))


def main() -> None:
    rng = np.random.default_rng(SEED)
    sets = sorted(glob.glob(str(RAW / "**" / "*_eeg.set"), recursive=True))
    if not sets:
        sys.exit(f"No .set under {RAW}")
    print(f"BrainLat subjects: {len(sets)}", flush=True)

    real_exp, real_r2, noise_exp, shuf_exp, phase_exp = [], [], [], [], []
    n_ok = 0
    for i, fp in enumerate(sets):
        d = _harmonise(fp)
        if d is None:
            continue
        n_ok += 1
        for ch in range(19):
            env = _alpha_env(d[ch])
            e, r2 = fm._dfa(env, fm.SHORT_SCALES, with_r2=True)
            if np.isfinite(e):
                real_exp.append(e)
                real_r2.append(r2)
            # (2) filtered-white-noise null: filter+envelope pipeline on white noise
            noise = _alpha_env(rng.standard_normal(d.shape[1]))
            ne = fm._dfa(noise, fm.SHORT_SCALES)
            if np.isfinite(ne):
                noise_exp.append(ne)
            # (3) temporally-shuffled envelope
            se = fm._dfa(rng.permutation(env), fm.SHORT_SCALES)
            if np.isfinite(se):
                shuf_exp.append(se)
            # (4) phase-randomised (FT) surrogate of the envelope
            pe = fm._dfa(_phase_randomise(env, rng), fm.SHORT_SCALES)
            if np.isfinite(pe):
                phase_exp.append(pe)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(sets)} processed", flush=True)

    def summ(a):
        a = np.asarray(a, float)
        return {"mean": float(np.mean(a)), "std": float(np.std(a)), "n": int(len(a))}

    out = {
        "meta": {"title": "BrainLat short-range (0.5-2s) DFA surrogate controls",
                 "n_subjects": n_ok, "scales_samples": [int(s) for s in fm.SHORT_SCALES],
                 "sf": SF, "seed": SEED,
                 "reproduce": "python scripts/analysis/brainlat_dfa_short_surrogate.py"},
        "real_exponent": summ(real_exp),
        "real_loglog_fit_r2": summ(real_r2),
        "filtered_white_noise_null_exponent": summ(noise_exp),
        "shuffled_envelope_exponent": summ(shuf_exp),
        "phase_randomised_exponent": summ(phase_exp),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print("\n=== BrainLat DFA_short (0.5-2s) controls ===", flush=True)
    print(f"real exponent          : {out['real_exponent']['mean']:.3f} +/- {out['real_exponent']['std']:.3f}", flush=True)
    print(f"real log-log fit R^2    : {out['real_loglog_fit_r2']['mean']:.3f} +/- {out['real_loglog_fit_r2']['std']:.3f}", flush=True)
    print(f"filtered-noise null exp : {out['filtered_white_noise_null_exponent']['mean']:.3f} +/- {out['filtered_white_noise_null_exponent']['std']:.3f}", flush=True)
    print(f"shuffled-envelope exp   : {out['shuffled_envelope_exponent']['mean']:.3f} +/- {out['shuffled_envelope_exponent']['std']:.3f}", flush=True)
    print(f"phase-randomised exp    : {out['phase_randomised_exponent']['mean']:.3f} +/- {out['phase_randomised_exponent']['std']:.3f}", flush=True)
    print(f"wrote {OUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
