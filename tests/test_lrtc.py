"""Tests for the fixed revised Paper 02 LRTC estimator."""

import numpy as np

from src.lrtc import (
    LRTCConfig,
    admissible_scales,
    alpha_peak_snr,
    dfa,
    estimate_alpha_envelope_dfa,
    robust_artifact_mask,
)


def _generate_fgn(n: int, hurst: float, rng: np.random.Generator) -> np.ndarray:
    """Davies-Harte fractional Gaussian noise for estimator validation."""

    k = np.arange(n)
    gamma = 0.5 * (
        np.abs(k - 1) ** (2 * hurst)
        - 2 * np.abs(k) ** (2 * hurst)
        + np.abs(k + 1) ** (2 * hurst)
    )
    row = np.concatenate([gamma, gamma[-2:0:-1]])
    eigenvalues = np.clip(np.fft.fft(row).real, 0, None)
    m = len(row)
    z = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    return np.real(np.fft.fft(np.sqrt(eigenvalues / (2 * m)) * z)[:n])


def test_primary_filter_spans_two_low_frequency_cycles():
    config = LRTCConfig()
    assert config.filter_numtaps == 51
    assert config.filter_numtaps / config.sfreq >= 2 / config.alpha_low_hz


def test_scales_never_exceed_ten_percent_of_analyzable_duration():
    config = LRTCConfig()
    n_samples = int(240 * config.sfreq)
    scales = admissible_scales(n_samples, config)
    analyzable = n_samples - 2 * round(config.edge_trim_seconds * config.sfreq)
    assert scales[0] >= round(2 * config.sfreq)
    assert scales[-1] <= np.floor(0.10 * analyzable)


def test_direct_dfa_tracks_fractional_gaussian_noise_hurst_exponent():
    rng = np.random.default_rng(7)
    scales = np.unique(np.rint(np.geomspace(32, 1024, 14)).astype(int))
    estimates = []
    requested = np.linspace(0.55, 0.90, 12)
    for hurst in requested:
        result = dfa(
            _generate_fgn(32768, hurst, rng),
            sfreq=1.0,
            scales=scales,
            min_windows_per_scale=8,
        )
        estimates.append(result.exponent)
    assert np.corrcoef(requested, estimates)[0, 1] > 0.90
    assert abs(np.mean(np.asarray(estimates) - requested)) < 0.08


def test_filtered_white_noise_envelope_is_near_uncorrelated_above_two_seconds():
    rng = np.random.default_rng(11)
    config = LRTCConfig()
    estimates = []
    for _ in range(6):
        white = rng.standard_normal(int(240 * config.sfreq))
        result = estimate_alpha_envelope_dfa(white, config)
        estimates.append(result.exponent[0])
    assert 0.42 < np.mean(estimates) < 0.62


def test_artifact_mask_marks_transient_and_padding():
    rng = np.random.default_rng(13)
    sfreq = 200.0
    data = rng.standard_normal((2, 4000)) * 20e-6
    data[0, 2000] = 2e-3
    mask = robust_artifact_mask(data, sfreq, pad_seconds=0.25)
    assert not mask[0, 1950:2051].any()
    assert mask[1].mean() > 0.99


def test_heavily_contaminated_channel_fails_explicitly():
    rng = np.random.default_rng(17)
    config = LRTCConfig()
    data = rng.standard_normal(int(240 * config.sfreq))
    clean = np.ones_like(data, dtype=bool)
    clean[::2] = False
    result = estimate_alpha_envelope_dfa(data, config, clean_mask=clean)
    assert np.isnan(result.exponent[0])
    assert result.clean_sample_fraction[0] == 0.5


def test_alpha_peak_snr_detects_prominent_oscillation():
    rng = np.random.default_rng(19)
    sfreq = 200.0
    time = np.arange(int(60 * sfreq)) / sfreq
    noise = rng.standard_normal(len(time))
    oscillatory = noise + 3 * np.sin(2 * np.pi * 10 * time)
    snr = alpha_peak_snr(np.vstack([noise, oscillatory]), sfreq)
    assert snr[1] > snr[0] * 3
