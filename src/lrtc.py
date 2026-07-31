"""Validated alpha-envelope long-range temporal correlation (LRTC) utilities.

The primary estimator follows the practical recommendations in Hardstone et al.
(2012):

* a linear-phase FIR band-pass filter whose order spans two cycles of the
  lowest pass-band frequency;
* Hilbert amplitude envelopes;
* DFA-1 with logarithmically spaced scales and 50% overlapping windows;
* a lower fitting bound of 2 s for the 8--13 Hz filter; and
* an upper bound no larger than 10% of the analyzable recording.

Artifact detection is intentionally separate from estimation.  Callers may
provide a manually curated mask or use :func:`robust_artifact_mask` as a
documented automatic sensitivity analysis.  Bad samples are interpolated only
to prevent filter ringing; DFA windows containing too many bad samples remain
excluded.

Reference
---------
Hardstone R, Poil S-S, Schiavone G, et al. Detrended fluctuation analysis: a
scale-free view on neuronal oscillations. Front Physiol. 2012;3:450.
doi:10.3389/fphys.2012.00450
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import filtfilt, firwin, hilbert, welch


@dataclass(frozen=True)
class LRTCConfig:
    """Prespecified settings for the Paper 02 primary LRTC estimand."""

    sfreq: float = 200.0
    alpha_low_hz: float = 8.0
    alpha_high_hz: float = 13.0
    filter_cycles: float = 2.0
    fit_min_seconds: float = 2.0
    fit_max_seconds: float = 25.0
    max_scale_fraction: float = 0.10
    n_scales: int = 14
    overlap: float = 0.50
    edge_trim_seconds: float = 1.0
    min_clean_fraction_per_window: float = 0.95
    min_windows_per_scale: int = 8
    min_valid_scales: int = 8

    @property
    def filter_numtaps(self) -> int:
        """Odd FIR length spanning at least ``filter_cycles`` low-band cycles."""

        samples = int(np.ceil(self.filter_cycles * self.sfreq / self.alpha_low_hz))
        return samples if samples % 2 == 1 else samples + 1

    def validate(self) -> None:
        if self.sfreq <= 0:
            raise ValueError("sfreq must be positive")
        if not 0 < self.alpha_low_hz < self.alpha_high_hz < self.sfreq / 2:
            raise ValueError("alpha band must lie strictly inside (0, Nyquist)")
        if self.fit_min_seconds <= 0 or self.fit_max_seconds <= self.fit_min_seconds:
            raise ValueError("invalid DFA fitting interval")
        if not 0 <= self.overlap < 1:
            raise ValueError("overlap must be in [0, 1)")
        if not 0 < self.max_scale_fraction <= 0.25:
            raise ValueError("max_scale_fraction must be in (0, 0.25]")
        if not 0 < self.min_clean_fraction_per_window <= 1:
            raise ValueError("min_clean_fraction_per_window must be in (0, 1]")
        if self.n_scales < self.min_valid_scales or self.min_valid_scales < 3:
            raise ValueError("need at least three valid DFA scales")


@dataclass(frozen=True)
class DFAResult:
    """One-channel DFA estimate and its audit trail."""

    exponent: float
    fit_r2: float
    scales_seconds: np.ndarray
    fluctuations: np.ndarray
    windows_per_scale: np.ndarray
    clean_sample_fraction: float
    n_valid_scales: int


@dataclass(frozen=True)
class AlphaEnvelopeDFAResult:
    """Per-channel alpha-envelope DFA estimates."""

    exponent: np.ndarray
    fit_r2: np.ndarray
    clean_sample_fraction: np.ndarray
    n_valid_scales: np.ndarray
    filter_numtaps: int
    effective_duration_seconds: float
    config: dict


def _as_2d(data: np.ndarray) -> tuple[np.ndarray, bool]:
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        return data[None, :], True
    if data.ndim != 2:
        raise ValueError("data must have shape [time] or [channels, time]")
    return data, False


def _as_mask_2d(mask: np.ndarray) -> tuple[np.ndarray, bool]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 1:
        return mask[None, :], True
    if mask.ndim != 2:
        raise ValueError("mask must have shape [time] or [channels, time]")
    return mask, False


def robust_artifact_mask(
    data: np.ndarray,
    sfreq: float,
    *,
    amplitude_z: float = 8.0,
    derivative_z: float = 10.0,
    absolute_limit_uv: float | None = 500.0,
    pad_seconds: float = 0.25,
) -> np.ndarray:
    """Return a channel-specific boolean mask of samples suitable for analysis.

    Detection combines robust amplitude and first-difference thresholds.  The
    absolute threshold assumes EEG values in volts; set it to ``None`` for data
    in unknown units.  This is a transparent automatic QC aid, not a substitute
    for manual review or ocular/muscle component removal.
    """

    x, was_1d = _as_2d(data)
    if sfreq <= 0:
        raise ValueError("sfreq must be positive")
    if x.shape[1] < 3:
        raise ValueError("at least three samples are required")

    finite = np.isfinite(x)
    x_safe = np.where(finite, x, np.nan)
    med = np.nanmedian(x_safe, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(x_safe - med), axis=1, keepdims=True)
    robust_sd = 1.4826 * mad
    fallback_sd = np.nanstd(x_safe, axis=1, keepdims=True)
    robust_sd = np.where(robust_sd > np.finfo(float).eps, robust_sd, fallback_sd)
    robust_sd = np.where(robust_sd > np.finfo(float).eps, robust_sd, 1.0)

    bad = ~finite | (np.abs(x - med) > amplitude_z * robust_sd)
    if absolute_limit_uv is not None:
        bad |= np.abs(x) > absolute_limit_uv * 1e-6

    dx = np.diff(x, axis=1, prepend=x[:, :1])
    dx_med = np.nanmedian(np.where(np.isfinite(dx), dx, np.nan), axis=1, keepdims=True)
    dx_mad = np.nanmedian(
        np.abs(np.where(np.isfinite(dx), dx, np.nan) - dx_med),
        axis=1,
        keepdims=True,
    )
    dx_sd = 1.4826 * dx_mad
    dx_fallback = np.nanstd(np.where(np.isfinite(dx), dx, np.nan), axis=1, keepdims=True)
    dx_sd = np.where(dx_sd > np.finfo(float).eps, dx_sd, dx_fallback)
    dx_sd = np.where(dx_sd > np.finfo(float).eps, dx_sd, 1.0)
    bad |= np.abs(dx - dx_med) > derivative_z * dx_sd

    pad = int(round(pad_seconds * sfreq))
    if pad > 0:
        kernel = np.ones(2 * pad + 1, dtype=np.int16)
        bad = np.vstack(
            [np.convolve(row.astype(np.int16), kernel, mode="same") > 0 for row in bad]
        )

    clean = ~bad
    return clean[0] if was_1d else clean


def interpolate_masked_samples(data: np.ndarray, clean_mask: np.ndarray) -> np.ndarray:
    """Linearly interpolate rejected samples without changing the exclusion mask."""

    x, was_1d = _as_2d(data)
    mask, _ = _as_mask_2d(clean_mask)
    if mask.shape != x.shape:
        raise ValueError("clean_mask must match data shape")

    out = x.copy()
    sample_index = np.arange(x.shape[1])
    for channel in range(x.shape[0]):
        good = mask[channel] & np.isfinite(x[channel])
        if good.sum() < 2:
            raise ValueError(f"channel {channel} has fewer than two clean samples")
        out[channel, ~good] = np.interp(
            sample_index[~good],
            sample_index[good],
            x[channel, good],
        )
    return out[0] if was_1d else out


def alpha_envelope(
    data: np.ndarray,
    config: LRTCConfig,
    *,
    clean_mask: np.ndarray | None = None,
) -> np.ndarray:
    """FIR-filter EEG and return its Hilbert amplitude envelope."""

    config.validate()
    x, was_1d = _as_2d(data)
    if clean_mask is None:
        mask = np.isfinite(x)
    else:
        mask, _ = _as_mask_2d(clean_mask)
        if mask.shape != x.shape:
            raise ValueError("clean_mask must match data shape")
        mask &= np.isfinite(x)

    x_fill = interpolate_masked_samples(x, mask)
    taps = firwin(
        config.filter_numtaps,
        [config.alpha_low_hz, config.alpha_high_hz],
        pass_zero=False,
        fs=config.sfreq,
        window="hamming",
    )
    padlen = 3 * (len(taps) - 1)
    if x.shape[1] <= padlen:
        raise ValueError(
            f"signal has {x.shape[1]} samples; more than {padlen} are required"
        )
    filtered = filtfilt(taps, [1.0], x_fill, axis=-1, padlen=padlen)
    envelope = np.abs(hilbert(filtered, axis=-1))
    return envelope[0] if was_1d else envelope


def alpha_peak_snr(
    data: np.ndarray,
    sfreq: float,
    *,
    alpha_band: tuple[float, float] = (8.0, 13.0),
    fit_band: tuple[float, float] = (2.0, 30.0),
    exclusion_band: tuple[float, float] = (7.0, 14.0),
) -> np.ndarray:
    """Estimate alpha-band power relative to a fitted aperiodic background.

    A log--log linear background is fitted outside 7--14 Hz.  The returned
    linear ratio is observed mean alpha power divided by the background power
    predicted at the same frequencies.  This is an interpretable spectral
    prominence QC measure, not a formal measurement-error model.
    """

    x, _ = _as_2d(data)
    frequency, power = welch(
        x,
        fs=sfreq,
        nperseg=min(x.shape[1], int(round(4 * sfreq))),
        axis=-1,
    )
    fit = (
        (frequency >= fit_band[0])
        & (frequency <= fit_band[1])
        & ~(
            (frequency >= exclusion_band[0])
            & (frequency <= exclusion_band[1])
        )
    )
    alpha = (frequency >= alpha_band[0]) & (frequency <= alpha_band[1])
    if fit.sum() < 3 or alpha.sum() < 1:
        raise ValueError("frequency grid cannot support alpha SNR estimation")

    log_frequency_fit = np.log10(frequency[fit])
    design = np.column_stack([np.ones(fit.sum()), log_frequency_fit])
    ratios = []
    for channel_power in power:
        positive = np.maximum(channel_power, np.finfo(float).tiny)
        coefficients = np.linalg.lstsq(
            design,
            np.log10(positive[fit]),
            rcond=None,
        )[0]
        predicted_log = (
            coefficients[0] + coefficients[1] * np.log10(frequency[alpha])
        )
        predicted = 10**predicted_log
        ratios.append(float(np.mean(positive[alpha] / predicted)))
    return np.asarray(ratios)


def admissible_scales(n_samples: int, config: LRTCConfig) -> np.ndarray:
    """Return unique DFA window lengths satisfying the prespecified bounds."""

    config.validate()
    trim = int(round(config.edge_trim_seconds * config.sfreq))
    analyzable = n_samples - 2 * trim
    if analyzable <= 0:
        raise ValueError("edge trimming leaves no analyzable samples")

    lower = int(np.ceil(config.fit_min_seconds * config.sfreq))
    requested_upper = int(np.floor(config.fit_max_seconds * config.sfreq))
    fraction_upper = int(np.floor(config.max_scale_fraction * analyzable))
    upper = min(requested_upper, fraction_upper)
    if upper <= lower:
        raise ValueError("recording is too short for the requested DFA fitting range")

    scales = np.unique(
        np.rint(np.geomspace(lower, upper, config.n_scales)).astype(int)
    )
    if len(scales) < config.min_valid_scales:
        raise ValueError("too few unique DFA scales")
    return scales


def dfa(
    series: np.ndarray,
    sfreq: float,
    scales: np.ndarray,
    *,
    clean_mask: np.ndarray | None = None,
    overlap: float = 0.5,
    min_clean_fraction_per_window: float = 0.95,
    min_windows_per_scale: int = 8,
    min_valid_scales: int = 8,
) -> DFAResult:
    """Estimate DFA-1 using the mean window standard deviation.

    ``series`` should be a continuous amplitude envelope (or a synthetic
    validation series), not the narrow-band carrier itself.
    """

    x = np.asarray(series, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("series must be one-dimensional")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    if clean_mask is None:
        clean = np.isfinite(x)
    else:
        clean = np.asarray(clean_mask, dtype=bool) & np.isfinite(x)
        if clean.shape != x.shape:
            raise ValueError("clean_mask must match series")
    if clean.sum() < 2:
        raise ValueError("fewer than two clean samples")

    filled = interpolate_masked_samples(x, clean)
    profile = np.cumsum(filled - np.mean(filled[clean]))

    used_scales: list[int] = []
    fluctuations: list[float] = []
    windows_per_scale: list[int] = []
    for scale in np.asarray(scales, dtype=int):
        if scale < 4 or scale > len(profile):
            continue
        step = max(1, int(round(scale * (1.0 - overlap))))
        starts = np.arange(0, len(profile) - scale + 1, step)
        clean_prefix = np.concatenate(
            ([0], np.cumsum(clean, dtype=np.int64))
        )
        clean_count = clean_prefix[scale:] - clean_prefix[:-scale]
        accepted = starts[
            clean_count[starts] >= min_clean_fraction_per_window * scale
        ]
        if len(accepted) < min_windows_per_scale:
            continue

        t = np.arange(scale, dtype=np.float64)
        t_centered = t - t.mean()
        denom = np.dot(t_centered, t_centered)
        windows = np.lib.stride_tricks.sliding_window_view(profile, scale)[accepted]
        window_mean = windows.mean(axis=1)
        slopes = (windows @ t_centered) / denom
        residual = (
            windows
            - window_mean[:, None]
            - slopes[:, None] * t_centered[None, :]
        )
        fluctuation = float(np.mean(np.sqrt(np.mean(residual**2, axis=1))))
        if np.isfinite(fluctuation) and fluctuation > 0:
            used_scales.append(int(scale))
            fluctuations.append(fluctuation)
            windows_per_scale.append(len(accepted))

    if len(used_scales) < min_valid_scales:
        return DFAResult(
            exponent=np.nan,
            fit_r2=np.nan,
            scales_seconds=np.asarray(used_scales, dtype=float) / sfreq,
            fluctuations=np.asarray(fluctuations),
            windows_per_scale=np.asarray(windows_per_scale, dtype=int),
            clean_sample_fraction=float(clean.mean()),
            n_valid_scales=len(used_scales),
        )

    log_scale = np.log(np.asarray(used_scales, dtype=float))
    log_fluctuation = np.log(np.asarray(fluctuations))
    slope, intercept = np.polyfit(log_scale, log_fluctuation, 1)
    fitted = intercept + slope * log_scale
    residual_ss = np.sum((log_fluctuation - fitted) ** 2)
    total_ss = np.sum((log_fluctuation - log_fluctuation.mean()) ** 2)
    fit_r2 = 1.0 - residual_ss / total_ss if total_ss > 0 else np.nan

    return DFAResult(
        exponent=float(slope),
        fit_r2=float(fit_r2),
        scales_seconds=np.asarray(used_scales, dtype=float) / sfreq,
        fluctuations=np.asarray(fluctuations),
        windows_per_scale=np.asarray(windows_per_scale, dtype=int),
        clean_sample_fraction=float(clean.mean()),
        n_valid_scales=len(used_scales),
    )


def estimate_alpha_envelope_dfa(
    data: np.ndarray,
    config: LRTCConfig | None = None,
    *,
    clean_mask: np.ndarray | None = None,
) -> AlphaEnvelopeDFAResult:
    """Estimate the Paper 02 primary alpha-envelope DFA target per channel."""

    config = config or LRTCConfig()
    config.validate()
    x, _ = _as_2d(data)
    if clean_mask is None:
        mask = np.isfinite(x)
    else:
        mask, _ = _as_mask_2d(clean_mask)
        if mask.shape != x.shape:
            raise ValueError("clean_mask must match data shape")
        mask &= np.isfinite(x)

    envelope = alpha_envelope(x, config, clean_mask=mask)
    return estimate_envelope_dfa(envelope, config, clean_mask=mask)


def estimate_envelope_dfa(
    envelope: np.ndarray,
    config: LRTCConfig | None = None,
    *,
    clean_mask: np.ndarray | None = None,
) -> AlphaEnvelopeDFAResult:
    """Estimate DFA from an already filtered Hilbert amplitude envelope."""

    config = config or LRTCConfig()
    config.validate()
    envelope, _ = _as_2d(envelope)
    if clean_mask is None:
        mask = np.isfinite(envelope)
    else:
        mask, _ = _as_mask_2d(clean_mask)
        if mask.shape != envelope.shape:
            raise ValueError("clean_mask must match envelope shape")
        mask &= np.isfinite(envelope)
    original_n_samples = envelope.shape[1]
    trim = int(round(config.edge_trim_seconds * config.sfreq))
    if trim:
        envelope = envelope[:, trim:-trim]
        mask = mask[:, trim:-trim]
    scales = admissible_scales(original_n_samples, config)

    results = [
        dfa(
            envelope[channel],
            config.sfreq,
            scales,
            clean_mask=mask[channel],
            overlap=config.overlap,
            min_clean_fraction_per_window=config.min_clean_fraction_per_window,
            min_windows_per_scale=config.min_windows_per_scale,
            min_valid_scales=config.min_valid_scales,
        )
        for channel in range(envelope.shape[0])
    ]
    return AlphaEnvelopeDFAResult(
        exponent=np.asarray([result.exponent for result in results]),
        fit_r2=np.asarray([result.fit_r2 for result in results]),
        clean_sample_fraction=np.asarray(
            [result.clean_sample_fraction for result in results]
        ),
        n_valid_scales=np.asarray(
            [result.n_valid_scales for result in results], dtype=int
        ),
        filter_numtaps=config.filter_numtaps,
        effective_duration_seconds=envelope.shape[1] / config.sfreq,
        config=asdict(config),
    )
