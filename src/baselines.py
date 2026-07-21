"""Classical EEG feature extraction and baseline models.

Extracts band power, Hjorth parameters, spectral entropy.
Trains LogReg + XGBoost baselines for comparison with foundation models.
"""

import logging

import numpy as np
from scipy import signal as sig
from scipy.integrate import trapezoid
from scipy.stats import entropy

logger = logging.getLogger(__name__)

# EEG frequency bands
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def compute_band_power(epoch: np.ndarray, sfreq: float) -> np.ndarray:
    """Compute relative band power for each channel.

    Args:
        epoch: [n_channels, n_times]
        sfreq: sampling frequency

    Returns:
        features: [n_channels * n_bands]
    """
    n_channels = epoch.shape[0]
    features = []

    for ch in range(n_channels):
        freqs, psd = sig.welch(epoch[ch], fs=sfreq, nperseg=min(256, epoch.shape[1]))
        total_power = trapezoid(psd, freqs)
        if total_power == 0:
            features.extend([0.0] * len(BANDS))
            continue

        for _, (fmin, fmax) in BANDS.items():
            band_mask = (freqs >= fmin) & (freqs <= fmax)
            band_power = trapezoid(psd[band_mask], freqs[band_mask])
            features.append(band_power / total_power)

    return np.array(features)


def compute_hjorth(epoch: np.ndarray) -> np.ndarray:
    """Compute Hjorth parameters (activity, mobility, complexity) per channel.

    Args:
        epoch: [n_channels, n_times]

    Returns:
        features: [n_channels * 3]
    """
    features = []
    for ch in range(epoch.shape[0]):
        x = epoch[ch]
        dx = np.diff(x)
        ddx = np.diff(dx)

        activity = np.var(x)
        mobility = np.sqrt(np.var(dx) / activity) if activity > 0 else 0
        complexity = (
            (np.sqrt(np.var(ddx) / np.var(dx)) / mobility)
            if mobility > 0 and np.var(dx) > 0
            else 0
        )

        features.extend([activity, mobility, complexity])

    return np.array(features)


def compute_spectral_entropy(epoch: np.ndarray, sfreq: float) -> np.ndarray:
    """Compute spectral entropy per channel.

    Args:
        epoch: [n_channels, n_times]
        sfreq: sampling frequency

    Returns:
        features: [n_channels]
    """
    features = []
    for ch in range(epoch.shape[0]):
        freqs, psd = sig.welch(epoch[ch], fs=sfreq, nperseg=min(256, epoch.shape[1]))
        psd_norm = psd / psd.sum() if psd.sum() > 0 else psd
        features.append(entropy(psd_norm + 1e-10))

    return np.array(features)


def _compute_alpha_asymmetry(epoch: np.ndarray, sfreq: float) -> float:
    """Compute hemispheric alpha power asymmetry (right - left, log-scaled).

    Uses even-indexed channels as right hemisphere proxies and odd-indexed
    channels as left hemisphere proxies (matching standard 10-20 convention).

    Args:
        epoch: [n_channels, n_times]
        sfreq: sampling frequency

    Returns:
        log(right_alpha + eps) - log(left_alpha + eps)
    """
    nperseg = min(256, epoch.shape[1])
    eps = 1e-10

    # Hemisphere assignment: even-indexed channels as right hemisphere proxies,
    # odd-indexed as left. This heuristic assumes standard 10-20 interleaved
    # ordering (e.g., Fp2, Fp1, F4, F3...) and is an approximation.
    # For montages with different ordering, pass channel_names and use
    # the 10-20 convention (even digit suffix = right, odd = left).
    right_powers = []
    left_powers = []
    for ch in range(epoch.shape[0]):
        freqs, psd = sig.welch(epoch[ch], fs=sfreq, nperseg=nperseg)
        alpha_mask = (freqs >= 8) & (freqs <= 13)
        alpha_power = trapezoid(psd[alpha_mask], freqs[alpha_mask])
        if ch % 2 == 0:
            right_powers.append(alpha_power)
        else:
            left_powers.append(alpha_power)

    right_mean = np.mean(right_powers) if right_powers else eps
    left_mean = np.mean(left_powers) if left_powers else eps

    return float(np.log(right_mean + eps) - np.log(left_mean + eps))


def _compute_frontal_theta_power(epoch: np.ndarray, sfreq: float) -> float:
    """Compute mean theta power (4-8 Hz) in frontal channels.

    Uses the first 4 channels as frontal proxies (Fp1, Fp2, F3, F4 in
    standard 10-20 ordering).

    Args:
        epoch: [n_channels, n_times]
        sfreq: sampling frequency

    Returns:
        Mean theta power across frontal channels.
    """
    nperseg = min(256, epoch.shape[1])
    n_frontal = min(4, epoch.shape[0])

    theta_powers = []
    for ch in range(n_frontal):
        freqs, psd = sig.welch(epoch[ch], fs=sfreq, nperseg=nperseg)
        theta_mask = (freqs >= 4) & (freqs <= 8)
        theta_power = trapezoid(psd[theta_mask], freqs[theta_mask])
        theta_powers.append(theta_power)

    return float(np.mean(theta_powers))


def extract_classical_features(epochs: np.ndarray, sfreq: float = 200.0) -> np.ndarray:
    """Extract all classical features for a set of epochs.

    Features per epoch:
        - Band power: 5 * n_ch
        - Hjorth: 3 * n_ch
        - Spectral entropy: n_ch
        - Alpha asymmetry: 1
        - Frontal theta power: 1
    Total: 9 * n_ch + 2

    Args:
        epochs: [n_epochs, n_channels, n_times]
        sfreq: sampling frequency

    Returns:
        features: [n_epochs, n_features]
    """
    all_features = []
    for epoch in epochs:
        bp = compute_band_power(epoch, sfreq)
        hj = compute_hjorth(epoch)
        se = compute_spectral_entropy(epoch, sfreq)
        alpha_asym = _compute_alpha_asymmetry(epoch, sfreq)
        frontal_theta = _compute_frontal_theta_power(epoch, sfreq)
        all_features.append(np.concatenate([bp, hj, se, [alpha_asym, frontal_theta]]))

    return np.array(all_features)


def extract_subject_features(processed_data: dict, sfreq: float = 200.0) -> dict:
    """Extract classical features for all subjects, mean-pooled per subject.

    Returns:
        Dict mapping subject_id -> {"features": np.ndarray, "label": int}
    """
    results = {}
    total = len(processed_data)

    for idx, (sid, data) in enumerate(processed_data.items()):
        if idx % 20 == 0:
            logger.info(f"Extracting classical features: {idx}/{total}")

        epochs = data["epochs"]
        features = extract_classical_features(epochs, sfreq)
        subject_features = features.mean(axis=0)  # Mean pool across epochs

        results[sid] = {
            "features": subject_features,
            "label": data["label"],
        }

    logger.info(
        f"Extracted {results[list(results.keys())[0]]['features'].shape[0]} classical features per subject"
    )
    return results
