"""Task 7 - deterministic Stage 1 frequency feature extraction.

See docs/planning/tasks7to9_gameplan.md and the frequency-path sections of
docs/training/training.md and docs/architecture/techStack.md. This module
is a deterministic extractor only: FFT log-magnitude summaries,
radial/angular power, periodic-peak prominence, residual autocorrelation,
and local pixel-dependency (NPR-style) statistics. It never independently
returns an authenticity verdict, and the Stage 1 early exit stays disabled
until its validation gates pass elsewhere in the pipeline.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

from cya_detector.features.common import FeatureResult, to_grayscale

FEATURE_NAME = "frequency"

MIN_DIMENSION = 16
MIN_STD = 1e-4
RADIAL_BINS = 32
ANGULAR_BINS = 36
AUTOCORR_LAG_RADIUS = 8
EPS = 1e-8


def _radial_bin_index(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    cy, cx = height / 2.0, width / 2.0
    y, x = np.ogrid[:height, :width]
    radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    max_radius = radius.max()
    bin_edges = np.linspace(0.0, max_radius + EPS, RADIAL_BINS + 1)
    bin_index = np.clip(np.digitize(radius, bin_edges) - 1, 0, RADIAL_BINS - 1)
    return bin_index, bin_edges


def _radial_power_profile(power: np.ndarray) -> np.ndarray:
    height, width = power.shape
    bin_index, _ = _radial_bin_index(height, width)
    profile = np.zeros(RADIAL_BINS, dtype=np.float64)
    for bin_id in range(RADIAL_BINS):
        mask = bin_index == bin_id
        if mask.any():
            profile[bin_id] = power[mask].mean()
    return profile


def _radial_slope(profile: np.ndarray) -> float:
    valid = profile[2:-2]
    radii = np.arange(2, RADIAL_BINS - 2, dtype=np.float64)
    if len(valid) < 4 or valid.max() <= 0:
        return 0.0
    log_radius = np.log(radii + 1.0)
    log_power = np.log(valid + EPS)
    slope, _ = np.polyfit(log_radius, log_power, 1)
    return float(slope)


def _radial_peak_stats(profile: np.ndarray) -> tuple[float, int]:
    # Bins 0-1 sit on/adjacent to the DC component, which can be orders of
    # magnitude larger than any AC peak; including them would swamp the
    # detrended max used below and hide every real periodic peak.
    band = profile[2:]
    if band.size == 0 or band.max() <= 0:
        return 0.0, 0
    baseline = np.convolve(band, np.ones(5) / 5.0, mode="same")
    detrended = np.clip(band - baseline, a_min=0.0, a_max=None)
    peaks, properties = find_peaks(detrended, prominence=EPS)
    if len(peaks) == 0:
        return 0.0, 0
    prominences = properties["prominences"]
    threshold = 0.1 * detrended.max()
    significant = prominences[prominences >= threshold]
    return float(prominences.max()), int(len(significant))


def _angular_anisotropy(power: np.ndarray) -> float:
    height, width = power.shape
    cy, cx = height / 2.0, width / 2.0
    y, x = np.ogrid[:height, :width]
    radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    angle = np.arctan2(y - cy, x - cx)
    max_radius = radius.max()
    band = (radius >= 0.1 * max_radius) & (radius <= 0.9 * max_radius)
    if not band.any():
        return 0.0
    angle_bins = np.linspace(-np.pi, np.pi, ANGULAR_BINS + 1)
    bin_index = np.clip(np.digitize(angle, angle_bins) - 1, 0, ANGULAR_BINS - 1)
    profile = np.zeros(ANGULAR_BINS, dtype=np.float64)
    for bin_id in range(ANGULAR_BINS):
        mask = band & (bin_index == bin_id)
        if mask.any():
            profile[bin_id] = power[mask].mean()
    mean = profile.mean()
    if mean <= EPS:
        return 0.0
    return float(profile.std() / mean)


def _residual_autocorrelation(grayscale: np.ndarray) -> tuple[float, float]:
    smoothed = gaussian_filter(grayscale, sigma=1.0)
    residual = grayscale - smoothed

    spectrum = np.fft.fft2(residual)
    power = np.abs(spectrum) ** 2
    autocorr = np.fft.ifft2(power).real
    autocorr = np.fft.fftshift(autocorr)

    height, width = autocorr.shape
    cy, cx = height // 2, width // 2
    center_value = autocorr[cy, cx]
    if center_value <= EPS:
        return 0.0, 0.0
    autocorr /= center_value

    lag = AUTOCORR_LAG_RADIUS
    y0, y1 = max(0, cy - lag), min(height, cy + lag + 1)
    x0, x1 = max(0, cx - lag), min(width, cx + lag + 1)
    window = autocorr[y0:y1, x0:x1].copy()
    local_cy, local_cx = cy - y0, cx - x0
    window[local_cy, local_cx] = -np.inf

    flat = window[np.isfinite(window)]
    if flat.size == 0:
        return 0.0, 0.0
    top_k = np.sort(flat)[-5:]
    return float(flat.max()), float(top_k.mean())


def _local_pixel_dependency(grayscale: np.ndarray) -> tuple[float, float]:
    up = np.roll(grayscale, 1, axis=0)
    down = np.roll(grayscale, -1, axis=0)
    left = np.roll(grayscale, 1, axis=1)
    right = np.roll(grayscale, -1, axis=1)
    predicted = (up + down + left + right) / 4.0

    center = grayscale.ravel()
    neighbor = predicted.ravel()
    if center.std() <= EPS or neighbor.std() <= EPS:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(center, neighbor)[0, 1])

    residual_energy = float(np.mean((grayscale - predicted) ** 2))
    return correlation, residual_energy


def extract_frequency_features(image: np.ndarray) -> FeatureResult:
    """Extract deterministic frequency-domain features from an RGB image.

    `image` is an RGB float32 array in [0, 1], as returned by
    `cya_detector.features.common.load_image_array`.
    """

    height, width = image.shape[0], image.shape[1]
    grayscale = to_grayscale(np.asarray(image, dtype=np.float64))

    valid = height >= MIN_DIMENSION and width >= MIN_DIMENSION and grayscale.std() > MIN_STD
    if not valid:
        return FeatureResult(
            name=FEATURE_NAME,
            values={
                "log_magnitude_mean": 0.0,
                "log_magnitude_std": 0.0,
                "high_freq_energy_ratio": 0.0,
                "radial_slope": 0.0,
                "radial_peak_prominence_max": 0.0,
                "radial_peak_count": 0.0,
                "angular_anisotropy_std": 0.0,
                "residual_autocorr_peak_max": 0.0,
                "residual_autocorr_peak_mean": 0.0,
                "npr_correlation_mean": 0.0,
                "npr_residual_energy": 0.0,
            },
            valid=False,
            confidence=0.0,
            notes="Image too small or near-constant for frequency analysis",
        )

    spectrum = np.fft.fftshift(np.fft.fft2(grayscale))
    magnitude = np.abs(spectrum)
    power = magnitude**2
    log_magnitude = np.log1p(magnitude)

    radial_profile = _radial_power_profile(power)
    total_energy = radial_profile.sum()
    high_freq_energy = radial_profile[RADIAL_BINS // 2 :].sum()
    high_freq_ratio = float(high_freq_energy / total_energy) if total_energy > EPS else 0.0

    radial_slope = _radial_slope(radial_profile)
    peak_prominence, peak_count = _radial_peak_stats(radial_profile)
    angular_anisotropy = _angular_anisotropy(power)
    autocorr_peak_max, autocorr_peak_mean = _residual_autocorrelation(grayscale)
    npr_correlation, npr_residual_energy = _local_pixel_dependency(grayscale)

    confidence = float(np.clip(min(height, width) / 128.0, 0.0, 1.0))

    return FeatureResult(
        name=FEATURE_NAME,
        values={
            "log_magnitude_mean": float(log_magnitude.mean()),
            "log_magnitude_std": float(log_magnitude.std()),
            "high_freq_energy_ratio": high_freq_ratio,
            "radial_slope": radial_slope,
            "radial_peak_prominence_max": peak_prominence,
            "radial_peak_count": float(peak_count),
            "angular_anisotropy_std": angular_anisotropy,
            "residual_autocorr_peak_max": autocorr_peak_max,
            "residual_autocorr_peak_mean": autocorr_peak_mean,
            "npr_correlation_mean": npr_correlation,
            "npr_residual_energy": npr_residual_energy,
        },
        valid=True,
        confidence=confidence,
    )
