"""Architecture-agnostic frequency and residual fingerprint extraction."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from scipy import fft as scipy_fft
from scipy import ndimage


@dataclass(frozen=True)
class FrequencyFeatureResult:
    names: tuple[str, ...]
    values: np.ndarray
    families: tuple[str, ...]
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values.tolist(), strict=True))


def frequency_cache_key(
    *, image_sha256: str, extractor_version: str, configuration: dict[str, Any]
) -> str:
    if not image_sha256 or not extractor_version:
        raise ValueError("Image hash and extractor version are required")
    payload = {
        "configuration": configuration,
        "extractor_version": extractor_version,
        "image_sha256": image_sha256,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _center_crop_limit(luminance: np.ndarray, max_size: int) -> np.ndarray:
    height, width = luminance.shape
    crop_height = min(height, max_size)
    crop_width = min(width, max_size)
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return luminance[top : top + crop_height, left : left + crop_width]


def _binned_mean(
    values: np.ndarray, coordinates: np.ndarray, *, bins: int, lower: float, upper: float
) -> np.ndarray:
    edges = np.linspace(lower, upper, bins + 1)
    indices = np.clip(np.digitize(coordinates.ravel(), edges) - 1, 0, bins - 1)
    flattened = values.ravel()
    sums = np.bincount(indices, weights=flattened, minlength=bins)
    counts = np.bincount(indices, minlength=bins)
    return sums / np.maximum(counts, 1)


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.ravel().astype(np.float64)
    right_flat = right.ravel().astype(np.float64)
    if left_flat.size < 2 or np.std(left_flat) < 1e-12 or np.std(right_flat) < 1e-12:
        return 0.0
    return float(np.corrcoef(left_flat, right_flat)[0, 1])


def _offset_correlation(image: np.ndarray, dy: int, dx: int) -> float:
    height, width = image.shape
    if abs(dy) >= height or abs(dx) >= width:
        return 0.0
    y_left = slice(max(0, dy), min(height, height + dy))
    y_right = slice(max(0, -dy), min(height, height - dy))
    x_left = slice(max(0, dx), min(width, width + dx))
    x_right = slice(max(0, -dx), min(width, width - dx))
    return _safe_correlation(image[y_left, x_left], image[y_right, x_right])


def _difference_statistics(values: np.ndarray) -> tuple[float, float, float]:
    flattened = values.ravel().astype(np.float64)
    standard_deviation = float(np.std(flattened))
    if standard_deviation < 1e-12:
        return 0.0, 0.0, 0.0
    centered = flattened - np.mean(flattened)
    kurtosis = float(np.mean((centered / standard_deviation) ** 4) - 3.0)
    return float(np.mean(np.abs(flattened))), standard_deviation, kurtosis


def _append_profile(
    names: list[str],
    values: list[float],
    families: list[str],
    *,
    prefix: str,
    profile: np.ndarray,
    family: str,
) -> None:
    centered = profile - np.mean(profile)
    scale = np.std(centered)
    normalized = centered / scale if scale > 1e-12 else centered
    for index, value in enumerate(normalized):
        names.append(f"{prefix}_{index:02d}")
        values.append(float(value))
        families.append(family)


def extract_frequency_features(
    image_path: Path,
    *,
    radial_bins: int = 24,
    angular_bins: int = 12,
    dct_bins: int = 16,
    phase_bins: int = 12,
    max_analysis_size: int = 1024,
) -> FrequencyFeatureResult:
    """Extract fixed-length magnitude, phase, residual, and dependency summaries."""

    if min(radial_bins, angular_bins, dct_bins, phase_bins) <= 1:
        raise ValueError("Frequency bin counts must exceed one")
    if max_analysis_size < 32:
        raise ValueError("max_analysis_size must be at least 32")
    with Image.open(image_path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        original_width, original_height = rgb.size
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    luminance = (
        0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2]
    )
    luminance = _center_crop_limit(luminance, max_analysis_size)
    height, width = luminance.shape
    if min(height, width) < 16:
        raise ValueError(f"Image is too small for frequency extraction: {width}x{height}")

    window = np.outer(np.hanning(height), np.hanning(width)).astype(np.float32)
    centered = luminance - float(np.mean(luminance))
    spectrum = np.fft.rfft2(centered * window)
    magnitude = np.abs(spectrum)
    power = magnitude**2
    log_magnitude = np.log1p(magnitude)
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.rfftfreq(width)[None, :]
    radius = np.sqrt(fy**2 + fx**2) / math.sqrt(0.5)
    angle = np.arctan2(fy, np.broadcast_to(fx, radius.shape))

    names: list[str] = []
    values: list[float] = []
    families: list[str] = []
    radial_profile = _binned_mean(
        log_magnitude, radius, bins=radial_bins, lower=0.0, upper=1.0
    )
    angular_profile = _binned_mean(
        log_magnitude, angle, bins=angular_bins, lower=-math.pi / 2, upper=math.pi / 2
    )
    _append_profile(
        names, values, families, prefix="fft_radial", profile=radial_profile, family="magnitude"
    )
    _append_profile(
        names,
        values,
        families,
        prefix="fft_angular",
        profile=angular_profile,
        family="magnitude",
    )

    total_power = float(np.sum(power)) + 1e-12
    for feature_name, mask in (
        ("fft_power_low", radius < 0.15),
        ("fft_power_mid", (radius >= 0.15) & (radius < 0.45)),
        ("fft_power_high", radius >= 0.45),
    ):
        names.append(feature_name)
        values.append(float(np.sum(power[mask]) / total_power))
        families.append("magnitude")
    slope = float(np.polyfit(np.linspace(0.0, 1.0, radial_bins)[1:], radial_profile[1:], 1)[0])
    interior = log_magnitude[radius > 0.05]
    peak_prominence = (
        float((np.max(interior) - np.median(interior)) / (np.std(interior) + 1e-12))
        if interior.size
        else 0.0
    )
    names.extend(("fft_radial_slope", "fft_periodic_peak_prominence"))
    values.extend((slope, peak_prominence))
    families.extend(("magnitude", "magnitude"))

    dct = np.abs(scipy_fft.dctn(centered * window, type=2, norm="ortho"))
    log_dct = np.log1p(dct)
    dct_y = np.arange(height)[:, None] / max(height - 1, 1)
    dct_x = np.arange(width)[None, :] / max(width - 1, 1)
    dct_radius = np.sqrt(dct_y**2 + dct_x**2) / math.sqrt(2.0)
    dct_profile = _binned_mean(
        log_dct, dct_radius, bins=dct_bins, lower=0.0, upper=1.0
    )
    _append_profile(
        names, values, families, prefix="dct_radial", profile=dct_profile, family="magnitude"
    )
    dct_energy = dct**2
    names.append("dct_high_energy_ratio")
    values.append(float(np.sum(dct_energy[dct_radius >= 0.5]) / (np.sum(dct_energy) + 1e-12)))
    families.append("magnitude")

    residual = centered - ndimage.gaussian_filter(centered, sigma=1.0, mode="reflect")
    for lag in (1, 2, 4):
        for direction, dy, dx in (
            ("h", 0, lag),
            ("v", lag, 0),
            ("d1", lag, lag),
            ("d2", lag, -lag),
        ):
            names.append(f"residual_corr_{direction}_{lag}")
            values.append(_offset_correlation(residual, dy, dx))
            families.append("residual")
    for direction, differences in (
        ("h", np.diff(luminance, axis=1)),
        ("v", np.diff(luminance, axis=0)),
    ):
        mean_absolute, standard_deviation, kurtosis = _difference_statistics(differences)
        for statistic, value in (
            ("mean_abs", mean_absolute),
            ("std", standard_deviation),
            ("kurtosis", kurtosis),
        ):
            names.append(f"neighbor_diff_{direction}_{statistic}")
            values.append(value)
            families.append("residual")
    names.extend(("pixel_corr_h", "pixel_corr_v", "residual_std"))
    values.extend(
        (
            _offset_correlation(luminance, 0, 1),
            _offset_correlation(luminance, 1, 0),
            float(np.std(residual)),
        )
    )
    families.extend(("residual", "residual", "residual"))

    phase = np.angle(spectrum)
    unit_phase = np.exp(1j * phase)
    phase_coherence = _binned_mean(
        np.real(unit_phase), radius, bins=phase_bins, lower=0.0, upper=1.0
    ) + 1j * _binned_mean(
        np.imag(unit_phase), radius, bins=phase_bins, lower=0.0, upper=1.0
    )
    for index, value in enumerate(np.clip(np.abs(phase_coherence), 0.0, 1.0)):
        names.append(f"phase_radial_coherence_{index:02d}")
        values.append(float(value))
        families.append("phase")
    histogram, _ = np.histogram(phase, bins=32, range=(-math.pi, math.pi), density=False)
    probabilities = histogram / max(np.sum(histogram), 1)
    nonzero = probabilities[probabilities > 0]
    phase_entropy = float(-np.sum(nonzero * np.log(nonzero)) / math.log(32))
    names.append("phase_entropy")
    values.append(phase_entropy)
    families.append("phase")

    vector = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"Non-finite frequency feature generated for {image_path}")
    return FrequencyFeatureResult(
        names=tuple(names),
        values=vector,
        families=tuple(families),
        metadata={
            "original_width": original_width,
            "original_height": original_height,
            "analysis_width": width,
            "analysis_height": height,
            "center_crop_only": True,
            "resize_applied": False,
        },
    )
