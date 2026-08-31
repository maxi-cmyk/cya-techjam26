"""Reference-free single-image summaries from the validated PRNU-v2 residual.

These features never compare an input with an enrolled camera fingerprint and
must not be interpreted as camera identification or authentication.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from cya_detector.features.common import AuxiliaryFeatureResult, validate_result
from cya_detector.features.prnu_reference_v2 import _wavelet_components


PRNU_RUNTIME_V2_FEATURES = (
    "prnu_v2_raw_residual_std",
    "prnu_v2_raw_residual_mean_abs",
    "prnu_v2_raw_residual_kurtosis",
    "prnu_v2_spectral_flatness",
    "prnu_v2_spectral_low_fraction",
    "prnu_v2_spectral_high_fraction",
    "prnu_v2_row_periodicity",
    "prnu_v2_column_periodicity",
    "prnu_v2_cfa_corr_h2",
    "prnu_v2_cfa_corr_v2",
    "prnu_v2_cfa_corr_d1",
    "prnu_v2_luminance_coupling",
    "prnu_v2_block_energy_mean",
    "prnu_v2_block_energy_std",
    "prnu_v2_block_consistency",
    "prnu_v2_mask_fraction",
)


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left_values = np.asarray(left, dtype=np.float64).ravel()
    right_values = np.asarray(right, dtype=np.float64).ravel()
    if (
        left_values.size < 2
        or left_values.size != right_values.size
        or float(np.std(left_values)) < 1e-12
        or float(np.std(right_values)) < 1e-12
    ):
        return 0.0
    return float(np.corrcoef(left_values, right_values)[0, 1])


def _offset_corr(array: np.ndarray, dy: int, dx: int) -> float:
    height, width = array.shape
    return _safe_corr(
        array[max(0, dy) : min(height, height + dy), max(0, dx) : min(width, width + dx)],
        array[max(0, -dy) : min(height, height - dy), max(0, -dx) : min(width, width - dx)],
    )


def _spectral_summary(residual: np.ndarray) -> tuple[float, float, float]:
    power = np.abs(np.fft.fftshift(np.fft.fft2(residual))) ** 2
    height, width = power.shape
    yy, xx = np.ogrid[:height, :width]
    radius = np.sqrt((yy - height // 2) ** 2 + (xx - width // 2) ** 2)
    radius /= max(float(radius.max()), 1.0)
    positive = power[power > 1e-20]
    flatness = (
        float(np.exp(np.mean(np.log(positive))) / np.mean(positive))
        if positive.size
        else 0.0
    )
    total = float(np.sum(power))
    if total <= 1e-20:
        return flatness, 0.0, 0.0
    return (
        flatness,
        float(np.sum(power[radius <= 0.20]) / total),
        float(np.sum(power[radius >= 0.65]) / total),
    )


def _block_summary(residual: np.ndarray, block_size: int) -> tuple[float, float, float]:
    normalized: list[np.ndarray] = []
    energies: list[float] = []
    for top in range(0, residual.shape[0] - block_size + 1, block_size):
        for left in range(0, residual.shape[1] - block_size + 1, block_size):
            block = residual[top : top + block_size, left : left + block_size]
            scale = float(np.std(block))
            energies.append(scale)
            if scale > 1e-12:
                normalized.append((block - float(np.mean(block))) / scale)
    consistency = float(
        np.median(
            [abs(_safe_corr(left, right)) for left, right in zip(normalized[:-1], normalized[1:])]
            or [0.0]
        )
    )
    return float(np.mean(energies or [0.0])), float(np.std(energies or [0.0])), consistency


def _unsupported_result(width: int, height: int, crop_size: int) -> AuxiliaryFeatureResult:
    return validate_result(
        AuxiliaryFeatureResult(
            names=PRNU_RUNTIME_V2_FEATURES,
            values=np.zeros(len(PRNU_RUNTIME_V2_FEATURES), dtype=np.float64),
            families=("prnu_v2_runtime",) * len(PRNU_RUNTIME_V2_FEATURES),
            valid={"prnu_v2_runtime": False},
            confidence={"prnu_v2_runtime": 0.0},
            metadata={
                "analysis_width": width,
                "analysis_height": height,
                "crop_size": crop_size,
                "eligible": False,
            },
        )
    )


def extract_prnu_runtime_v2(
    image_path: Path,
    *,
    crop_size: int = 512,
    wavelet: str = "db2",
    wavelet_levels: int = 4,
    edge_keep_quantile: float = 0.75,
    block_size: int = 64,
) -> AuxiliaryFeatureResult:
    """Extract a fixed, reference-free PRNU-v2 vector from one received image."""

    if crop_size < 128 or block_size < 16 or crop_size % block_size:
        raise ValueError("PRNU-v2 runtime crop must be >=128 and divisible by block size")
    if wavelet_levels < 1 or not 0.0 < edge_keep_quantile <= 1.0:
        raise ValueError("Invalid PRNU-v2 runtime wavelet or masking configuration")
    with Image.open(image_path) as image:
        width, height = image.size
    if min(width, height) < crop_size:
        return _unsupported_result(width, height, crop_size)

    luminance, raw_residual, residual, mask = _wavelet_components(
        image_path,
        crop_size=crop_size,
        wavelet=wavelet,
        wavelet_levels=wavelet_levels,
        edge_keep_quantile=edge_keep_quantile,
    )
    masked_raw = raw_residual[mask]
    standard_deviation = float(np.std(masked_raw)) if masked_raw.size else 0.0
    centered = masked_raw - float(np.mean(masked_raw)) if masked_raw.size else masked_raw
    kurtosis = (
        float(np.mean(centered**4) / max(float(np.mean(centered**2)) ** 2, 1e-20) - 3.0)
        if centered.size
        else 0.0
    )
    spectral_flatness, spectral_low, spectral_high = _spectral_summary(residual)
    block_mean, block_std, block_consistency = _block_summary(residual, block_size)
    values = np.asarray(
        [
            standard_deviation,
            float(np.mean(np.abs(masked_raw))) if masked_raw.size else 0.0,
            kurtosis,
            spectral_flatness,
            spectral_low,
            spectral_high,
            abs(_offset_corr(residual, 1, 0)),
            abs(_offset_corr(residual, 0, 1)),
            _offset_corr(residual, 0, 2),
            _offset_corr(residual, 2, 0),
            _offset_corr(residual, 1, 1),
            _safe_corr(luminance[mask], np.abs(raw_residual[mask])),
            block_mean,
            block_std,
            block_consistency,
            float(np.mean(mask)),
        ],
        dtype=np.float64,
    )
    mask_fraction = float(np.mean(mask))
    valid = bool(masked_raw.size and standard_deviation > 1e-12 and np.all(np.isfinite(values)))
    return validate_result(
        AuxiliaryFeatureResult(
            names=PRNU_RUNTIME_V2_FEATURES,
            values=np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0),
            families=("prnu_v2_runtime",) * len(PRNU_RUNTIME_V2_FEATURES),
            valid={"prnu_v2_runtime": valid},
            confidence={"prnu_v2_runtime": mask_fraction if valid else 0.0},
            metadata={
                "analysis_width": width,
                "analysis_height": height,
                "crop_size": crop_size,
                "eligible": True,
                "reference_comparison_used": False,
            },
        )
    )
