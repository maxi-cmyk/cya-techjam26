"""Single-image PRNU-coherence proxy; this is not camera identification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

from cya_detector.features.common import AuxiliaryFeatureResult, validate_result


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = left.ravel().astype(np.float64)
    right = right.ravel().astype(np.float64)
    if left.size < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _offset_corr(array: np.ndarray, dy: int, dx: int) -> float:
    height, width = array.shape
    return _safe_corr(
        array[max(0, dy) : min(height, height + dy), max(0, dx) : min(width, width + dx)],
        array[max(0, -dy) : min(height, height - dy), max(0, -dx) : min(width, width - dx)],
    )


def extract_prnu_features(
    image_path: Path,
    *,
    block_size: int = 64,
    denoise_sigma: float = 1.0,
    min_dimension: int = 128,
    max_analysis_size: int = 1024,
) -> AuxiliaryFeatureResult:
    if block_size < 16 or denoise_sigma <= 0 or min_dimension < 32:
        raise ValueError("Invalid PRNU extractor configuration")
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((max_analysis_size, max_analysis_size), Image.Resampling.LANCZOS)
        rgb = np.asarray(image, dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]
    supported = min(height, width) >= min_dimension
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    denoised = ndimage.gaussian_filter(luminance, sigma=denoise_sigma, mode="reflect")
    residual = luminance - denoised
    block_energy: list[float] = []
    block_brightness: list[float] = []
    normalized_blocks: list[np.ndarray] = []
    for top in range(0, height - block_size + 1, block_size):
        for left in range(0, width - block_size + 1, block_size):
            patch = residual[top : top + block_size, left : left + block_size]
            brightness = luminance[top : top + block_size, left : left + block_size]
            block_energy.append(float(np.std(patch)))
            block_brightness.append(float(np.mean(brightness)))
            scale = float(np.std(patch))
            if scale > 1e-8:
                normalized_blocks.append((patch - np.mean(patch)) / scale)
    adjacent_correlations = [
        _safe_corr(left, right)
        for left, right in zip(normalized_blocks[:-1], normalized_blocks[1:])
    ]
    energy = np.asarray(block_energy or [0.0])
    brightness = np.asarray(block_brightness or [0.0])
    self_consistency = float(np.median(np.abs(adjacent_correlations or [0.0])))
    irradiance_coupling = _safe_corr(brightness, energy)
    cfa_values = [_offset_corr(residual, 0, 2), _offset_corr(residual, 2, 0), _offset_corr(residual, 1, 1)]
    coherence = float(np.clip((self_consistency + np.mean(np.abs(cfa_values))) / 2.0, 0.0, 1.0))
    names = (
        "prnu_residual_std",
        "prnu_block_energy_mean",
        "prnu_block_energy_std",
        "prnu_irradiance_coupling",
        "prnu_block_self_consistency",
        "prnu_cfa_corr_h2",
        "prnu_cfa_corr_v2",
        "prnu_cfa_corr_d1",
        "prnu_coherence",
        "prnu_block_coverage",
    )
    expected_blocks = max((height // block_size) * (width // block_size), 1)
    coverage = min(len(block_energy) / expected_blocks, 1.0)
    values = np.asarray(
        [
            np.std(residual), np.mean(energy), np.std(energy), irradiance_coupling,
            self_consistency, *cfa_values, coherence, coverage,
        ],
        dtype=np.float64,
    )
    confidence_value = float(np.clip(coverage * min(len(block_energy) / 4.0, 1.0), 0.0, 1.0)) if supported else 0.0
    return validate_result(
        AuxiliaryFeatureResult(
            names=names,
            values=values,
            families=("prnu",) * len(names),
            valid={"prnu": supported and len(block_energy) >= 4},
            confidence={"prnu": confidence_value},
            metadata={"analysis_width": width, "analysis_height": height, "block_count": len(block_energy)},
        )
    )
