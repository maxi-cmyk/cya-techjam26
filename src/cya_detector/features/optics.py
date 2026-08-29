"""Confidence-masked radial chromatic-aberration estimation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from cya_detector.features.common import AuxiliaryFeatureResult, validate_result


def _edge_map(channel: np.ndarray) -> np.ndarray:
    horizontal = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
    vertical = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(horizontal, vertical)
    scale = float(np.quantile(magnitude, 0.9))
    return magnitude / max(scale, 1e-6)


def _radial_scale(image: np.ndarray, scale: float) -> np.ndarray:
    height, width = image.shape
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), 0.0, scale)
    return cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101
    )


def _fit_scale(reference: np.ndarray, target: np.ndarray, scales: np.ndarray, mask: np.ndarray) -> tuple[float, float, float, list[float]]:
    baseline = float(np.mean(np.abs(reference[mask] - target[mask])))
    losses = [float(np.mean(np.abs(reference[mask] - _radial_scale(target, float(scale))[mask]))) for scale in scales]
    best_index = int(np.argmin(losses))
    best_loss = losses[best_index]
    region_losses: list[float] = []
    height, width = reference.shape
    fitted = _radial_scale(target, float(scales[best_index]))
    for ys, xs in ((slice(0, height // 2), slice(0, width // 2)), (slice(0, height // 2), slice(width // 2, width)), (slice(height // 2, height), slice(0, width // 2)), (slice(height // 2, height), slice(width // 2, width))):
        region_mask = mask[ys, xs]
        region_losses.append(float(np.mean(np.abs(reference[ys, xs][region_mask] - fitted[ys, xs][region_mask]))) if np.any(region_mask) else best_loss)
    improvement = max(baseline - best_loss, 0.0) / max(baseline, 1e-8)
    return float(scales[best_index] - 1.0), best_loss, improvement, region_losses


def extract_optics_features(
    image_path: Path,
    *,
    scale_limit: float = 0.006,
    scale_steps: int = 25,
    min_dimension: int = 256,
    min_edge_fraction: float = 0.02,
    max_analysis_size: int = 1024,
) -> AuxiliaryFeatureResult:
    if scale_limit <= 0 or scale_steps < 3 or min_dimension < 32:
        raise ValueError("Invalid optics extractor configuration")
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((max_analysis_size, max_analysis_size), Image.Resampling.LANCZOS)
        rgb = np.asarray(image, dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]
    edges = [_edge_map(rgb[..., index]) for index in range(3)]
    combined = np.maximum.reduce(edges)
    threshold = float(np.quantile(combined, 0.8))
    mask = combined >= max(threshold, 1e-4)
    edge_fraction = float(np.mean(mask))
    supported = min(height, width) >= min_dimension and edge_fraction >= min_edge_fraction and int(np.sum(mask)) >= 256
    scales = np.linspace(1.0 - scale_limit, 1.0 + scale_limit, scale_steps)
    red = _fit_scale(edges[1], edges[0], scales, mask)
    blue = _fit_scale(edges[1], edges[2], scales, mask)
    consistency = float(np.exp(-10.0 * (np.std(red[3]) + np.std(blue[3]))))
    improvement = float((red[2] + blue[2]) / 2.0)
    confidence_value = float(np.clip(edge_fraction / 0.2, 0.0, 1.0) * np.clip(improvement * 10.0, 0.0, 1.0) * consistency) if supported else 0.0
    names = (
        "ca_red_radial_scale_delta", "ca_blue_radial_scale_delta",
        "ca_red_fit_residual", "ca_blue_fit_residual",
        "ca_red_improvement", "ca_blue_improvement",
        "ca_regional_consistency", "ca_edge_fraction", "ca_confidence",
    )
    values = np.asarray([red[0], blue[0], red[1], blue[1], red[2], blue[2], consistency, edge_fraction, confidence_value], dtype=np.float64)
    return validate_result(
        AuxiliaryFeatureResult(
            names=names,
            values=values,
            families=("ca",) * len(names),
            valid={"ca": supported and confidence_value > 0.0, "radial_distortion": False},
            confidence={"ca": confidence_value, "radial_distortion": 0.0},
            metadata={
                "analysis_width": width, "analysis_height": height,
                "optical_center_x": (width - 1) / 2.0, "optical_center_y": (height - 1) / 2.0,
                "radial_distortion_reason": "deferred_until_eligible_line_support_is_available",
            },
        )
    )
