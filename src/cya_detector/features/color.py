"""RGB/Lab inter-channel correlation features with numerical guards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2lab

from cya_detector.features.common import AuxiliaryFeatureResult, validate_result


def _correlation(left: np.ndarray, right: np.ndarray, epsilon: float) -> tuple[float, bool]:
    left = left.astype(np.float64, copy=False).ravel()
    right = right.astype(np.float64, copy=False).ravel()
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left.size < 2 or left_std < epsilon or right_std < epsilon:
        return 0.0, False
    standardized_left = (left - np.mean(left)) / left_std
    standardized_right = (right - np.mean(right)) / right_std
    return float(np.mean(standardized_left * standardized_right)), True


def _space_features(
    array: np.ndarray,
    *,
    prefix: str,
    channel_names: tuple[str, str, str],
    window_size: int,
    epsilon: float,
) -> tuple[list[str], list[float], bool, float]:
    names: list[str] = []
    values: list[float] = []
    pair_validity: list[bool] = []
    height, width, _ = array.shape
    pairs = ((0, 1), (0, 2), (1, 2))
    for left_index, right_index in pairs:
        pair = f"{channel_names[left_index]}_{channel_names[right_index]}"
        global_corr, global_valid = _correlation(
            array[..., left_index], array[..., right_index], epsilon
        )
        local: list[float] = []
        total_windows = 0
        for top in range(0, height, window_size):
            for left in range(0, width, window_size):
                patch = array[top : top + window_size, left : left + window_size]
                if min(patch.shape[:2]) < max(8, window_size // 2):
                    continue
                total_windows += 1
                correlation, valid = _correlation(
                    patch[..., left_index], patch[..., right_index], epsilon
                )
                if valid:
                    local.append(correlation)
        coverage = len(local) / max(total_windows, 1)
        local_array = np.asarray(local or [0.0], dtype=np.float64)
        statistics = {
            "global": global_corr,
            "local_mean": float(np.mean(local_array)),
            "local_std": float(np.std(local_array)),
            "local_median": float(np.median(local_array)),
            "local_q10": float(np.quantile(local_array, 0.1)),
            "local_q90": float(np.quantile(local_array, 0.9)),
            "local_coverage": coverage,
            "global_valid": float(global_valid),
        }
        for statistic, value in statistics.items():
            names.append(f"{prefix}_{pair}_{statistic}")
            values.append(value)
        pair_validity.append(global_valid and coverage > 0.0)
    confidence = float(np.mean([values[index] for index, name in enumerate(names) if name.endswith("local_coverage")]))
    return names, values, all(pair_validity), confidence


def extract_color_features(
    image_path: Path,
    *,
    window_size: int = 64,
    low_variance_epsilon: float = 1e-4,
    max_analysis_size: int = 1024,
) -> AuxiliaryFeatureResult:
    if window_size < 16 or max_analysis_size < 32 or low_variance_epsilon <= 0:
        raise ValueError("Invalid color extractor configuration")
    with Image.open(image_path) as image:
        rgb_image = ImageOps.exif_transpose(image).convert("RGB")
        rgb_image.thumbnail((max_analysis_size, max_analysis_size), Image.Resampling.LANCZOS)
        rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0
    if min(rgb.shape[:2]) < 16:
        raise ValueError("Image is too small for color-correlation extraction")
    lab = rgb2lab(rgb).astype(np.float32)
    names: list[str] = []
    values: list[float] = []
    families: list[str] = []
    valid: dict[str, bool] = {}
    confidence: dict[str, float] = {}
    for array, prefix, channels, family in (
        (rgb, "rgb", ("r", "g", "b"), "rgb"),
        (lab, "lab", ("l", "a", "b"), "lab"),
    ):
        space_names, space_values, space_valid, space_confidence = _space_features(
            array,
            prefix=prefix,
            channel_names=channels,
            window_size=window_size,
            epsilon=low_variance_epsilon,
        )
        names.extend(space_names)
        values.extend(space_values)
        families.extend([family] * len(space_names))
        valid[family] = space_valid
        confidence[family] = space_confidence
    return validate_result(
        AuxiliaryFeatureResult(
            names=tuple(names),
            values=np.asarray(values, dtype=np.float64),
            families=tuple(families),
            valid=valid,
            confidence=confidence,
            metadata={"analysis_width": rgb.shape[1], "analysis_height": rgb.shape[0]},
        )
    )
