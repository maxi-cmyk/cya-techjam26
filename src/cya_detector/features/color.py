"""Task 8 - deterministic inter-channel color correlation features.

See docs/planning/tasks7to9_gameplan.md and the CHROMA-inspired sections of
docs/architecture/techStack.md. Extracts standardized global and local-window
RGB and Lab inter-channel correlations (R/G, R/B, G/B, L/a, L/b, a/b) with
low-variance masking and numerical guards. Deterministic only; never an
authenticity verdict on its own.
"""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2lab

from cya_detector.features.common import FeatureResult

FEATURE_NAME = "color"

MIN_DIMENSION = 16
MIN_CHANNEL_STD = 1e-4
LOCAL_BLOCK_SIZE = 16
EPS = 1e-8

CHANNEL_PAIRS = (
    ("rgb", "r", "g"),
    ("rgb", "r", "b"),
    ("rgb", "g", "b"),
    ("lab", "l", "a"),
    ("lab", "l", "b"),
    ("lab", "a", "b"),
)


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> tuple[float, bool]:
    """Pearson correlation with a low-variance guard; returns (value, valid)."""

    if first.std() < MIN_CHANNEL_STD or second.std() < MIN_CHANNEL_STD:
        return 0.0, False
    correlation = np.corrcoef(first.ravel(), second.ravel())[0, 1]
    if not np.isfinite(correlation):
        return 0.0, False
    return float(np.clip(correlation, -1.0, 1.0)), True


def _local_window_correlation(first: np.ndarray, second: np.ndarray) -> tuple[float, bool]:
    """Mean per-block correlation over non-overlapping windows with enough variance."""

    height, width = first.shape
    block = min(LOCAL_BLOCK_SIZE, height, width)
    if block < 4:
        return 0.0, False

    values: list[float] = []
    for top in range(0, height - block + 1, block):
        for left in range(0, width - block + 1, block):
            block_first = first[top : top + block, left : left + block]
            block_second = second[top : top + block, left : left + block]
            value, valid = _safe_correlation(block_first, block_second)
            if valid:
                values.append(value)

    if not values:
        return 0.0, False
    return float(np.mean(values)), True


def extract_color_features(image: np.ndarray) -> FeatureResult:
    """Extract deterministic RGB/Lab inter-channel correlation features.

    `image` is an RGB float32 array in [0, 1], as returned by
    `cya_detector.features.common.load_image_array`.
    """

    height, width = image.shape[0], image.shape[1]
    if height < MIN_DIMENSION or width < MIN_DIMENSION:
        return FeatureResult(
            name=FEATURE_NAME,
            values={},
            valid=False,
            confidence=0.0,
            notes="Image too small for color correlation analysis",
        )

    rgb = np.asarray(image, dtype=np.float64)
    channels = {"r": rgb[..., 0], "g": rgb[..., 1], "b": rgb[..., 2]}

    lab = rgb2lab(np.clip(rgb, 0.0, 1.0))
    channels.update({"l": lab[..., 0], "a": lab[..., 1], "b_lab": lab[..., 2]})
    lab_channel_names = {"l": "l", "a": "a", "b": "b_lab"}

    values: dict[str, float] = {}
    valid_pair_count = 0
    for space, first_name, second_name in CHANNEL_PAIRS:
        first_key = lab_channel_names[first_name] if space == "lab" else first_name
        second_key = lab_channel_names[second_name] if space == "lab" else second_name
        first, second = channels[first_key], channels[second_key]

        global_value, global_valid = _safe_correlation(first, second)
        local_value, local_valid = _local_window_correlation(first, second)

        pair_label = f"{space}_{first_name}{second_name}"
        values[f"{pair_label}_global_corr"] = global_value
        values[f"{pair_label}_local_corr"] = local_value
        values[f"{pair_label}_valid"] = float(global_valid)
        if global_valid:
            valid_pair_count += 1

    valid = valid_pair_count > 0
    confidence = float(valid_pair_count / len(CHANNEL_PAIRS)) if valid else 0.0

    return FeatureResult(
        name=FEATURE_NAME,
        values=values,
        valid=valid,
        confidence=confidence,
        notes="" if valid else "All channel pairs had insufficient variance",
    )
