"""Shared output contract and image loading for Task 7-9 extractors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class FeatureResult:
    """Output every deterministic extractor in this package must return.

    `values` holds the raw feature vector. `valid` marks whether the image
    carried enough signal to trust `values`. `confidence` stays low when
    `valid` is False so a fusion head can mask the vector instead of
    treating a forced zero as evidence either way.
    """

    name: str
    values: dict[str, float]
    valid: bool
    confidence: float
    notes: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not self.name:
            raise ValueError("name is required")


def load_image_array(path: Path) -> np.ndarray:
    """Load an image as an RGB float32 array in [0, 1]."""

    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0


_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB array to grayscale using fixed ITU-R BT.601 luma weights."""

    return image[..., :3] @ _LUMA_WEIGHTS
