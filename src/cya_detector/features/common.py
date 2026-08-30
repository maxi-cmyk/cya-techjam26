"""Shared deterministic auxiliary-feature result helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np


_LUMA_WEIGHTS = np.asarray([0.299, 0.587, 0.114], dtype=np.float64)


@dataclass(frozen=True)
class AuxiliaryFeatureResult:
    names: tuple[str, ...]
    values: np.ndarray
    families: tuple[str, ...]
    valid: dict[str, bool]
    confidence: dict[str, float]
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values.tolist(), strict=True))


def auxiliary_cache_key(
    *, image_sha256: str, extractor_version: str, configuration: dict[str, Any]
) -> str:
    if not image_sha256 or not extractor_version:
        raise ValueError("Image hash and extractor version are required")
    payload = {
        "configuration": configuration,
        "extractor_version": extractor_version,
        "image_sha256": image_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_result(result: AuxiliaryFeatureResult) -> AuxiliaryFeatureResult:
    if len(result.names) != len(result.values) or len(result.names) != len(result.families):
        raise ValueError("Auxiliary feature schema lengths do not match")
    if len(result.names) != len(set(result.names)):
        raise ValueError("Auxiliary feature names must be unique")
    if not np.all(np.isfinite(result.values)):
        raise ValueError("Auxiliary feature vector contains non-finite values")
    if any(not 0.0 <= value <= 1.0 for value in result.confidence.values()):
        raise ValueError("Auxiliary confidence must be in [0, 1]")
    return result


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB array to grayscale for deterministic texture selection."""

    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError("Expected an RGB image array")
    return np.asarray(image[..., :3], dtype=np.float64) @ _LUMA_WEIGHTS
