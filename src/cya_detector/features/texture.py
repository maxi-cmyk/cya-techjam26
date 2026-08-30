"""Task 9 - texture-aware local-detail patch selection under a fixed budget.

See docs/planning/tasks7to9_gameplan.md. Selects a fixed top-k set of
non-overlapping, highest-detail patches from a multi-scale Laplacian/Sobel
energy map, while retaining the global view so patch selection cannot
discard semantic context. Patch selection is deterministic; texture energy
chooses where to look and is never an authenticity threshold on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from cya_detector.features.common import to_grayscale

LAPLACIAN_SCALES = (0.0, 2.0)


@dataclass(frozen=True)
class PatchSelection:
    """Fixed top-k non-overlapping detail patches plus their energy scores."""

    patch_boxes: list[tuple[int, int, int, int]]
    """(top, left, height, width) in pixels, in descending energy order."""

    energy_scores: list[float]
    image_shape: tuple[int, int]

    def __post_init__(self) -> None:
        if len(self.patch_boxes) != len(self.energy_scores):
            raise ValueError("patch_boxes and energy_scores must be the same length")


@dataclass(frozen=True)
class PreparedPatchViews:
    """Materialized, independent local patches and their fixed availability mask."""

    patches: tuple[np.ndarray, ...]
    patch_boxes: tuple[tuple[int, int, int, int], ...]
    availability_mask: tuple[bool, ...]
    original_shape: tuple[int, int]
    padded_shape: tuple[int, int]


def _normalized(array: np.ndarray) -> np.ndarray:
    std = array.std()
    return array / std if std > 1e-8 else np.zeros_like(array)


def _multiscale_energy_map(grayscale: np.ndarray) -> np.ndarray:
    """Combine multi-scale Laplacian and Sobel gradient energy into one map."""

    energy = np.zeros_like(grayscale, dtype=np.float64)
    for sigma in LAPLACIAN_SCALES:
        smoothed = gaussian_filter(grayscale, sigma=sigma) if sigma > 0 else grayscale
        laplacian = cv2.Laplacian(smoothed, cv2.CV_64F, ksize=3)
        energy += _normalized(np.abs(laplacian))

    sobel_x = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.hypot(sobel_x, sobel_y)
    energy += _normalized(sobel_magnitude)

    return energy


def select_texture_patches(image: np.ndarray, *, patch_size: int, top_k: int) -> PatchSelection:
    """Select up to `top_k` non-overlapping, highest-detail patches.

    `image` is an RGB float32 array in [0, 1], as returned by
    `cya_detector.features.common.load_image_array`. Detail is measured with
    a multi-scale Laplacian/Sobel energy map; selection is deterministic
    given the same image and parameters.
    """

    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    height, width = image.shape[0], image.shape[1]
    rows, cols = height // patch_size, width // patch_size
    if rows == 0 or cols == 0:
        return PatchSelection(patch_boxes=[], energy_scores=[], image_shape=(height, width))

    grayscale = to_grayscale(np.asarray(image, dtype=np.float64))
    energy_map = _multiscale_energy_map(grayscale)

    cells: list[tuple[float, int, int]] = []
    for row in range(rows):
        for col in range(cols):
            top, left = row * patch_size, col * patch_size
            cell_energy = float(energy_map[top : top + patch_size, left : left + patch_size].mean())
            cells.append((cell_energy, row, col))

    # Stable sort on (-energy, row, col) makes tie-breaking deterministic
    # regardless of dict/set iteration order upstream.
    cells.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = cells[:top_k]

    patch_boxes = [(row * patch_size, col * patch_size, patch_size, patch_size) for _, row, col in selected]
    energy_scores = [energy for energy, _, _ in selected]
    return PatchSelection(patch_boxes=patch_boxes, energy_scores=energy_scores, image_shape=(height, width))


def prepare_texture_patch_views(
    image: np.ndarray, *, patch_size: int, patch_count: int
) -> PreparedPatchViews:
    """Pad an image to one patch, then materialize the ranked selected patches."""

    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if patch_count <= 0:
        raise ValueError("patch_count must be positive")

    source = np.asarray(image)
    if source.ndim != 3 or source.shape[-1] < 3:
        raise ValueError("Expected an RGB image array")
    if not np.all(np.isfinite(source)):
        raise ValueError("Image array contains non-finite values")
    height, width = source.shape[:2]
    pad_height = max(0, patch_size - height)
    pad_width = max(0, patch_size - width)
    top_pad, bottom_pad = pad_height // 2, pad_height - pad_height // 2
    left_pad, right_pad = pad_width // 2, pad_width - pad_width // 2
    if pad_height or pad_width:
        padded = np.pad(
            source,
            ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
            mode="edge",
        )
    else:
        padded = source

    selection = select_texture_patches(padded, patch_size=patch_size, top_k=patch_count)
    boxes = tuple(selection.patch_boxes)
    patches = tuple(
        np.array(padded[top : top + height_, left : left + width_], copy=True)
        for top, left, height_, width_ in boxes
    )
    mask = tuple([True] * len(patches) + [False] * (patch_count - len(patches)))
    return PreparedPatchViews(
        patches=patches,
        patch_boxes=boxes,
        availability_mask=mask,
        original_shape=(height, width),
        padded_shape=tuple(padded.shape[:2]),
    )


def texture_patch_cache_key(
    *,
    image_sha256: str,
    patch_boxes: tuple[tuple[int, int, int, int], ...],
    model_identifier: str,
    resolved_revision: str,
    preprocessing_version: str,
    extractor_version: str,
) -> str:
    """Return a stable SHA-256 key for all patch-view identity inputs."""

    payload = {
        "extractor_version": extractor_version,
        "image_sha256": image_sha256,
        "model_identifier": model_identifier,
        "patch_boxes": patch_boxes,
        "preprocessing_version": preprocessing_version,
        "resolved_revision": resolved_revision,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
