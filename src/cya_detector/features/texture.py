"""Task 9 - texture-aware local-detail patch selection under a fixed budget.

See docs/planning/tasks7to9_gameplan.md. Selects a fixed top-k set of
non-overlapping, highest-detail patches from a multi-scale Laplacian/Sobel
energy map, while retaining the global view so patch selection cannot
discard semantic context. Patch selection is deterministic; texture energy
chooses where to look and is never an authenticity threshold on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

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
