"""Task 8 - single-image PRNU-coherence residual proxy.

See docs/planning/tasks7to9_gameplan.md and the PRNU sections of
docs/architecture/techStack.md. This is an experimental single-image
coherence proxy for photo-response non-uniformity, not real camera
fingerprint identification: no reference camera image set is available, so
there is no normalized correlation against a known sensor fingerprint here.
Deterministic only; never an authenticity verdict on its own.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from cya_detector.features.common import FeatureResult, to_grayscale

FEATURE_NAME = "prnu"

MIN_DIMENSION = 32
MIN_STD = 1e-4
DENOISE_SIGMA = 1.0
BLOCK_SIZE = 16
EPS = 1e-8


def _residual(grayscale: np.ndarray) -> np.ndarray:
    denoised = gaussian_filter(grayscale, sigma=DENOISE_SIGMA)
    return grayscale - denoised


def _luminance_residual_coupling(grayscale: np.ndarray, residual: np.ndarray) -> float:
    luminance = grayscale.ravel()
    magnitude = np.abs(residual).ravel()
    if luminance.std() < EPS or magnitude.std() < EPS:
        return 0.0
    correlation = np.corrcoef(luminance, magnitude)[0, 1]
    return float(np.clip(correlation, -1.0, 1.0)) if np.isfinite(correlation) else 0.0


def _block_self_consistency(residual: np.ndarray) -> float:
    """Mean pairwise correlation between block-wise residual signatures.

    A weak self-consistency proxy: repeating block-scale structure in the
    residual (rather than a real multi-image sensor fingerprint) that could
    indicate a coherent pattern across the frame.
    """

    height, width = residual.shape
    block = min(BLOCK_SIZE, height, width)
    if block < 4:
        return 0.0

    blocks = []
    for top in range(0, height - block + 1, block):
        for left in range(0, width - block + 1, block):
            patch = residual[top : top + block, left : left + block]
            if patch.std() > EPS:
                blocks.append((patch - patch.mean()).ravel() / patch.std())

    if len(blocks) < 2:
        return 0.0

    stacked = np.stack(blocks)
    correlation_matrix = np.corrcoef(stacked)
    upper = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
    upper = upper[np.isfinite(upper)]
    if upper.size == 0:
        return 0.0
    return float(np.mean(np.abs(upper)))


def _cfa_periodicity(residual: np.ndarray) -> tuple[float, float]:
    """Row/column energy ratio at the Nyquist (period-2) frequency.

    A simplified proxy for Bayer color-filter-array periodicity: a strong
    peak at the highest spatial frequency along rows/columns can indicate a
    regular sensor-grid pattern surviving in the residual.
    """

    def nyquist_ratio(signal_1d: np.ndarray) -> float:
        if signal_1d.size < 4:
            return 0.0
        spectrum = np.abs(np.fft.rfft(signal_1d - signal_1d.mean()))
        if spectrum.size < 2 or spectrum.mean() <= EPS:
            return 0.0
        return float(spectrum[-1] / (spectrum.mean() + EPS))

    row_energy = residual.mean(axis=1)
    col_energy = residual.mean(axis=0)
    return nyquist_ratio(row_energy), nyquist_ratio(col_energy)


def extract_prnu_features(image: np.ndarray) -> FeatureResult:
    """Extract deterministic PRNU-coherence residual summary features.

    `image` is an RGB float32 array in [0, 1], as returned by
    `cya_detector.features.common.load_image_array`.
    """

    height, width = image.shape[0], image.shape[1]
    grayscale = to_grayscale(np.asarray(image, dtype=np.float64))

    valid = height >= MIN_DIMENSION and width >= MIN_DIMENSION and grayscale.std() > MIN_STD
    if not valid:
        return FeatureResult(
            name=FEATURE_NAME,
            values={
                "residual_variance": 0.0,
                "residual_energy": 0.0,
                "luminance_residual_coupling": 0.0,
                "block_self_consistency": 0.0,
                "cfa_row_periodicity": 0.0,
                "cfa_col_periodicity": 0.0,
            },
            valid=False,
            confidence=0.0,
            notes="Image too small or near-constant for PRNU-coherence analysis",
        )

    residual = _residual(grayscale)
    row_periodicity, col_periodicity = _cfa_periodicity(residual)
    confidence = float(np.clip(min(height, width) / 256.0, 0.0, 1.0))

    return FeatureResult(
        name=FEATURE_NAME,
        values={
            "residual_variance": float(residual.var()),
            "residual_energy": float(np.mean(residual**2)),
            "luminance_residual_coupling": _luminance_residual_coupling(grayscale, residual),
            "block_self_consistency": _block_self_consistency(residual),
            "cfa_row_periodicity": row_periodicity,
            "cfa_col_periodicity": col_periodicity,
        },
        valid=True,
        confidence=confidence,
    )
