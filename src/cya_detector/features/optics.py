"""Task 8 - chromatic aberration and optional radial lens distortion fitting.

See docs/planning/tasks7to9_gameplan.md and the optics sections of
docs/architecture/techStack.md. Chromatic aberration is estimated as a
global R-vs-G and B-vs-G channel misalignment via phase correlation, a
simplified proxy for the Johnson-Farid radial expansion/contraction model.
Radial lens distortion (stretch) is estimated only when enough long,
approximately-straight edges are detected; otherwise it stays masked
neutral. Insufficient edge support or a poor fit must never become
`authentic` or `ai_generated` evidence on its own. Deterministic only.
"""

from __future__ import annotations

import cv2
import numpy as np

from cya_detector.features.common import FeatureResult

FEATURE_NAME = "optics"

MIN_DIMENSION = 32
MIN_EDGE_DENSITY = 0.01
MIN_LINE_SUPPORT = 3
CANNY_LOW = 50
CANNY_HIGH = 150
EPS = 1e-8


def _edge_density(grayscale_uint8: np.ndarray) -> tuple[np.ndarray, float]:
    edges = cv2.Canny(grayscale_uint8, CANNY_LOW, CANNY_HIGH)
    density = float(np.count_nonzero(edges)) / edges.size
    return edges, density


def _phase_correlation_shift(reference: np.ndarray, moving: np.ndarray) -> tuple[float, float, float]:
    """Sub-pixel (dy, dx) shift of `moving` relative to `reference`, plus response."""

    height, width = reference.shape
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    ref = reference.astype(np.float32) * window
    mov = moving.astype(np.float32) * window
    (dx, dy), response = cv2.phaseCorrelate(ref, mov)
    return float(dy), float(dx), float(response)


def _fit_radial_distortion(edges: np.ndarray) -> tuple[float, float, int]:
    """Estimate a barrel/pincushion coefficient from detected line segments.

    Long line segments are found with a probabilistic Hough transform; each
    segment's constituent edge pixels are checked for bowing away from the
    fitted straight line. The mean signed deviation (normalized by segment
    length) is used as a lightweight distortion coefficient proxy, not a
    calibrated plumb-line fit.
    """

    min_length = max(edges.shape) // 6
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=40, minLineLength=min_length, maxLineGap=5
    )
    if lines is None or len(lines) < MIN_LINE_SUPPORT:
        return 0.0, 0.0, 0 if lines is None else len(lines)

    ys, xs = np.nonzero(edges)
    edge_points = np.column_stack([xs, ys]).astype(np.float64)

    # cv2.HoughLinesP returns shape (N, 1, 4) on some OpenCV builds and
    # (N, 4) on others; reshape defensively rather than pin a version.
    deviations: list[float] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        length = np.hypot(x2 - x1, y2 - y1)
        if length < 1:
            continue
        direction = np.array([x2 - x1, y2 - y1]) / length
        normal = np.array([-direction[1], direction[0]])

        along = (edge_points[:, 0] - x1) * direction[0] + (edge_points[:, 1] - y1) * direction[1]
        near_segment = (along >= -2) & (along <= length + 2)
        if not near_segment.any():
            continue
        perpendicular = (edge_points[near_segment, 0] - x1) * normal[0] + (
            edge_points[near_segment, 1] - y1
        ) * normal[1]
        nearby = perpendicular[np.abs(perpendicular) < 3.0]
        if nearby.size < 3:
            continue
        deviations.append(float(np.mean(nearby)) / length)

    if len(deviations) < MIN_LINE_SUPPORT:
        return 0.0, 0.0, len(deviations)

    coefficient = float(np.mean(deviations))
    residual = float(np.std(deviations))
    return coefficient, residual, len(deviations)


def extract_optics_features(image: np.ndarray) -> FeatureResult:
    """Extract deterministic chromatic-aberration/radial-distortion features.

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
            notes="Image too small for optical analysis",
        )

    rgb_uint8 = np.clip(np.asarray(image, dtype=np.float64) * 255.0, 0, 255).astype(np.uint8)
    red, green, blue = rgb_uint8[..., 0], rgb_uint8[..., 1], rgb_uint8[..., 2]
    grayscale_uint8 = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    edges, edge_density = _edge_density(grayscale_uint8)

    if edge_density < MIN_EDGE_DENSITY:
        return FeatureResult(
            name=FEATURE_NAME,
            values={
                "chromatic_shift_r_dy": 0.0,
                "chromatic_shift_r_dx": 0.0,
                "chromatic_shift_r_response": 0.0,
                "chromatic_shift_b_dy": 0.0,
                "chromatic_shift_b_dx": 0.0,
                "chromatic_shift_b_response": 0.0,
                "radial_distortion_coefficient": 0.0,
                "radial_distortion_residual": 0.0,
                "radial_distortion_line_support": 0.0,
                "edge_density": edge_density,
            },
            valid=False,
            confidence=0.0,
            notes="Insufficient edge support for chromatic-aberration/distortion fitting",
        )

    r_dy, r_dx, r_response = _phase_correlation_shift(green.astype(np.float64), red.astype(np.float64))
    b_dy, b_dx, b_response = _phase_correlation_shift(green.astype(np.float64), blue.astype(np.float64))

    distortion_coefficient, distortion_residual, line_support = _fit_radial_distortion(edges)
    distortion_valid = line_support >= MIN_LINE_SUPPORT

    confidence = float(np.clip(edge_density / 0.05, 0.0, 1.0))

    return FeatureResult(
        name=FEATURE_NAME,
        values={
            "chromatic_shift_r_dy": r_dy,
            "chromatic_shift_r_dx": r_dx,
            "chromatic_shift_r_response": r_response,
            "chromatic_shift_b_dy": b_dy,
            "chromatic_shift_b_dx": b_dx,
            "chromatic_shift_b_response": b_response,
            "radial_distortion_coefficient": distortion_coefficient if distortion_valid else 0.0,
            "radial_distortion_residual": distortion_residual if distortion_valid else 0.0,
            "radial_distortion_line_support": float(line_support),
            "edge_density": edge_density,
        },
        valid=True,
        confidence=confidence,
        notes="" if distortion_valid else "Insufficient straight-line support for radial distortion fit",
    )
