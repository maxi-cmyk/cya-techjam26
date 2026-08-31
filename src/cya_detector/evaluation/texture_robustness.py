"""Frozen contract for Task 9 texture robustness Stage 1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from cya_detector.config import ConfigError
from cya_detector.transforms import benchmark_cells

STAGE1_CELL_IDS = (
    "jpeg_q90",
    "jpeg_q70",
    "jpeg_q50",
    "jpeg_q30",
    "blur_sigma_0.5",
    "blur_sigma_1.0",
    "blur_sigma_2.0",
    "resize_scale_0.5",
    "resize_scale_0.25",
)

_EXPERIMENT_NAME = "robustness_stage1_v1"
_VARIANTS = ("global_only", "local_only", "global_local")
_SEEDS = (42, 43, 44)
_CONTROLLING_COMPARATORS = ("global_only", "controlled_rine")
_AGGREGATE_CLASS_TOLERANCE = 0.01
_WORST_CELL_TOLERANCE = 0.03
_SECTION_KEYS = frozenset(
    {
        "experiment_name",
        "cell_ids",
        "aggregate_class_tolerance",
        "worst_cell_tolerance",
    }
)


@dataclass(frozen=True)
class RobustnessContract:
    """Immutable Stage-1 matrix and decision tolerances."""

    experiment_name: str
    cell_ids: tuple[str, ...]
    variants: tuple[str, ...]
    seeds: tuple[int, ...]
    controlling_comparators: tuple[str, ...]
    aggregate_class_tolerance: float
    worst_cell_tolerance: float


def _require_frozen_tolerance(value: Any, expected: float, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) != expected
    ):
        raise ConfigError(f"{name} must remain {expected}")
    return float(value)


def validate_robustness_contract(config: dict[str, Any]) -> RobustnessContract:
    """Validate and return the exact frozen Task 9 Stage-1 contract."""

    section = config.get("texture_robustness_stage1")
    if not isinstance(section, dict):
        raise ConfigError("Configuration section texture_robustness_stage1 must be an object")
    missing = sorted(_SECTION_KEYS - section.keys())
    if missing:
        raise ConfigError(
            "Missing texture_robustness_stage1 key(s): " + ", ".join(missing)
        )
    unknown = sorted(section.keys() - _SECTION_KEYS)
    if unknown:
        raise ConfigError(
            "Unknown texture_robustness_stage1 key(s): " + ", ".join(unknown)
        )

    if section["experiment_name"] != _EXPERIMENT_NAME:
        raise ConfigError(
            "texture_robustness_stage1.experiment_name must remain "
            f"{_EXPERIMENT_NAME}"
        )
    configured_cells = section["cell_ids"]
    if not isinstance(configured_cells, list) or tuple(configured_cells) != STAGE1_CELL_IDS:
        raise ConfigError("Texture robustness Stage-1 cell IDs must remain exact and ordered")

    available_cell_ids = {cell.cell_id for cell in benchmark_cells(config)}
    missing_benchmark_cells = sorted(set(STAGE1_CELL_IDS) - available_cell_ids)
    if missing_benchmark_cells:
        raise ConfigError(
            "Texture robustness Stage-1 cells are absent from benchmark_transforms: "
            + ", ".join(missing_benchmark_cells)
        )

    texture = config.get("texture")
    if not isinstance(texture, dict):
        raise ConfigError("Configuration section texture must be an object")
    if tuple(texture.get("variants", ())) != _VARIANTS:
        raise ConfigError("Texture robustness variants must remain locked to the clean pilot")
    if tuple(texture.get("seeds", ())) != _SEEDS:
        raise ConfigError("Texture robustness seeds must remain locked to the clean pilot")

    aggregate_class_tolerance = _require_frozen_tolerance(
        section["aggregate_class_tolerance"],
        _AGGREGATE_CLASS_TOLERANCE,
        name="texture_robustness_stage1.aggregate_class_tolerance",
    )
    worst_cell_tolerance = _require_frozen_tolerance(
        section["worst_cell_tolerance"],
        _WORST_CELL_TOLERANCE,
        name="texture_robustness_stage1.worst_cell_tolerance",
    )

    return RobustnessContract(
        experiment_name=_EXPERIMENT_NAME,
        cell_ids=STAGE1_CELL_IDS,
        variants=_VARIANTS,
        seeds=_SEEDS,
        controlling_comparators=_CONTROLLING_COMPARATORS,
        aggregate_class_tolerance=aggregate_class_tolerance,
        worst_cell_tolerance=worst_cell_tolerance,
    )
