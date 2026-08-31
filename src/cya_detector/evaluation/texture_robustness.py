"""Frozen contract for Task 9 texture robustness Stage 1."""

from __future__ import annotations

import json
import math
import os
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cya_detector.config import ConfigError
from cya_detector.data.manifest import (
    read_manifest,
    sha256_file,
    write_json,
    write_manifest,
)
from cya_detector.transforms import benchmark_cells
from cya_detector.transforms.benchmark import TransformCell
from cya_detector.transforms.materialize import materialize_benchmarks

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
_BINARY_LABELS = frozenset({"authentic", "ai_generated"})
_FIXED_Q96_SUFFIX = "__matched_clean__fixed_q96"


class TextureRobustnessError(RuntimeError):
    """Raised when the locked Stage-1 evaluation boundary is violated."""


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


def _resolved(path: Path, *, owner: str) -> Path:
    try:
        return path.resolve()
    except OSError as exc:
        raise TextureRobustnessError(f"Could not resolve {owner}: {path}") from exc


def _require_locked_parent_rows(
    *,
    input_manifest: Path,
    output_root: Path,
    output_manifest: Path,
    report_path: Path,
    cells: Sequence[TransformCell],
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]], int]:
    resolved_paths = {
        "input_manifest": _resolved(input_manifest, owner="input_manifest"),
        "output_manifest": _resolved(output_manifest, owner="output_manifest"),
        "report_path": _resolved(report_path, owner="report_path"),
    }
    if len(set(resolved_paths.values())) != len(resolved_paths):
        raise TextureRobustnessError(
            "Input manifest, output manifest, and report must not alias"
        )
    resolved_output_root = _resolved(output_root, owner="output_root")
    if resolved_output_root in resolved_paths.values() or (
        output_root.exists() and not output_root.is_dir()
    ):
        raise TextureRobustnessError("output_root must be a distinct directory")
    try:
        input_rows = sorted(
            read_manifest(input_manifest), key=lambda row: row.get("sample_id", "")
        )
    except (OSError, ValueError) as exc:
        raise TextureRobustnessError(
            f"Could not read fixed-Q96 input manifest {input_manifest}: {exc}"
        ) from exc
    forbidden_splits = sorted(
        {
            row.get("split", "")
            for row in input_rows
            if row.get("split") not in {"seed_train", "selection_val"}
        }
    )
    if forbidden_splits:
        raise TextureRobustnessError(
            "Stage-1 source manifest contains forbidden split(s): "
            + ", ".join(repr(value) for value in forbidden_splits)
        )
    parents = [row for row in input_rows if row.get("split") == "selection_val"]
    if not parents:
        raise TextureRobustnessError("Stage-1 input manifest must contain selection_val parents")

    by_id: dict[str, dict[str, str]] = {}
    planned_destinations: set[Path] = set()
    cells_by_id = {cell.cell_id: cell for cell in cells}
    for row in parents:
        sample_id = row.get("sample_id", "")
        if not sample_id or Path(sample_id).name != sample_id or sample_id in by_id:
            raise TextureRobustnessError(
                f"Stage-1 parent sample IDs must be unique safe basenames: {sample_id!r}"
            )
        if row.get("image_view") != "matched_clean" or row.get("transform") != "clean":
            raise TextureRobustnessError(
                "Stage-1 parents must be direct matched-clean views; transform chaining is forbidden"
            )
        if any(
            row.get(field)
            for field in (
                "transform_parameter",
                "realized_parameters",
                "transform_version",
            )
        ):
            raise TextureRobustnessError(
                "Stage-1 clean parents must not contain transform-chain provenance"
            )
        if not sample_id.endswith(_FIXED_Q96_SUFFIX):
            raise TextureRobustnessError("Stage-1 accepts only fixed_q96 matched-clean parent IDs")
        if (
            row.get("normalization_codec") != "JPEG"
            or row.get("normalization_quality") != "96"
            or row.get("output_storage_format") != "JPEG"
        ):
            raise TextureRobustnessError("Stage-1 accepts only the fixed-Q96 JPEG policy")
        if row.get("label") not in _BINARY_LABELS:
            raise TextureRobustnessError(f"Unsupported Stage-1 label: {row.get('label')!r}")
        if not row.get("source_id") or not row.get("parent_id"):
            raise TextureRobustnessError("Stage-1 parents require source_id and source parent_id")

        image_path = Path(row.get("image_path", ""))
        clean_image_path = Path(row.get("clean_image_path", ""))
        if not image_path.is_file():
            raise TextureRobustnessError(f"Stage-1 parent image is missing: {image_path}")
        if _resolved(image_path, owner=f"image_path[{sample_id}]") != _resolved(
            clean_image_path, owner=f"clean_image_path[{sample_id}]"
        ):
            raise TextureRobustnessError("Stage-1 image_path must equal clean_image_path")
        raw_declared_hash = row.get("sha256", "")
        declared_hash = raw_declared_hash.lower()
        if len(declared_hash) != 64 or any(
            character not in "0123456789abcdef" for character in declared_hash
        ) or raw_declared_hash != declared_hash:
            raise TextureRobustnessError(f"Invalid parent SHA-256 for {sample_id}")
        if sha256_file(image_path) != declared_hash:
            raise TextureRobustnessError(f"Parent SHA-256 mismatch for {sample_id}")

        by_id[sample_id] = row
        for cell_id, cell in cells_by_id.items():
            suffix = ".jpg" if cell.output_format == "JPEG" else ".png"
            destination = _resolved(
                output_root / cell_id / f"{sample_id}{suffix}",
                owner=f"image_destination[{sample_id}:{cell_id}]",
            )
            if (
                destination in resolved_paths.values()
                or destination in planned_destinations
            ):
                raise TextureRobustnessError(f"Unsafe Stage-1 output path alias: {destination}")
            planned_destinations.add(destination)
    return parents, by_id, len(input_rows)


def _staging_sibling(path: Path, token: str) -> Path:
    return path.with_name(f".{path.name}.stage1-{token}.tmp")


def _validate_materialized_rows(
    *,
    rows: list[dict[str, str]],
    parents_by_id: dict[str, dict[str, str]],
    output_root: Path,
) -> None:
    expected_pairs = {
        (parent_id, cell_id)
        for parent_id in parents_by_id
        for cell_id in STAGE1_CELL_IDS
    }
    observed_pairs: set[tuple[str, str]] = set()
    for row in rows:
        parent_id = row.get("parent_id", "")
        parent = parents_by_id.get(parent_id)
        if parent is None:
            raise TextureRobustnessError(f"Unknown materialized parent: {parent_id!r}")
        try:
            parameters = json.loads(row.get("transform_parameter", ""))
            cell_id = parameters["cell_id"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise TextureRobustnessError("Invalid materialized cell provenance") from exc
        pair = (parent_id, cell_id)
        if pair not in expected_pairs or pair in observed_pairs:
            raise TextureRobustnessError(f"Unexpected or duplicate Stage-1 row: {pair!r}")
        observed_pairs.add(pair)
        if row.get("sample_id") != f"{parent_id}__benchmark__{cell_id}":
            raise TextureRobustnessError(f"Invalid materialized sample identity for {pair!r}")
        if row.get("split") != "selection_val" or row.get("image_view") != "benchmark":
            raise TextureRobustnessError(
                "Materialized rows must remain selection_val benchmark views"
            )
        if row.get("parent_sha256") != parent["sha256"]:
            raise TextureRobustnessError(f"Materialized parent SHA-256 mismatch for {pair!r}")
        if _resolved(Path(row.get("clean_image_path", "")), owner="clean_image_path") != _resolved(
            Path(parent["image_path"]), owner="parent image_path"
        ):
            raise TextureRobustnessError(f"Materialized clean-parent path mismatch for {pair!r}")
        image_path = Path(row.get("image_path", ""))
        try:
            image_path.resolve().relative_to(output_root.resolve())
        except (OSError, ValueError) as exc:
            raise TextureRobustnessError(
                f"Materialized output escaped output_root: {image_path}"
            ) from exc
        if not image_path.is_file() or sha256_file(image_path) != row.get("sha256"):
            raise TextureRobustnessError(f"Materialized output SHA-256 mismatch for {pair!r}")
    if observed_pairs != expected_pairs:
        missing = sorted(expected_pairs - observed_pairs)
        raise TextureRobustnessError(f"Stage-1 materialization is incomplete: {missing[:3]!r}")


def materialize_texture_stage1(
    *,
    input_manifest: Path,
    output_root: Path,
    output_manifest: Path,
    report_path: Path,
    config: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialize the exact nine Stage-1 cells from validated clean parents."""

    contract = validate_robustness_contract(config)
    declared_by_id = {cell.cell_id: cell for cell in benchmark_cells(config)}
    stage1_cells = tuple(declared_by_id[cell_id] for cell_id in contract.cell_ids)
    parents, parents_by_id, source_row_count = _require_locked_parent_rows(
        input_manifest=input_manifest,
        output_root=output_root,
        output_manifest=output_manifest,
        report_path=report_path,
        cells=stage1_cells,
    )
    expected_count = len(parents) * len(contract.cell_ids)
    token = uuid.uuid4().hex
    staged_parents = _staging_sibling(output_manifest, f"parents-{token}")
    staged_manifest = _staging_sibling(output_manifest, token)
    staged_report = _staging_sibling(report_path, token)
    try:
        write_manifest(staged_parents, parents)
        source_manifest_hash = sha256_file(input_manifest)
        selected_manifest_hash = sha256_file(staged_parents)
        report = materialize_benchmarks(
            input_manifest=staged_parents,
            output_root=output_root,
            output_manifest=staged_manifest,
            report_path=staged_report,
            config=config,
            cells=stage1_cells,
            overwrite=overwrite,
        )
        rows = read_manifest(staged_manifest)
        expected_counts = {cell_id: len(parents) for cell_id in contract.cell_ids}
        if (
            report.get("parent_count") != len(parents)
            or report.get("image_count") != expected_count
            or report.get("cell_counts") != dict(sorted(expected_counts.items()))
            or len(rows) != expected_count
        ):
            raise TextureRobustnessError("Materializer did not return the exact Stage-1 matrix")
        _validate_materialized_rows(
            rows=rows,
            parents_by_id=parents_by_id,
            output_root=output_root,
        )
        if report.get("input_manifest_sha256") != selected_manifest_hash:
            raise TextureRobustnessError("Selected parent manifest changed during materialization")
        if sha256_file(input_manifest) != source_manifest_hash:
            raise TextureRobustnessError("Source manifest changed during materialization")
        if report.get("output_manifest_sha256") != sha256_file(staged_manifest):
            raise TextureRobustnessError("Staged output manifest SHA-256 mismatch")

        report.update(
            {
                "output_manifest": str(output_manifest.resolve()),
                "input_manifest": str(input_manifest.resolve()),
                "input_manifest_sha256": source_manifest_hash,
                "selected_parent_manifest_sha256": selected_manifest_hash,
                "source_row_count": source_row_count,
                "ignored_seed_train_count": source_row_count - len(parents),
                "expected_image_count": expected_count,
                "stage1_cell_ids": list(contract.cell_ids),
                "matching_policy": "fixed_q96",
                "selection_split": "selection_val",
                "direct_clean_parents_only": True,
            }
        )
        write_json(staged_report, report)
        output_manifest.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged_manifest, output_manifest)
        os.replace(staged_report, report_path)
        return report
    except TextureRobustnessError:
        raise
    except Exception as exc:
        raise TextureRobustnessError(f"Stage-1 materialization failed: {exc}") from exc
    finally:
        staged_parents.unlink(missing_ok=True)
        staged_parents.with_suffix(staged_parents.suffix + ".tmp").unlink(
            missing_ok=True
        )
        staged_manifest.unlink(missing_ok=True)
        staged_manifest.with_suffix(staged_manifest.suffix + ".tmp").unlink(missing_ok=True)
        staged_report.unlink(missing_ok=True)
        staged_report.with_suffix(staged_report.suffix + ".tmp").unlink(missing_ok=True)
