"""Frozen contract for Task 9 texture robustness Stage 1."""

from __future__ import annotations

import json
import csv
import hashlib
import math
import os
import time
import uuid
from contextlib import nullcontext
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
import numpy as np

from cya_detector.config import ConfigError
from cya_detector.models.clip_baseline import load_frozen_clip, require_ml_dependencies
from cya_detector.models.texture import build_texture_head
from cya_detector.features.texture import prepare_texture_patch_views
from cya_detector.predictions import read_predictions
from cya_detector.data.manifest import (
    read_manifest,
    sha256_file,
    write_json,
    write_manifest,
)
from cya_detector.transforms import benchmark_cells
from cya_detector.transforms.benchmark import TransformCell, apply_benchmark, derive_seed
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
_EVALUATION_SCHEMA_VERSION = 1
_EVALUATION_FIELDS = (
    "sample_id", "parent_id", "source_id", "split", "label", "cell_id",
    "cell_parameters", "variant", "seed", "logit", "probability", "prediction",
    "paired_clean_probability", "paired_clean_prediction", "checkpoint_sha256",
    "input_sha256", "parent_sha256", "global_feature_sha256",
    "patch_feature_sha256", "cache_contract_sha256", "patch_boxes",
    "available_patch_count", "matching_policy", "transform", "transform_parameter",
)


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
    cells: Sequence[TransformCell],
    config: dict[str, Any],
) -> None:
    cells_by_id = {cell.cell_id: cell for cell in cells}
    expected_pairs = {
        (parent_id, cell_id)
        for parent_id in parents_by_id
        for cell_id in cells_by_id
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
        cell = cells_by_id[cell_id]
        if row.get("sample_id") != f"{parent_id}__benchmark__{cell_id}":
            raise TextureRobustnessError(f"Invalid materialized sample identity for {pair!r}")
        if row.get("split") != "selection_val" or row.get("image_view") != "benchmark":
            raise TextureRobustnessError(
                "Materialized rows must remain selection_val benchmark views"
            )
        for field in ("label", "source_id", "source_path", "dataset_name"):
            if row.get(field) != parent.get(field):
                raise TextureRobustnessError(
                    f"Materialized {field} does not match its clean parent for {pair!r}"
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

        expected_declared = {
            "cell_id": cell.cell_id,
            "name": cell.name,
            "output_format": cell.output_format,
            "parameter": cell.parameter,
            "stochastic": cell.stochastic,
        }
        expected_declared_json = json.dumps(
            expected_declared, sort_keys=True, separators=(",", ":")
        )
        if row.get("transform_parameter") != expected_declared_json:
            raise TextureRobustnessError(
                f"Materialized declared parameters do not match locked cell {cell_id!r}"
            )
        expected_seed = derive_seed(config["runtime"]["seed"], parent_id, cell_id)
        if row.get("transform_seed") != str(expected_seed):
            raise TextureRobustnessError(
                f"Materialized transform seed is not deterministic for {pair!r}"
            )
        if row.get("transform") != cell.name:
            raise TextureRobustnessError(
                f"Materialized transform name does not match locked cell {cell_id!r}"
            )
        if row.get("output_storage_format") != cell.output_format:
            raise TextureRobustnessError(
                f"Materialized storage format does not match locked cell {cell_id!r}"
            )
        engine = config["transform_engine"]
        if (
            row.get("transform_version") != engine["version"]
            or row.get("preprocessing_version") != engine["preprocessing_version"]
        ):
            raise TextureRobustnessError(
                f"Materialized engine versions do not match the frozen contract for {pair!r}"
            )
        try:
            with Image.open(parent["image_path"]) as parent_image:
                parent_image.load()
                expected_realized = apply_benchmark(
                    parent_image,
                    cell,
                    parent_id,
                    config["runtime"]["seed"],
                ).realized
        except (OSError, ValueError) as exc:
            raise TextureRobustnessError(
                f"Could not verify realized transform parameters for {pair!r}"
            ) from exc
        expected_realized_json = json.dumps(
            expected_realized, sort_keys=True, separators=(",", ":")
        )
        if row.get("realized_parameters") != expected_realized_json:
            raise TextureRobustnessError(
                f"Materialized realized parameters do not match locked cell {cell_id!r}"
            )
    if observed_pairs != expected_pairs:
        missing = sorted(expected_pairs - observed_pairs)
        raise TextureRobustnessError(f"Stage-1 materialization is incomplete: {missing[:3]!r}")


def _publish_materialization_bundle(
    *,
    staged_manifest: Path,
    staged_report: Path,
    output_manifest: Path,
    report_path: Path,
    token: str,
) -> None:
    """Publish report-last and restore the prior manifest if report publication fails."""

    previous_manifest = output_manifest.read_bytes() if output_manifest.is_file() else None
    rollback_manifest = _staging_sibling(output_manifest, f"rollback-{token}")
    manifest_published = False
    try:
        os.replace(staged_manifest, output_manifest)
        manifest_published = True
        os.replace(staged_report, report_path)
    except OSError as publication_error:
        if manifest_published:
            try:
                if previous_manifest is None:
                    output_manifest.unlink(missing_ok=True)
                else:
                    rollback_manifest.write_bytes(previous_manifest)
                    os.replace(rollback_manifest, output_manifest)
            except OSError as rollback_error:
                raise TextureRobustnessError(
                    "Stage-1 bundle publication failed and manifest rollback failed"
                ) from rollback_error
        raise TextureRobustnessError(
            "Stage-1 bundle publication failed before the report completion marker"
        ) from publication_error
    finally:
        rollback_manifest.unlink(missing_ok=True)


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
            cells=stage1_cells,
            config=config,
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
        _publish_materialization_bundle(
            staged_manifest=staged_manifest,
            staged_report=staged_report,
            output_manifest=output_manifest,
            report_path=report_path,
            token=token,
        )
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


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_stage1_evaluation_rows(
    *, transformed_manifest: Path, materialization_report: Path, contract: RobustnessContract
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    try:
        rows = read_manifest(transformed_manifest)
        report = json.loads(materialization_report.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise TextureRobustnessError("Could not read the Stage-1 materialization bundle") from exc
    if not rows:
        raise TextureRobustnessError("Stage-1 transformed manifest is empty")
    if any(row.get("split") != "selection_val" for row in rows):
        raise TextureRobustnessError("Stage-1 evaluation accepts selection_val only")
    if any(row.get("image_view") != "benchmark" for row in rows):
        raise TextureRobustnessError("Stage-1 evaluation accepts benchmark views only")
    if report.get("stage1_cell_ids") != list(contract.cell_ids):
        raise TextureRobustnessError("Materialization report does not match the locked cell contract")
    if report.get("matching_policy") != "fixed_q96" or not report.get("direct_clean_parents_only"):
        raise TextureRobustnessError("Materialization report lacks fixed-Q96 direct-parent provenance")
    if report.get("output_manifest_sha256") != sha256_file(transformed_manifest):
        raise TextureRobustnessError("Transformed manifest hash does not match its report")

    expected_pairs: set[tuple[str, str]] = set()
    observed_pairs: set[tuple[str, str]] = set()
    labels: dict[str, str] = {}
    for row in rows:
        try:
            parameters = json.loads(row.get("transform_parameter", ""))
            cell_id = parameters["cell_id"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise TextureRobustnessError("Invalid Stage-1 transform cell provenance") from exc
        parent_id = row.get("parent_id", "")
        if cell_id not in contract.cell_ids or not parent_id:
            raise TextureRobustnessError("Unexpected Stage-1 cell or missing clean parent")
        pair = (parent_id, cell_id)
        if pair in observed_pairs:
            raise TextureRobustnessError("Duplicate Stage-1 parent/cell prediction input")
        observed_pairs.add(pair)
        labels.setdefault(parent_id, row.get("label", ""))
        if labels[parent_id] != row.get("label") or row.get("label") not in _BINARY_LABELS:
            raise TextureRobustnessError("Stage-1 labels do not align across transformed cells")
        image_path = Path(row.get("image_path", ""))
        if not image_path.is_file() or sha256_file(image_path) != row.get("sha256"):
            raise TextureRobustnessError(f"Transformed image hash mismatch: {image_path}")
        parent_hash = row.get("parent_sha256", "")
        if len(parent_hash) != 64:
            raise TextureRobustnessError("Stage-1 row lacks clean-parent SHA-256 provenance")
    for parent_id in labels:
        expected_pairs.update((parent_id, cell_id) for cell_id in contract.cell_ids)
    if observed_pairs != expected_pairs:
        raise TextureRobustnessError("Stage-1 transformed matrix is incomplete or misaligned")
    return sorted(rows, key=lambda row: (row["sample_id"], row["transform_parameter"])), report


def _cache_path(root: Path, family: str, digest: str) -> Path:
    return root / family / digest[:2] / f"{digest}.pt"


def _atomic_torch(path: Path, payload: Any) -> None:
    torch, _, _ = require_ml_dependencies()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.pt")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_valid_cache(path: Path, *, contract: dict[str, Any], kind: str) -> Any | None:
    torch, _, _ = require_ml_dependencies()
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("cache_contract") != contract:
            return None
        if kind == "global":
            value = payload["global_features"]
            valid = value.ndim == 2 and value.dtype.is_floating_point
        else:
            value, mask = payload["patch_features"], payload["patch_mask"]
            valid = (
                value.ndim == 2 and value.dtype.is_floating_point and mask.ndim == 1
                and mask.dtype == torch.bool and value.shape[0] == mask.shape[0]
                and bool(mask.any())
            )
        if not valid or not bool(torch.isfinite(value).all()):
            return None
        return payload
    except (OSError, KeyError, RuntimeError, TypeError, ValueError):
        return None


def _extract_transformed_feature_bank(
    *, transformed_manifest: Path, rows: list[dict[str, str]], cache_root: Path,
    config: dict[str, Any], device: str, batch_size: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract each transformed global/patch representation once under a strict cache key."""

    torch, _, _ = require_ml_dependencies()
    if batch_size <= 0:
        raise TextureRobustnessError("Evaluation batch_size must be positive")
    model_config, texture = config["model"], config["texture"]
    loaded = load_frozen_clip(
        model_config["identifier"], revision=model_config["revision"], device=device
    )
    patch_size, patch_count = int(texture["patch_size"]), int(texture["patch_count"])
    selected_layers = tuple(int(value) for value in model_config["rine_layers"])
    if int(loaded.model.config.num_hidden_layers) < max(selected_layers):
        raise TextureRobustnessError("Frozen CLIP revision lacks a configured RINE layer")
    started = time.perf_counter()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    encoded_image_count = 0
    cache_hits = 0
    results: list[dict[str, Any]] = []
    manifest_hash = sha256_file(transformed_manifest)

    def precision_context() -> Any:
        return (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device.startswith("cuda") else nullcontext()
        )

    for start in range(0, len(rows), batch_size):
        prepared: list[dict[str, Any]] = []
        for row in rows[start : start + batch_size]:
            parameters = json.loads(row["transform_parameter"])
            with Image.open(row["image_path"]) as image:
                source = np.asarray(image.convert("RGB"))
            views = prepare_texture_patch_views(
                source, patch_size=patch_size, patch_count=patch_count
            )
            contract = {
                "schema_version": _EVALUATION_SCHEMA_VERSION,
                "transformed_manifest_sha256": manifest_hash,
                "sample_id": row["sample_id"],
                "input_sha256": row["sha256"],
                "parent_id": row["parent_id"],
                "parent_sha256": row["parent_sha256"],
                "cell_id": parameters["cell_id"],
                "cell_contract": parameters,
                "model_identifier": loaded.identifier,
                "resolved_revision": loaded.resolved_revision,
                "preprocessing_version": config["preprocessing"]["version"],
                "rine_representation_version": model_config["rine_representation_version"],
                "texture_extractor_version": texture["extractor_version"],
                "matching_policy": "fixed_q96",
                "patch_size": patch_size,
                "patch_count": patch_count,
                "patch_boxes": [list(box) for box in views.patch_boxes],
                "layers": list(selected_layers),
            }
            contract_hash = _canonical_sha256(contract)
            global_path = _cache_path(cache_root, "global", contract_hash)
            patch_path = _cache_path(cache_root, "patch", contract_hash)
            global_payload = _load_valid_cache(global_path, contract=contract, kind="global")
            patch_payload = _load_valid_cache(patch_path, contract=contract, kind="patch")
            if global_payload is not None and patch_payload is not None:
                cache_hits += 1
            prepared.append(
                {
                    "row": row, "source": source, "views": views, "contract": contract,
                    "contract_hash": contract_hash, "global_path": global_path,
                    "patch_path": patch_path, "global_payload": global_payload,
                    "patch_payload": patch_payload,
                }
            )

        missing = [
            item for item in prepared
            if item["global_payload"] is None or item["patch_payload"] is None
        ]
        if missing:
            global_pixels = torch.cat(
                [
                    loaded.processor(
                        images=Image.fromarray(item["source"], mode="RGB"),
                        return_tensors="pt",
                    )["pixel_values"]
                    for item in missing
                ],
                dim=0,
            ).to(device)
            with torch.inference_mode(), precision_context():
                output = loaded.model(
                    pixel_values=global_pixels, output_hidden_states=True, return_dict=True
                )
                global_batch = torch.stack(
                    [output.hidden_states[layer][:, 0, :] for layer in selected_layers],
                    dim=1,
                ).detach().float().cpu()
            encoded_image_count += len(missing)

            flattened_patches = [
                (index, patch)
                for index, item in enumerate(missing)
                for patch in item["views"].patches
            ]
            patch_embeddings: list[list[Any]] = [[] for _ in missing]
            for patch_start in range(0, len(flattened_patches), batch_size * patch_count):
                patch_batch = flattened_patches[
                    patch_start : patch_start + batch_size * patch_count
                ]
                patch_pixels = torch.cat(
                    [
                        loaded.processor(
                            images=Image.fromarray(patch, mode="RGB"), return_tensors="pt"
                        )["pixel_values"]
                        for _, patch in patch_batch
                    ],
                    dim=0,
                ).to(device)
                with torch.inference_mode(), precision_context():
                    embeddings = loaded.model(pixel_values=patch_pixels).image_embeds
                embeddings = embeddings.detach().float().cpu()
                for (index, _), embedding in zip(patch_batch, embeddings, strict=True):
                    patch_embeddings[index].append(embedding)
                encoded_image_count += len(patch_batch)

            for index, item in enumerate(missing):
                global_features = global_batch[index]
                patch_features = torch.zeros(
                    (patch_count, loaded.embedding_dimension), dtype=torch.float32
                )
                if patch_embeddings[index]:
                    patch_features[: len(patch_embeddings[index])] = torch.stack(
                        patch_embeddings[index]
                    )
                patch_mask = torch.tensor(
                    item["views"].availability_mask, dtype=torch.bool
                )
                if (
                    not bool(torch.isfinite(global_features).all())
                    or not bool(torch.isfinite(patch_features).all())
                ):
                    raise TextureRobustnessError(
                        "Frozen encoder returned non-finite features"
                    )
                item["global_payload"] = {
                    "global_features": global_features,
                    "cache_contract": item["contract"],
                }
                item["patch_payload"] = {
                    "patch_features": patch_features, "patch_mask": patch_mask,
                    "patch_boxes": item["contract"]["patch_boxes"],
                    "cache_contract": item["contract"],
                }
                _atomic_torch(item["global_path"], item["global_payload"])
                _atomic_torch(item["patch_path"], item["patch_payload"])

        for item in prepared:
            results.append(
                {
                    "manifest_row": item["row"],
                    "global_features": item["global_payload"]["global_features"].float(),
                    "patch_features": item["patch_payload"]["patch_features"].float(),
                    "patch_mask": item["patch_payload"]["patch_mask"],
                    "patch_boxes": tuple(
                        tuple(box) for box in item["contract"]["patch_boxes"]
                    ),
                    "global_feature_sha256": sha256_file(item["global_path"]),
                    "patch_feature_sha256": sha256_file(item["patch_path"]),
                    "cache_contract_sha256": item["contract_hash"],
                }
            )
    return results, {
        "row_count": len(results), "encoded_image_count": encoded_image_count,
        "cache_hit_count": cache_hits, "elapsed_seconds": time.perf_counter() - started,
        "model_identifier": loaded.identifier, "resolved_revision": loaded.resolved_revision,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated())
            if device.startswith("cuda") and torch.cuda.is_available() else 0
        ),
    }


def _load_clean_run(
    root: Path, *, variant: str, seed: int, threshold: float
) -> tuple[Path, str, dict[str, Any]]:
    run = root / variant / f"seed_{seed}"
    checkpoint = run / "checkpoints" / "best_clean.pt"
    predictions = run / "predictions" / "selection_val.csv"
    metadata_path = run / "metadata" / "run_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TextureRobustnessError(f"Clean Task 9 run is incomplete: {run}") from exc
    if (
        metadata.get("status") != "completed" or metadata.get("variant") != variant
        or metadata.get("seed") != seed or metadata.get("matching_policy") != "fixed_q96"
    ):
        raise TextureRobustnessError(f"Clean Task 9 run metadata mismatch: {run}")
    hashes = metadata.get("artifact_sha256")
    if not isinstance(hashes, dict):
        raise TextureRobustnessError(f"Clean Task 9 run lacks committed hashes: {run}")
    required = ("checkpoints/best_clean.pt", "predictions/selection_val.csv")
    if any(relative not in hashes for relative in required):
        raise TextureRobustnessError(f"Clean Task 9 run lacks required hashes: {run}")
    for relative, digest in hashes.items():
        path = run / relative
        if not path.is_file() or digest != sha256_file(path):
            raise TextureRobustnessError(f"Clean Task 9 artifact hash mismatch: {path}")
    optimization = metadata.get("optimization")
    if (
        not isinstance(optimization, dict)
        or float(optimization.get("threshold", math.nan)) != threshold
    ):
        raise TextureRobustnessError(f"Clean Task 9 threshold mismatch: {run}")
    clean_rows = read_predictions(predictions)
    if (
        not clean_rows or any(row.split != "selection_val" or row.transform != "clean" for row in clean_rows)
        or len({row.sample_id for row in clean_rows}) != len(clean_rows)
    ):
        raise TextureRobustnessError(f"Clean Task 9 predictions are invalid: {predictions}")
    return checkpoint, sha256_file(checkpoint), {row.sample_id: row for row in clean_rows}


def _valid_prediction_slice(path: Path, metadata_path: Path, contract: dict[str, Any]) -> bool:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("contract") != contract or metadata.get("csv_sha256") != sha256_file(path):
            return False
        inference = metadata.get("inference")
        if (
            not isinstance(inference, dict)
            or inference.get("sample_count") != metadata.get("row_count")
            or any(
                isinstance(inference.get(key), bool)
                or not isinstance(inference.get(key), (int, float))
                or not math.isfinite(float(inference[key]))
                or float(inference[key]) < 0
                for key in ("latency_seconds", "peak_gpu_memory_bytes")
            )
        ):
            return False
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        return (
            list(rows[0].keys()) == list(_EVALUATION_FIELDS) if rows else False
        ) and len(rows) == metadata.get("row_count") and all(
            math.isfinite(float(row["logit"])) and math.isfinite(float(row["probability"]))
            for row in rows
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _publish_prediction_slice(
    path: Path, metadata_path: Path, rows: list[dict[str, Any]], contract: dict[str, Any],
    inference: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    csv_stage = path.with_name(f".{path.name}.{token}.tmp")
    metadata_stage = metadata_path.with_name(f".{metadata_path.name}.{token}.tmp")
    try:
        with csv_stage.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=_EVALUATION_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        write_json(
            metadata_stage,
            {"status": "completed", "contract": contract, "row_count": len(rows),
             "csv_sha256": sha256_file(csv_stage), "inference": inference},
        )
        os.replace(csv_stage, path)
        os.replace(metadata_stage, metadata_path)
    finally:
        csv_stage.unlink(missing_ok=True)
        metadata_stage.unlink(missing_ok=True)


def evaluate_texture_stage1(
    *, transformed_manifest: Path, materialization_report: Path,
    clean_experiment_root: Path, cache_root: Path, output_root: Path,
    config: dict[str, Any], device: str, overwrite: bool = False,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Evaluate all frozen clean texture heads over the complete Stage-1 matrix."""

    torch, _, _ = require_ml_dependencies()
    contract = validate_robustness_contract(config)
    rows, report = _require_stage1_evaluation_rows(
        transformed_manifest=Path(transformed_manifest),
        materialization_report=Path(materialization_report), contract=contract,
    )
    feature_rows, extraction = _extract_transformed_feature_bank(
        transformed_manifest=Path(transformed_manifest), rows=rows,
        cache_root=Path(cache_root), config=config, device=device, batch_size=batch_size,
    )
    by_cell = {
        cell_id: [
            row for row in feature_rows
            if json.loads(row["manifest_row"]["transform_parameter"])["cell_id"] == cell_id
        ]
        for cell_id in contract.cell_ids
    }
    threshold = float(config["evaluation"]["threshold"])
    if not 0.0 < threshold < 1.0:
        raise TextureRobustnessError("Clean threshold must remain strictly between zero and one")
    completed = resumed = 0
    checkpoint_hashes: dict[Path, str] = {}
    for variant in contract.variants:
        for seed in contract.seeds:
            checkpoint, checkpoint_hash, clean = _load_clean_run(
                Path(clean_experiment_root), variant=variant, seed=seed,
                threshold=threshold,
            )
            checkpoint_hashes[checkpoint] = checkpoint_hash
            first = feature_rows[0]
            global_shape = tuple(first["global_features"].shape)
            patch_shape = tuple(first["patch_features"].shape)
            model = build_texture_head(
                variant=variant, layer_count=global_shape[0],
                global_dimension=global_shape[1], patch_dimension=patch_shape[1],
                fusion_dimension=int(config["texture"]["fusion_dimension"]),
            ).to(device)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if (
                payload.get("stage") != "texture_stage_d" or payload.get("variant") != variant
                or payload.get("seed") != seed or not isinstance(payload.get("model_state_dict"), dict)
            ):
                raise TextureRobustnessError(f"Best-clean checkpoint identity mismatch: {checkpoint}")
            model.load_state_dict(payload["model_state_dict"], strict=True)
            model.eval()
            for cell_id in contract.cell_ids:
                selected = by_cell[cell_id]
                clean_ids = {row["manifest_row"]["parent_id"] for row in selected}
                if clean_ids != set(clean):
                    raise TextureRobustnessError(
                        f"Clean/transformed sample alignment mismatch for {variant}/seed_{seed}"
                    )
                slice_contract = {
                    "schema_version": _EVALUATION_SCHEMA_VERSION,
                    "materialization_report_sha256": sha256_file(Path(materialization_report)),
                    "transformed_manifest_sha256": sha256_file(Path(transformed_manifest)),
                    "variant": variant, "seed": seed, "cell_id": cell_id,
                    "checkpoint_sha256": checkpoint_hash, "threshold": threshold,
                    "calibration": "unchanged_clean_raw_sigmoid",
                    "feature_contract_sha256": _canonical_sha256(
                        sorted(row["cache_contract_sha256"] for row in selected)
                    ),
                }
                path = Path(output_root) / variant / f"seed_{seed}" / f"{cell_id}.csv"
                metadata_path = path.with_suffix(".meta.json")
                if not overwrite and _valid_prediction_slice(path, metadata_path, slice_contract):
                    completed += 1
                    resumed += 1
                    continue
                globals_ = torch.stack([row["global_features"] for row in selected]).to(device)
                patches = torch.stack([row["patch_features"] for row in selected]).to(device)
                masks = torch.stack([row["patch_mask"] for row in selected]).to(device)
                if device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                inference_started = time.perf_counter()
                with torch.inference_mode():
                    logits = model(globals_, patches, masks).squeeze(1).detach().float().cpu()
                    probabilities = torch.sigmoid(logits)
                inference = {
                    "sample_count": len(selected),
                    "latency_seconds": time.perf_counter() - inference_started,
                    "peak_gpu_memory_bytes": (
                        int(torch.cuda.max_memory_allocated())
                        if device.startswith("cuda") and torch.cuda.is_available() else 0
                    ),
                }
                if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(probabilities).all()):
                    raise TextureRobustnessError("Frozen texture head returned non-finite outputs")
                output_rows: list[dict[str, Any]] = []
                for feature, logit, probability in zip(
                    selected, logits.tolist(), probabilities.tolist(), strict=True
                ):
                    row = feature["manifest_row"]
                    clean_row = clean[row["parent_id"]]
                    if clean_row.label != row["label"] or not math.isfinite(clean_row.probability):
                        raise TextureRobustnessError("Paired clean label/probability mismatch")
                    output_rows.append(
                        {
                            "sample_id": row["sample_id"], "parent_id": row["parent_id"],
                            "source_id": row["source_id"], "split": row["split"],
                            "label": row["label"], "cell_id": cell_id,
                            "cell_parameters": row["transform_parameter"], "variant": variant,
                            "seed": seed, "logit": logit, "probability": probability,
                            "prediction": int(probability >= threshold),
                            "paired_clean_probability": clean_row.probability,
                            "paired_clean_prediction": int(clean_row.probability >= threshold),
                            "checkpoint_sha256": checkpoint_hash, "input_sha256": row["sha256"],
                            "parent_sha256": row["parent_sha256"],
                            "global_feature_sha256": feature["global_feature_sha256"],
                            "patch_feature_sha256": feature["patch_feature_sha256"],
                            "cache_contract_sha256": feature["cache_contract_sha256"],
                            "patch_boxes": json.dumps(feature["patch_boxes"], separators=(",", ":")),
                            "available_patch_count": int(feature["patch_mask"].sum().item()),
                            "matching_policy": "fixed_q96", "transform": row["transform"],
                            "transform_parameter": row["transform_parameter"],
                        }
                    )
                _publish_prediction_slice(
                    path, metadata_path, output_rows, slice_contract, inference
                )
                completed += 1
    changed = [
        str(path) for path, digest in checkpoint_hashes.items()
        if sha256_file(path) != digest
    ]
    if changed:
        raise TextureRobustnessError("Best-clean checkpoint changed during evaluation")
    return {
        "status": "completed", "completed_slices": completed,
        "resumed_slices": resumed, "computed_slices": completed - resumed,
        "expected_slices": len(contract.cell_ids) * len(contract.variants) * len(contract.seeds),
        "materialization_report_sha256": sha256_file(Path(materialization_report)),
        "extraction": extraction, "final_test_read": False,
    }
