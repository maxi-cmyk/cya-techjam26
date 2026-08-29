"""Persist deterministic benchmark variants and their provenance."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image
from PIL import __version__ as pillow_version

from cya_detector.data.manifest import (
    MANIFEST_FIELDS,
    read_manifest,
    sha256_file,
    write_json,
    write_manifest,
)
from cya_detector.transforms.benchmark import (
    TransformCell,
    TransformContractError,
    apply_benchmark,
    benchmark_cells,
    derive_seed,
    validate_parent_record,
)


class TransformMaterializationError(RuntimeError):
    """Raised when a benchmark variant cannot be safely persisted."""


def _compact_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _requested_settings(cell: TransformCell) -> dict[str, Any]:
    return {
        "cell_id": cell.cell_id,
        "name": cell.name,
        "output_format": cell.output_format,
        "parameter": cell.parameter,
        "stochastic": cell.stochastic,
    }


def _validated_inputs(
    records: list[dict[str, str]],
    selected_cells: Sequence[TransformCell],
    config: dict[str, Any],
) -> tuple[list[dict[str, str]], list[TransformCell]]:
    parents = sorted(records, key=lambda row: row.get("sample_id", ""))
    parent_ids: set[str] = set()
    for parent in parents:
        validate_parent_record(parent)
        parent_id = parent.get("sample_id", "")
        if not parent_id or Path(parent_id).name != parent_id:
            raise TransformContractError(f"Unsafe benchmark parent sample_id: {parent_id!r}")
        if parent_id in parent_ids:
            raise TransformContractError(f"Duplicate benchmark parent sample_id: {parent_id!r}")
        parent_ids.add(parent_id)

    declared_cells = set(benchmark_cells(config))
    ordered_cells = sorted(selected_cells, key=lambda cell: cell.cell_id)
    seen_cell_ids: set[str] = set()
    for cell in ordered_cells:
        if cell not in declared_cells:
            raise TransformContractError(f"Undeclared benchmark cell: {cell!r}")
        if not cell.cell_id or Path(cell.cell_id).name != cell.cell_id:
            raise TransformContractError(f"Unsafe benchmark cell_id: {cell.cell_id!r}")
        if cell.cell_id in seen_cell_ids:
            raise TransformContractError(f"Duplicate benchmark cell_id: {cell.cell_id!r}")
        seen_cell_ids.add(cell.cell_id)
    return parents, ordered_cells


def _save_verified_sibling(
    *,
    image: Image.Image,
    encoded_bytes: bytes | None,
    destination: Path,
    cell: TransformCell,
    overwrite: bool,
) -> tuple[str, int, int, str, str, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.",
        suffix=f".tmp{destination.suffix}",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if cell.output_format == "JPEG":
            if encoded_bytes is None:
                raise TransformMaterializationError(
                    f"JPEG cell {cell.cell_id!r} did not provide encoded bytes"
                )
            temporary.write_bytes(encoded_bytes)
        else:
            image.save(temporary, format=cell.output_format)

        with Image.open(temporary) as stored:
            stored.verify()
        with Image.open(temporary) as stored:
            stored.load()
            width, height = stored.size
            image_format = stored.format or ""
            mode = stored.mode
        if image_format != cell.output_format:
            raise TransformMaterializationError(
                f"Stored format {image_format!r} does not match {cell.output_format!r}"
            )
        if width < 1 or height < 1:
            raise TransformMaterializationError(
                f"Stored benchmark image has invalid dimensions: {(width, height)!r}"
            )

        expected_hash = sha256_file(temporary)
        if destination.exists():
            existing_hash = sha256_file(destination)
            if existing_hash != expected_hash and not overwrite:
                raise TransformMaterializationError(
                    f"Output collision at {destination}: existing hash differs"
                )
            if existing_hash == expected_hash:
                temporary.unlink()
            else:
                temporary.replace(destination)
        else:
            temporary.replace(destination)
        return expected_hash, width, height, image_format, mode, destination.stat().st_size
    except TransformMaterializationError:
        raise
    except (OSError, ValueError) as exc:
        raise TransformMaterializationError(
            f"Could not persist benchmark output {destination}: {exc}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _materialize_one(
    *,
    parent: dict[str, str],
    cell: TransformCell,
    output_root: Path,
    config: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    parent_id = parent["sample_id"]
    parent_path = Path(parent["image_path"])
    try:
        parent_hash = sha256_file(parent_path)
        with Image.open(parent_path) as source:
            source.load()
            input_width, input_height = source.size
            parent_mode = source.mode
            parent_format = source.format or parent_path.suffix.lstrip(".").upper()
            result = apply_benchmark(
                source,
                cell,
                parent_id,
                config["runtime"]["seed"],
            )
    except TransformContractError:
        raise
    except (OSError, ValueError) as exc:
        raise TransformMaterializationError(
            f"Could not read benchmark parent {parent_path}: {exc}"
        ) from exc

    extension = ".jpg" if cell.output_format == "JPEG" else ".png"
    destination = output_root / cell.cell_id / f"{parent_id}{extension}"
    output_hash, width, height, image_format, mode, file_size = _save_verified_sibling(
        image=result.image,
        encoded_bytes=result.encoded_bytes,
        destination=destination,
        cell=cell,
        overwrite=overwrite,
    )

    engine = config["transform_engine"]
    seed = derive_seed(config["runtime"]["seed"], parent_id, cell.cell_id)
    record: dict[str, Any] = {field: "" for field in MANIFEST_FIELDS}
    record.update(parent)
    record.update(
        {
            "sample_id": f"{parent_id}__benchmark__{cell.cell_id}",
            "parent_id": parent_id,
            "parent_width": input_width,
            "parent_height": input_height,
            "parent_mode": parent_mode,
            "parent_format": parent_format,
            "image_path": str(destination.resolve()),
            "clean_image_path": str(parent_path.resolve()),
            "image_view": "benchmark",
            "sha256": output_hash,
            "perceptual_hash": "",
            "transform": cell.name,
            "transform_parameter": _compact_json(_requested_settings(cell)),
            "transform_seed": seed,
            "width": width,
            "height": height,
            "format": image_format,
            "mode": mode,
            "file_size": file_size,
            "encoder_version": f"Pillow-{pillow_version}",
            "output_storage_format": cell.output_format,
            "parent_sha256": parent_hash,
            "realized_parameters": _compact_json(result.realized),
            "transform_version": engine["version"],
            "preprocessing_version": engine["preprocessing_version"],
            "corruption_error": "",
        }
    )
    if cell.name == "resize":
        intermediate_width, intermediate_height = result.realized["intermediate_size"]
        record.update(
            {
                "resize_scale": cell.parameter,
                "down_interpolation": engine["resize_interpolation"],
                "up_interpolation": engine["resize_interpolation"],
                "resize_library": result.realized["resize_library"],
                "resize_library_version": result.realized["resize_library_version"],
                "antialias": engine["resize_filtering"],
                "dimension_rounding": result.realized["dimension_rounding"],
                "intermediate_width": intermediate_width,
                "intermediate_height": intermediate_height,
            }
        )
    return record


def materialize_benchmarks(
    *,
    input_manifest: Path,
    output_root: Path,
    output_manifest: Path,
    report_path: Path,
    config: dict[str, Any],
    cells: Sequence[TransformCell] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialize one independent variant per clean parent and requested cell."""

    parents, ordered_cells = _validated_inputs(
        read_manifest(input_manifest),
        benchmark_cells(config) if cells is None else cells,
        config,
    )
    output_records = [
        _materialize_one(
            parent=parent,
            cell=cell,
            output_root=output_root,
            config=config,
            overwrite=overwrite,
        )
        for parent in parents
        for cell in ordered_cells
    ]

    write_manifest(output_manifest, output_records)
    cell_counts = Counter(
        json.loads(record["transform_parameter"])["cell_id"] for record in output_records
    )
    report: dict[str, Any] = {
        "input_manifest": str(input_manifest.resolve()),
        "input_manifest_sha256": sha256_file(input_manifest),
        "output_root": str(output_root.resolve()),
        "output_manifest": str(output_manifest.resolve()),
        "output_manifest_sha256": sha256_file(output_manifest),
        "parent_count": len(parents),
        "image_count": len(output_records),
        "cell_counts": dict(sorted(cell_counts.items())),
        "label_counts": dict(
            sorted(Counter(record["label"] for record in output_records).items())
        ),
        "seed": config["runtime"]["seed"],
        "transform_version": config["transform_engine"]["version"],
        "preprocessing_version": config["transform_engine"]["preprocessing_version"],
        "encoder_version": f"Pillow-{pillow_version}",
    }
    write_json(report_path, report)
    return report
