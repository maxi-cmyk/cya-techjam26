"""Deterministic label-independent matched views for Task 8B."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, __version__ as pillow_version

from cya_detector.data.manifest import (
    MANIFEST_FIELDS,
    DatasetContractError,
    assign_duplicate_groups,
    read_manifest,
    scan_image,
    write_json,
    write_manifest,
)


MATCHED_VIEW = "task8b_matched_tiff"
MATCHED_VERSION = "task8b-matched-tiff-v1"
TASK8B_SPLITS = {"seed_train", "selection_val", "heldout_test"}


def _crop_origin(*, source_id: str, seed: int, width: int, height: int, size: int) -> tuple[int, int]:
    digest = hashlib.sha256(f"{seed}:{source_id}:{MATCHED_VERSION}".encode()).digest()
    x_range = width - size + 1
    y_range = height - size + 1
    left = int.from_bytes(digest[:8], "big") % x_range
    top = int.from_bytes(digest[8:16], "big") % y_range
    return left, top


def build_task8b_matched_views(
    *,
    source_manifest: Path,
    output_root: Path,
    output_manifest: Path,
    report_path: Path,
    size: int,
    seed: int,
    perceptual_distance: int = 4,
    minimum_rgb_std: float = 2.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create equal-size uncompressed RGB TIFF crops without resizing any source."""

    if size < 64:
        raise DatasetContractError("Task 8B matched crop size must be at least 64 pixels")
    if output_manifest.exists() and not overwrite:
        raise DatasetContractError(
            f"Matched manifest already exists: {output_manifest}; pass overwrite after review"
        )
    source_rows = [
        row
        for row in read_manifest(source_manifest)
        if row.get("eligible_for_split") == "true" and row.get("split") in TASK8B_SPLITS
    ]
    if not source_rows:
        raise DatasetContractError("Task 8B source manifest has no eligible split rows")

    records: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    crop_origins: Counter[str] = Counter()
    for source in sorted(source_rows, key=lambda row: row["sample_id"]):
        source_path = Path(source["image_path"])
        try:
            with Image.open(source_path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                width, height = image.size
                if min(width, height) < size:
                    excluded.append(
                        {
                            "sample_id": source["sample_id"],
                            "reason": f"minimum_dimension_below_{size}",
                            "width": str(width),
                            "height": str(height),
                        }
                    )
                    continue
                left, top = _crop_origin(
                    source_id=source["source_id"],
                    seed=seed,
                    width=width,
                    height=height,
                    size=size,
                )
                crop = image.crop((left, top, left + size, top + size))
        except Exception as exc:
            excluded.append(
                {
                    "sample_id": source.get("sample_id", ""),
                    "reason": f"decode_error:{type(exc).__name__}:{exc}",
                    "width": "",
                    "height": "",
                }
            )
            continue

        pixels = np.asarray(crop, dtype=np.uint8).copy()
        pixel_std = float(pixels.astype(np.float32).std())
        if pixel_std < minimum_rgb_std:
            excluded.append(
                {
                    "sample_id": source["sample_id"],
                    "reason": "low_information_crop",
                    "width": str(width),
                    "height": str(height),
                }
            )
            continue

        # Reconstruct from pixels so no source EXIF/ICC/TIFF encoder hints can
        # survive into the normalized container.
        crop = Image.fromarray(pixels, mode="RGB")

        sample_id = f"{source['source_id']}__{MATCHED_VIEW}"
        target = output_root / source["split"] / source["label"] / f"{sample_id}.tiff"
        if target.exists() and not overwrite:
            raise DatasetContractError(f"Matched view already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp.tiff")
        crop.save(temporary, format="TIFF", compression="raw")
        temporary.replace(target)
        scanned = scan_image(target)
        if scanned.corruption_error:
            raise DatasetContractError(f"Generated matched view is corrupt: {target}")

        record = {field: source.get(field, "") for field in MANIFEST_FIELDS}
        record.update(
            {
                "sample_id": sample_id,
                "parent_id": source["sample_id"],
                "parent_width": width,
                "parent_height": height,
                "parent_mode": "RGB",
                "parent_format": source.get("format", ""),
                "source_path": source.get("source_path") or source["image_path"],
                "image_path": str(target.resolve()),
                "clean_image_path": str(target.resolve()),
                "image_view": MATCHED_VIEW,
                "sha256": scanned.sha256,
                "perceptual_hash": scanned.perceptual_hash,
                "duplicate_group_id": "",
                "duplicate_is_primary": "true",
                "review_required": "false",
                "eligible_for_split": "true",
                "processing_state": "matched_native_crop",
                "transform": "clean",
                "transform_parameter": f"crop_{size}",
                "transform_seed": seed,
                "width": scanned.width,
                "height": scanned.height,
                "format": scanned.format,
                "mode": scanned.mode,
                "file_size": scanned.file_size,
                "normalization_codec": "TIFF-raw-RGB",
                "normalization_quality": "lossless",
                "encoder_version": f"Pillow-{pillow_version}",
                "output_storage_format": "TIFF",
                "parent_sha256": source["sha256"],
                "realized_parameters": f"left={left};top={top};size={size};resize=false",
                "transform_version": MATCHED_VERSION,
                "preprocessing_version": MATCHED_VERSION,
            }
        )
        records.append(record)
        crop_origins[f"{left},{top}"] += 1

    if not records:
        raise DatasetContractError("No Task 8B matched views were materialized")
    duplicate_stats = assign_duplicate_groups(
        records, max_hamming_distance=perceptual_distance
    )
    duplicate_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record["duplicate_group_id"]:
            duplicate_splits[record["duplicate_group_id"]].add(record["split"])
    cross_split_duplicates = sorted(
        group for group, splits in duplicate_splits.items() if len(splits) > 1
    )
    if duplicate_stats["cross_label_duplicate_groups"]:
        raise DatasetContractError(
            "Matched crop duplicate audit failed: "
            f"cross_label={duplicate_stats['cross_label_duplicate_groups']}"
        )

    unresolved_duplicate_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record["eligible_for_split"] == "true" and record["duplicate_group_id"]:
            unresolved_duplicate_splits[record["duplicate_group_id"]].add(record["split"])
    unresolved_cross_split = sorted(
        group for group, splits in unresolved_duplicate_splits.items() if len(splits) > 1
    )
    if unresolved_cross_split:
        raise DatasetContractError(
            f"Matched duplicate primary selection left {len(unresolved_cross_split)} split overlaps"
        )

    stale_removed: list[str] = []
    if overwrite:
        referenced = {Path(record["image_path"]).resolve() for record in records}
        for path in output_root.rglob("*.tiff"):
            if path.resolve() not in referenced:
                path.unlink()
                stale_removed.append(str(path.resolve()))
    write_manifest(output_manifest, records)
    file_sizes = sorted({int(record["file_size"]) for record in records})
    report = {
        "source_manifest": str(source_manifest.resolve()),
        "output_manifest": str(output_manifest.resolve()),
        "output_root": str(output_root.resolve()),
        "version": MATCHED_VERSION,
        "seed": seed,
        "crop_size": size,
        "minimum_rgb_std": minimum_rgb_std,
        "resize_applied": False,
        "mode": "RGB",
        "codec": "uncompressed_tiff",
        "metadata_policy": "stripped",
        "source_row_count": len(source_rows),
        "materialized_count": len(records),
        "eligible_count": sum(row["eligible_for_split"] == "true" for row in records),
        "excluded_count": len(excluded),
        "excluded": excluded,
        "stale_generated_files_removed": stale_removed,
        "label_counts": dict(sorted(Counter(row["label"] for row in records).items())),
        "split_label_counts": dict(
            sorted(Counter(f"{row['split']}:{row['label']}" for row in records).items())
        ),
        "output_file_sizes": file_sizes,
        "all_output_file_sizes_equal": len(file_sizes) == 1,
        "unique_crop_origin_count": len(crop_origins),
        "cross_split_duplicate_groups_resolved": cross_split_duplicates,
        "unresolved_cross_split_duplicate_groups": unresolved_cross_split,
        **duplicate_stats,
    }
    write_json(report_path, report)
    return report
