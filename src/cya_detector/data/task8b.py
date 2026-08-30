"""Licensed native-camera and synthetic-source ingestion for Task 8B."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cya_detector.data.manifest import (
    MANIFEST_FIELDS,
    DatasetContractError,
    assign_duplicate_groups,
    scan_image,
    stable_source_id,
)


TASK8B_REQUIRED_COLUMNS = {
    "relative_path",
    "dataset_name",
    "source_subset",
    "label",
    "license_status",
    "processing_state",
    "device_id",
    "camera_make",
    "camera_model",
    "lens_model",
    "focal_length",
    "content_category",
    "generator_paradigm",
    "generator_name",
    "generator_checkpoint",
    "decoder_family",
}

PREMIER_LICENSE = "cc-by-sa-4.0"
GENIMAGE_LICENSE = "cc-by-nc-sa-4.0"
PREMIER_URL = "https://sites.google.com/unitn.it/premier/resources/datasets"
GENIMAGE_URL = "https://github.com/GenImage-Dataset/GenImage"
TINY_GENIMAGE_URL = "https://huggingface.co/datasets/TheKernel01/Tiny-GenImage"
PREMIER_SUBSETS = {"N1", "N2", "N3"}
PREMIER_PROCESSING = {"native_camera", "minimally_processed_camera"}
GENIMAGE_PROCESSING = {"native_generator_export"}
TASK8B_SPLITS = ("seed_train", "selection_val", "heldout_test")


def _safe_source_path(dataset_root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if not relative_path or relative.is_absolute() or ".." in relative.parts:
        raise DatasetContractError(f"Unsafe Task 8B relative path: {relative_path!r}")
    root = dataset_root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise DatasetContractError(
            f"Task 8B path escapes the dataset root: {relative_path!r}"
        ) from exc
    return candidate


def _canonical_dataset_name(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "premier": "premier",
        "premier_v3": "premier",
        "genimage": "genimage",
        "genimage_ai": "genimage",
        "tiny_genimage": "genimage",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise DatasetContractError(f"Unsupported Task 8B dataset: {value!r}") from exc


def _split_group_id(row: dict[str, str], dataset_name: str) -> str:
    if dataset_name == "premier":
        if not Path(row["relative_path"]).parts or Path(row["relative_path"]).parts[0] != "premier":
            raise DatasetContractError("PREMIER rows must live below task8b/premier")
        # A PREMIER device is the physical sensor unit. EXIF model strings can vary
        # within one device, so grouping on them would leak a device across splits.
        return f"authentic:device:{row['device_id'].strip().lower()}"
    return f"ai_generated:generator:{row['generator_name'].strip().lower()}"


def _validate_source_row(
    row: dict[str, str],
    *,
    allow_noncommercial_genimage: bool,
) -> tuple[str, str, str, str]:
    declared_dataset = row["dataset_name"].strip().lower().replace("-", "_")
    dataset_name = _canonical_dataset_name(declared_dataset)
    label = row["label"].strip().lower()
    license_status = row["license_status"].strip().lower()
    processing_state = row["processing_state"].strip().lower()

    if dataset_name == "premier":
        if label != "authentic":
            raise DatasetContractError("PREMIER N1/N2/N3 rows must be authentic")
        subset = row["source_subset"].strip().upper().removeprefix("PREMIER-")
        if subset not in PREMIER_SUBSETS:
            raise DatasetContractError("Only PREMIER native subsets N1, N2, and N3 are allowed")
        if license_status != PREMIER_LICENSE:
            raise DatasetContractError(f"PREMIER rows must declare {PREMIER_LICENSE}")
        if processing_state not in PREMIER_PROCESSING:
            raise DatasetContractError(
                "PREMIER rows must be native_camera or minimally_processed_camera"
            )
        if not row["device_id"].strip():
            raise DatasetContractError("PREMIER rows require a physical device_id")
        return dataset_name, label, subset, PREMIER_URL

    if not Path(row["relative_path"]).parts or Path(row["relative_path"]).parts[0] != "genimage_ai":
        raise DatasetContractError("GenImage rows must live below task8b/genimage_ai")
    if label != "ai_generated":
        raise DatasetContractError("Only GenImage AI outputs are allowed in Task 8B")
    if any(part.lower() == "nature" for part in Path(row["relative_path"]).parts):
        raise DatasetContractError("GenImage nature/ImageNet rows are excluded from Task 8B")
    if license_status != GENIMAGE_LICENSE:
        raise DatasetContractError(f"GenImage rows must declare {GENIMAGE_LICENSE}")
    if not allow_noncommercial_genimage:
        raise DatasetContractError("GenImage requires explicit non-commercial-use acceptance")
    if processing_state not in GENIMAGE_PROCESSING:
        raise DatasetContractError("GenImage rows must be native_generator_export")
    if not row["generator_name"].strip():
        raise DatasetContractError("GenImage rows require generator_name metadata")
    source_url = TINY_GENIMAGE_URL if declared_dataset == "tiny_genimage" else GENIMAGE_URL
    return dataset_name, label, row["source_subset"].strip(), source_url


def build_task8b_manifest(
    *,
    dataset_root: Path,
    inventory_path: Path,
    allow_noncommercial_genimage: bool,
    supported_extensions: set[str],
    minimum_images_per_device: int,
    perceptual_distance: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a fail-closed Task 8B manifest from a curated source inventory."""

    if minimum_images_per_device < 2:
        raise DatasetContractError("Task 8B requires at least two images per device")
    if not inventory_path.is_file():
        raise DatasetContractError(f"Missing Task 8B inventory: {inventory_path}")
    normalized_extensions = {value.lower() for value in supported_extensions}
    records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    with inventory_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = TASK8B_REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise DatasetContractError(
                f"Task 8B inventory is missing columns: {', '.join(sorted(missing))}"
            )
        for row_number, source_row in enumerate(reader, start=2):
            row = {key: (value or "").strip() for key, value in source_row.items()}
            relative_path = row["relative_path"]
            if relative_path in seen_paths:
                raise DatasetContractError(f"Duplicate inventory path: {relative_path}")
            seen_paths.add(relative_path)
            image_path = _safe_source_path(dataset_root, relative_path)
            if image_path.suffix.lower() not in normalized_extensions:
                raise DatasetContractError(
                    f"Unsupported Task 8B source format on row {row_number}: {image_path.suffix}"
                )
            if not image_path.is_file():
                raise DatasetContractError(f"Missing Task 8B image: {image_path}")
            dataset_name, label, subset, source_url = _validate_source_row(
                row,
                allow_noncommercial_genimage=allow_noncommercial_genimage,
            )
            scanned = scan_image(image_path)
            source_id = stable_source_id(dataset_name, relative_path)
            record = {field: "" for field in MANIFEST_FIELDS}
            record.update(
                {
                    "sample_id": f"{source_id}__source_original",
                    "source_id": source_id,
                    "source_path": str(image_path.resolve()),
                    "image_path": str(image_path.resolve()),
                    "image_view": "source_original",
                    "sha256": scanned.sha256,
                    "perceptual_hash": scanned.perceptual_hash,
                    "duplicate_is_primary": "true",
                    "review_required": "false",
                    "eligible_for_split": str(not scanned.corruption_error).lower(),
                    "label": label,
                    "dataset_name": dataset_name,
                    "license_status": row["license_status"].lower(),
                    "license_verified": "true",
                    "source_url": source_url,
                    "source_subset": subset,
                    "processing_state": row["processing_state"].lower(),
                    "physical_source_status": row["processing_state"].lower(),
                    "device_id": row["device_id"],
                    "camera_make": row["camera_make"],
                    "camera_model": row["camera_model"],
                    "lens_model": row["lens_model"],
                    "focal_length": row["focal_length"],
                    "content_category": row["content_category"],
                    "split_group_id": _split_group_id(row, dataset_name),
                    "transform": "clean",
                    "width": scanned.width,
                    "height": scanned.height,
                    "original_width": scanned.width,
                    "original_height": scanned.height,
                    "format": scanned.format,
                    "original_format": scanned.format,
                    "mode": scanned.mode,
                    "file_size": scanned.file_size,
                    "generator_paradigm": row["generator_paradigm"] or "unknown",
                    "generator_name": row["generator_name"] or "unknown",
                    "generator_checkpoint": row["generator_checkpoint"] or "unknown",
                    "decoder_family": row["decoder_family"] or "unknown",
                    "capture_source": row["device_id"] or "generator_export",
                    "c2pa_status": "not_scanned_task8b",
                    "c2pa_validation_state": "not_applicable_to_import_audit",
                    "corruption_error": scanned.corruption_error,
                }
            )
            records.append(record)

    if not records:
        raise DatasetContractError("Task 8B inventory contains no rows")

    device_counts = Counter(
        record["device_id"] for record in records if record["dataset_name"] == "premier"
    )
    undersized_devices = {
        device_id for device_id, count in device_counts.items() if count < minimum_images_per_device
    }
    for record in records:
        if record["device_id"] in undersized_devices:
            record["eligible_for_split"] = "false"

    duplicate_stats = assign_duplicate_groups(
        records,
        max_hamming_distance=perceptual_distance,
    )
    label_counts = Counter(record["label"] for record in records)
    report = {
        "dataset_root": str(dataset_root.resolve()),
        "inventory": str(inventory_path.resolve()),
        "row_count": len(records),
        "label_counts": dict(sorted(label_counts.items())),
        "dataset_counts": dict(
            sorted(Counter(record["dataset_name"] for record in records).items())
        ),
        "device_count": len(device_counts),
        "generator_count": len(
            {
                record["generator_name"]
                for record in records
                if record["dataset_name"] == "genimage"
            }
        ),
        "undersized_devices": sorted(undersized_devices),
        "minimum_images_per_device": minimum_images_per_device,
        "eligible_count": sum(record["eligible_for_split"] == "true" for record in records),
        "license_counts": dict(
            sorted(Counter(record["license_status"] for record in records).items())
        ),
        "noncommercial_genimage_accepted": allow_noncommercial_genimage,
        "raw_or_heic_status": "preserve_but_do_not_inventory_until_a_decoder_is_pinned",
        "chromatic_aberration_metadata": {
            "lens_model_known": sum(
                record["lens_model"].lower() not in {"", "unknown"}
                for record in records
                if record["dataset_name"] == "premier"
            ),
            "focal_length_known": sum(
                record["focal_length"].lower() not in {"", "unknown"}
                for record in records
                if record["dataset_name"] == "premier"
            ),
            "status": "audit_before_enabling_ca",
        },
        **duplicate_stats,
    }
    return records, report


def _allocate_group_counts(total: int, fractions: dict[str, float]) -> dict[str, int]:
    if total < len(fractions):
        raise DatasetContractError(
            f"Task 8B requires at least {len(fractions)} groups per label; found {total}"
        )
    raw = {name: total * value for name, value in fractions.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(fractions, key=lambda name: (raw[name] - counts[name], name), reverse=True)
    for name in order[:remaining]:
        counts[name] += 1
    for name in fractions:
        if counts[name] == 0:
            donor = max(counts, key=counts.get)
            if counts[donor] <= 1:
                raise DatasetContractError("Task 8B cannot allocate non-empty held-out splits")
            counts[donor] -= 1
            counts[name] += 1
    return counts


def assign_task8b_splits(
    records: list[dict[str, str]],
    *,
    seed: int,
    fractions: dict[str, float],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Assign complete physical devices and generator families without leakage."""

    if tuple(fractions) != TASK8B_SPLITS or abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise DatasetContractError(f"Task 8B split order must be {TASK8B_SPLITS}")

    duplicate_groups: dict[str, set[str]] = defaultdict(set)
    for record in records:
        duplicate_group = record.get("duplicate_group_id")
        if duplicate_group:
            duplicate_groups[duplicate_group].add(record["split_group_id"])
    leaking_duplicates = {
        group for group, split_groups in duplicate_groups.items() if len(split_groups) > 1
    }
    for record in records:
        if record.get("duplicate_group_id") in leaking_duplicates:
            record["eligible_for_split"] = "false"
            record["review_required"] = "true"
            record["split"] = "excluded_review"

    groups_by_label: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.get("eligible_for_split") == "true":
            groups_by_label[record["label"]].add(record["split_group_id"])

    assignments: dict[str, str] = {}
    allocation: dict[str, dict[str, int]] = {}
    for label in ("authentic", "ai_generated"):
        groups = sorted(
            groups_by_label[label],
            key=lambda group: hashlib.sha256(f"{seed}:{label}:{group}".encode()).hexdigest(),
        )
        counts = _allocate_group_counts(len(groups), fractions)
        allocation[label] = counts
        cursor = 0
        for split_name in fractions:
            for group in groups[cursor : cursor + counts[split_name]]:
                assignments[group] = split_name
            cursor += counts[split_name]

    for record in records:
        if record.get("split") == "excluded_review":
            continue
        if record["split_group_id"] in assignments:
            record["split"] = assignments[record["split_group_id"]]
        else:
            record["split"] = "excluded"

    overlaps = 0
    seen: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record["split"] in fractions:
            seen[record["split_group_id"]].add(record["split"])
    overlaps = sum(len(splits) > 1 for splits in seen.values())
    report = {
        "seed": seed,
        "fractions": fractions,
        "group_allocation": allocation,
        "cross_group_duplicate_count": len(leaking_duplicates),
        "split_group_overlap_count": overlaps,
        "counts": dict(
            sorted(
                Counter(
                    f"{record['split']}:{record['label']}"
                    for record in records
                    if record["split"] in fractions
                ).items()
            )
        ),
        "final_test_read": False,
    }
    if overlaps:
        raise DatasetContractError("Task 8B split-group leakage detected")
    return records, report
