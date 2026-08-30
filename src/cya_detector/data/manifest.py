"""Build an immutable-source manifest from the SID dataset handoff."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

MANIFEST_FIELDS = [
    "sample_id",
    "source_id",
    "parent_id",
    "parent_width",
    "parent_height",
    "parent_mode",
    "parent_format",
    "source_path",
    "image_path",
    "clean_image_path",
    "image_view",
    "sha256",
    "perceptual_hash",
    "duplicate_group_id",
    "duplicate_is_primary",
    "review_required",
    "eligible_for_split",
    "label",
    "split",
    "dataset_name",
    "license_status",
    "transform",
    "transform_parameter",
    "transform_seed",
    "width",
    "height",
    "format",
    "mode",
    "file_size",
    "normalization_codec",
    "normalization_quality",
    "encoder_version",
    "chroma_subsampling",
    "original_format",
    "estimated_original_quality",
    "quantization_table_hash",
    "resize_scale",
    "down_interpolation",
    "up_interpolation",
    "resize_library",
    "resize_library_version",
    "antialias",
    "dimension_rounding",
    "intermediate_width",
    "intermediate_height",
    "original_width",
    "original_height",
    "output_storage_format",
    "generator_paradigm",
    "generator_name",
    "generator_checkpoint",
    "decoder_family",
    "tokenizer_family",
    "upsampling_factor",
    "capture_source",
    "c2pa_status",
    "c2pa_validation_state",
    "corruption_error",
    "parent_sha256",
    "realized_parameters",
    "transform_version",
    "preprocessing_version",
]

BINARY_LABEL_ALIASES = {
    "0": "authentic",
    "real": "authentic",
    "authentic": "authentic",
    "1": "ai_generated",
    "synthetic": "ai_generated",
    "fake": "ai_generated",
    "full_synthetic": "ai_generated",
    "fully_synthetic": "ai_generated",
    "ai_generated": "ai_generated",
}

EXCLUDED_LABEL_ALIASES = {
    "2",
    "tampered",
    "mixed",
    "ai_edited",
    "edited",
    "ambiguous",
}


class DatasetContractError(ValueError):
    """Raised when source data cannot be mapped safely to the binary contract."""


@dataclass(frozen=True)
class ScannedImage:
    sha256: str
    perceptual_hash: str
    width: int
    height: int
    format: str
    mode: str
    file_size: int
    corruption_error: str = ""


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(image: Image.Image) -> str:
    grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(grayscale.getdata())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    red, green, blue = image.convert("RGB").resize((1, 1), Image.Resampling.BOX).getpixel((0, 0))
    color_signature = ((red >> 4) << 8) | ((green >> 4) << 4) | (blue >> 4)
    return f"{value:016x}{color_signature:03x}"


def scan_image(path: Path) -> ScannedImage:
    file_digest = sha256_file(path)
    file_size = path.stat().st_size
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format or path.suffix.lstrip(".").upper()
            mode = image.mode
            perceptual_hash = difference_hash(image)
        return ScannedImage(
            sha256=file_digest,
            perceptual_hash=perceptual_hash,
            width=width,
            height=height,
            format=image_format,
            mode=mode,
            file_size=file_size,
        )
    except Exception as exc:
        return ScannedImage(
            sha256=file_digest,
            perceptual_hash="",
            width=0,
            height=0,
            format=path.suffix.lstrip(".").upper(),
            mode="",
            file_size=file_size,
            corruption_error=f"{type(exc).__name__}: {exc}",
        )


def normalize_label(row: dict[str, str]) -> str | None:
    candidates = []
    for field in ("label_name", "label"):
        value = row.get(field, "").strip().lower().replace("-", "_").replace(" ", "_")
        if value:
            candidates.append(value)

    mapped = {BINARY_LABEL_ALIASES[value] for value in candidates if value in BINARY_LABEL_ALIASES}
    excluded = any(value in EXCLUDED_LABEL_ALIASES for value in candidates)
    unknown = [
        value
        for value in candidates
        if value not in BINARY_LABEL_ALIASES and value not in EXCLUDED_LABEL_ALIASES
    ]

    if len(mapped) > 1 or (mapped and excluded):
        raise DatasetContractError(f"Conflicting labels in row: {row}")
    if unknown and not mapped and not excluded:
        raise DatasetContractError(f"Unknown label values {unknown!r} in row: {row}")
    if mapped:
        return mapped.pop()
    if excluded:
        return None
    raise DatasetContractError(f"Row has no usable label: {row}")


def inspect_c2pa(path: Path, *, enabled: bool) -> tuple[str, str]:
    if not enabled:
        return "not_checked", ""

    try:
        from c2pa import Context, Reader
    except ImportError:
        return "dependency_missing", ""

    try:
        context = Context.from_dict(
            {"verify": {"remote_manifest_fetch": False, "ocsp_fetch": False}}
        )
        reader = Reader.try_create(path, context=context)
        if reader is None:
            return "no_manifest", ""
        with reader:
            validation_state = str(reader.get_validation_state())
            json.loads(reader.json())
        return "manifest_present", validation_state
    except Exception as exc:
        return "scan_error", f"{type(exc).__name__}: {exc}"


def stable_source_id(dataset_name: str, filename: str) -> str:
    digest = hashlib.sha256(f"{dataset_name}:{filename}".encode()).hexdigest()[:20]
    return f"{dataset_name}_{digest}"


def _hash_bands(value: int, *, bit_count: int, band_count: int) -> Iterable[tuple[int, int]]:
    base_width, wider_bands = divmod(bit_count, band_count)
    widths = [base_width + int(index < wider_bands) for index in range(band_count)]
    shift = 0
    for band_index, width in enumerate(widths):
        mask = (1 << width) - 1
        yield band_index, (value >> shift) & mask
        shift += width


def assign_duplicate_groups(
    records: list[dict[str, Any]], *, max_hamming_distance: int = 4
) -> dict[str, int]:
    """Group exact and near duplicates without an all-pairs comparison."""

    disjoint = DisjointSet(len(records))
    exact_representatives: dict[str, int] = {}
    hash_representatives: dict[str, int] = {}

    for index, record in enumerate(records):
        exact = record["sha256"]
        if exact in exact_representatives:
            disjoint.union(index, exact_representatives[exact])
        else:
            exact_representatives[exact] = index

        perceptual = record["perceptual_hash"]
        if not perceptual:
            continue
        if perceptual in hash_representatives:
            disjoint.union(index, hash_representatives[perceptual])
        else:
            hash_representatives[perceptual] = index

    band_buckets: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    compared: set[tuple[int, int]] = set()
    for perceptual, index in sorted(hash_representatives.items()):
        value = int(perceptual, 16)
        for band in _hash_bands(
            value,
            bit_count=len(perceptual) * 4,
            band_count=max_hamming_distance + 1,
        ):
            for other_value, other_index in band_buckets[band]:
                pair = (min(index, other_index), max(index, other_index))
                if pair in compared:
                    continue
                compared.add(pair)
                if (value ^ other_value).bit_count() <= max_hamming_distance:
                    disjoint.union(index, other_index)
            band_buckets[band].append((value, index))

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        groups[disjoint.find(index)].append(index)

    duplicate_groups = 0
    cross_label_groups = 0
    for indices in groups.values():
        if len(indices) == 1:
            records[indices[0]]["duplicate_group_id"] = ""
            records[indices[0]]["duplicate_is_primary"] = "true"
            continue

        duplicate_groups += 1
        source_ids = sorted(records[index]["source_id"] for index in indices)
        group_digest = hashlib.sha256("|".join(source_ids).encode()).hexdigest()[:16]
        group_id = f"duplicate_{group_digest}"
        labels = {records[index]["label"] for index in indices}
        cross_label = len(labels) > 1
        if cross_label:
            cross_label_groups += 1

        primary = min(indices, key=lambda index: records[index]["source_id"])
        for index in indices:
            records[index]["duplicate_group_id"] = group_id
            records[index]["duplicate_is_primary"] = str(index == primary).lower()
            if cross_label:
                records[index]["review_required"] = "true"
                records[index]["eligible_for_split"] = "false"
            elif index != primary:
                records[index]["eligible_for_split"] = "false"

    return {
        "duplicate_groups": duplicate_groups,
        "cross_label_duplicate_groups": cross_label_groups,
    }


def build_source_manifest(
    *,
    dataset_root: Path,
    dataset_name: str = "sid_set",
    license_status: str = "cc-by-4.0",
    check_c2pa: bool = True,
    perceptual_distance: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels_path = dataset_root / "labels.csv"
    images_root = dataset_root / "images"
    if not labels_path.is_file():
        raise DatasetContractError(f"Missing labels file: {labels_path}")
    if not images_root.is_dir():
        raise DatasetContractError(f"Missing images directory: {images_root}")

    records: list[dict[str, Any]] = []
    excluded_counts: Counter[str] = Counter()
    listed_filenames: set[str] = set()

    with labels_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "filename" not in reader.fieldnames:
            raise DatasetContractError("labels.csv must contain a filename column")

        for row_number, source_row in enumerate(reader, start=2):
            filename = source_row.get("filename", "").strip()
            if not filename or Path(filename).name != filename:
                raise DatasetContractError(f"Unsafe filename on CSV row {row_number}: {filename!r}")
            if filename in listed_filenames:
                raise DatasetContractError(f"Duplicate filename in labels.csv: {filename}")
            listed_filenames.add(filename)

            label = normalize_label(source_row)
            if label is None:
                excluded_key = source_row.get("label_name") or source_row.get("label") or "unknown"
                excluded_counts[excluded_key] += 1
                continue

            image_path = images_root / filename
            if not image_path.is_file():
                raise DatasetContractError(f"Image listed in CSV is missing: {image_path}")

            scanned = scan_image(image_path)
            c2pa_status, c2pa_validation = inspect_c2pa(image_path, enabled=check_c2pa)
            source_id = stable_source_id(dataset_name, filename)
            eligible = not scanned.corruption_error and c2pa_status != "scan_error"
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
                    "eligible_for_split": str(eligible).lower(),
                    "label": label,
                    "dataset_name": dataset_name,
                    "license_status": license_status,
                    "transform": "clean",
                    "width": scanned.width,
                    "height": scanned.height,
                    "original_width": scanned.width,
                    "original_height": scanned.height,
                    "format": scanned.format,
                    "original_format": scanned.format,
                    "mode": scanned.mode,
                    "file_size": scanned.file_size,
                    "generator_paradigm": source_row.get("generator_paradigm", "unknown")
                    or "unknown",
                    "generator_name": source_row.get("generator_name", "unknown") or "unknown",
                    "generator_checkpoint": source_row.get("generator_checkpoint", "unknown")
                    or "unknown",
                    "decoder_family": source_row.get("decoder_family", "unknown") or "unknown",
                    "tokenizer_family": source_row.get("tokenizer_family", "unknown") or "unknown",
                    "upsampling_factor": source_row.get("upsampling_factor", "unknown") or "unknown",
                    "capture_source": source_row.get("capture_source", "unknown") or "unknown",
                    "c2pa_status": c2pa_status,
                    "c2pa_validation_state": c2pa_validation,
                    "corruption_error": scanned.corruption_error,
                }
            )
            records.append(record)

    image_filenames = {path.name for path in images_root.iterdir() if path.is_file()}
    unlisted_images = sorted(image_filenames - listed_filenames)
    duplicate_stats = assign_duplicate_groups(
        records, max_hamming_distance=perceptual_distance
    )

    report = {
        "dataset_root": str(dataset_root.resolve()),
        "dataset_name": dataset_name,
        "csv_rows": len(listed_filenames),
        "included_binary_rows": len(records),
        "excluded_label_counts": dict(sorted(excluded_counts.items())),
        "unlisted_image_count": len(unlisted_images),
        "unlisted_images_preview": unlisted_images[:20],
        "label_counts": dict(sorted(Counter(record["label"] for record in records).items())),
        "corrupt_count": sum(bool(record["corruption_error"]) for record in records),
        "c2pa_status_counts": dict(
            sorted(Counter(record["c2pa_status"] for record in records).items())
        ),
        "eligible_primary_count": sum(
            record["eligible_for_split"] == "true" for record in records
        ),
        "perceptual_hamming_distance": perceptual_distance,
        **duplicate_stats,
    }
    return records, report


def write_manifest(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)
    return path


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
