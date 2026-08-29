"""Create label-independent canonical matched-clean JPEG views."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, __version__ as pillow_version

from cya_detector.data.manifest import (
    MANIFEST_FIELDS,
    read_manifest,
    sha256_file,
    write_json,
    write_manifest,
)


SUPPORTED_POLICIES = {"fixed_q96", "uniform_q95_q100"}
SAFE_C2PA_STATUSES = {"no_manifest", "manifest_present"}


class MatchedViewError(ValueError):
    """Raised when canonical matched views cannot be created safely."""


def quality_for(policy: str, *, source_id: str, seed: int) -> int:
    if policy == "fixed_q96":
        return 96
    if policy == "uniform_q95_q100":
        digest = hashlib.sha256(f"{seed}:{source_id}:jpeg-quality".encode()).digest()
        return 95 + int.from_bytes(digest[:4], "big") % 6
    raise MatchedViewError(f"Unsupported quality policy: {policy}")


def jpeg_quantization_hash(path: Path) -> str:
    with Image.open(path) as image:
        quantization = image.quantization or {}
    serialized = json.dumps(quantization, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _eligible_rows(
    records: list[dict[str, str]],
    *,
    allow_unchecked_c2pa: bool,
    limit_per_label: int | None,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    for record in sorted(records, key=lambda row: (row["label"], row["source_id"])):
        if record.get("eligible_for_split") != "true":
            continue
        if record.get("duplicate_is_primary") != "true":
            continue
        if record.get("review_required") == "true" or record.get("corruption_error"):
            continue
        status = record.get("c2pa_status", "not_checked")
        if status not in SAFE_C2PA_STATUSES and not allow_unchecked_c2pa:
            raise MatchedViewError(
                f"C2PA was not successfully checked for {record['source_id']}: {status}"
            )
        if limit_per_label is not None and counts[record["label"]] >= limit_per_label:
            continue
        counts[record["label"]] += 1
        selected.append(record)
    return selected


def build_matched_clean(
    *,
    source_manifest: Path,
    output_root: Path,
    output_manifest: Path,
    report_path: Path,
    policy: str,
    seed: int = 42,
    limit_per_label: int | None = None,
    allow_unchecked_c2pa: bool = False,
) -> dict[str, Any]:
    if policy not in SUPPORTED_POLICIES:
        raise MatchedViewError(f"Unsupported quality policy: {policy}")

    source_records = read_manifest(source_manifest)
    selected = _eligible_rows(
        source_records,
        allow_unchecked_c2pa=allow_unchecked_c2pa,
        limit_per_label=limit_per_label,
    )
    output_records: list[dict[str, Any]] = []
    quality_counts: Counter[int] = Counter()

    for source in selected:
        source_path = Path(source["source_path"])
        quality = quality_for(policy, source_id=source["source_id"], seed=seed)
        output_path = output_root / policy / source["label"] / f"{source['source_id']}.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        temporary_path = output_path.with_name(f"{output_path.stem}.tmp.jpg")
        with Image.open(source_path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            rgb.save(
                temporary_path,
                format="JPEG",
                quality=quality,
                subsampling=0,
                optimize=False,
                progressive=False,
                exif=b"",
            )
        temporary_path.replace(output_path)

        with Image.open(output_path) as image:
            width, height = image.size
            image_format = image.format or "JPEG"
            mode = image.mode

        record = {field: "" for field in MANIFEST_FIELDS}
        record.update(source)
        record.update(
            {
                "sample_id": f"{source['source_id']}__matched_clean__{policy}",
                "parent_id": source["sample_id"],
                "image_path": str(output_path.resolve()),
                "clean_image_path": str(output_path.resolve()),
                "image_view": "matched_clean",
                "sha256": sha256_file(output_path),
                "perceptual_hash": "",
                "transform": "clean",
                "transform_parameter": "",
                "transform_seed": seed,
                "width": width,
                "height": height,
                "format": image_format,
                "mode": mode,
                "file_size": output_path.stat().st_size,
                "normalization_codec": "JPEG",
                "normalization_quality": quality,
                "encoder_version": f"Pillow-{pillow_version}",
                "chroma_subsampling": "4:4:4",
                "quantization_table_hash": jpeg_quantization_hash(output_path),
                "output_storage_format": "JPEG",
                "corruption_error": "",
            }
        )
        output_records.append(record)
        quality_counts[quality] += 1

    write_manifest(output_manifest, output_records)
    report = {
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest),
        "output_manifest": str(output_manifest.resolve()),
        "output_manifest_sha256": sha256_file(output_manifest),
        "policy": policy,
        "seed": seed,
        "limit_per_label": limit_per_label,
        "image_count": len(output_records),
        "label_counts": dict(sorted(Counter(row["label"] for row in output_records).items())),
        "quality_counts": {str(key): value for key, value in sorted(quality_counts.items())},
        "encoder_version": f"Pillow-{pillow_version}",
        "subsampling": "4:4:4",
        "metadata_policy": "strip_exif",
        "resize_applied": False,
    }
    write_json(report_path, report)
    return report
