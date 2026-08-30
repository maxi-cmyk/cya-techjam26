"""Prepare a reviewed Task 8B source inventory from extracted local datasets."""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image

from cya_detector.data.manifest import write_json
from cya_detector.data.task8b import TASK8B_REQUIRED_COLUMNS


GENERATOR_METADATA = {
    "adm": ("diffusion", "ADM", "pixel_space"),
    "biggan": ("gan", "BigGAN", "convolutional_generator"),
    "glide": ("diffusion", "GLIDE", "pixel_space"),
    "midjourney": ("unknown", "Midjourney", "unknown"),
    "stablediffusionv14": ("latent_diffusion", "Stable Diffusion V1.4", "VAE"),
    "stablediffusionv15": ("latent_diffusion", "Stable Diffusion V1.5", "VAE"),
    "vqdm": ("diffusion", "VQDM", "VQ"),
    "wukong": ("latent_diffusion", "Wukong", "VAE"),
}
DEVICE_PATTERN = re.compile(r"^([A-Z]\d{2,})(?:[_-](.+))?$", re.IGNORECASE)
SUBSET_PATTERN = re.compile(r"^(?:PREMIER[-_])?(N[123])$", re.IGNORECASE)


class Task8BPreparationError(ValueError):
    """Raised when extracted sources cannot produce a safe inventory."""


def _slug(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _stable_order(rows: list[dict[str, str]], salt: str) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{salt}:{row['relative_path']}".encode()
        ).hexdigest(),
    )


def _read_exif(path: Path) -> dict[str, str]:
    try:
        with Image.open(path) as image:
            raw = image.getexif()
        named = {ExifTags.TAGS.get(key, str(key)): value for key, value in raw.items()}
    except Exception:
        return {}

    def clean(name: str) -> str:
        value = named.get(name, "")
        if isinstance(value, bytes):
            value = value.decode(errors="replace")
        return str(value).strip().strip("\x00")

    return {
        "camera_make": clean("Make"),
        "camera_model": clean("Model"),
        "lens_model": clean("LensModel"),
        "focal_length": clean("FocalLength"),
    }


def _premier_identity(relative: Path) -> tuple[str, str, str, str] | None:
    subset = ""
    device_id = ""
    device_description = ""
    for part in relative.parts:
        subset_match = SUBSET_PATTERN.match(part)
        if subset_match:
            subset = subset_match.group(1).upper()
        device_match = DEVICE_PATTERN.match(part)
        if device_match:
            device_id = device_match.group(1).upper()
            device_description = (device_match.group(2) or "").replace("_", " ")
    if not device_id:
        filename_match = re.match(r"^([A-Z]\d{2,})[_-]", relative.name, re.IGNORECASE)
        if filename_match:
            device_id = filename_match.group(1).upper()
    if not subset or not device_id:
        return None
    description_parts = device_description.split(maxsplit=1)
    camera_make = description_parts[0] if description_parts else ""
    camera_model = description_parts[1] if len(description_parts) > 1 else device_description
    return subset, device_id, camera_make, camera_model


def _content_category(relative: Path) -> str:
    lowered = {part.lower() for part in relative.parts}
    if any("calibration" in part or "checkerboard" in part for part in lowered):
        return "calibration_target"
    if any("flat" in part for part in lowered):
        return "flat"
    if "indoor" in lowered:
        return "indoor"
    if "outdoor" in lowered:
        return "outdoor"
    return "natural"


def _premier_rows(
    task8b_root: Path,
    supported_extensions: set[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    source_root = task8b_root / "premier"
    if not source_root.is_dir():
        raise Task8BPreparationError(f"Missing extracted PREMIER directory: {source_root}")
    rows: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in supported_extensions:
            continue
        relative = path.relative_to(task8b_root)
        identity = _premier_identity(relative)
        if identity is None:
            rejected.append(
                {
                    "relative_path": relative.as_posix(),
                    "reason": "could_not_infer_both_premier_subset_and_device_id",
                }
            )
            continue
        subset, device_id, inferred_make, inferred_model = identity
        exif = _read_exif(path)
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "dataset_name": "premier",
                "source_subset": subset,
                "label": "authentic",
                "license_status": "cc-by-sa-4.0",
                "processing_state": "native_camera",
                "device_id": device_id,
                "camera_make": exif.get("camera_make") or inferred_make or "unknown",
                "camera_model": exif.get("camera_model") or inferred_model or "unknown",
                "lens_model": exif.get("lens_model") or "unknown",
                "focal_length": exif.get("focal_length") or "unknown",
                "content_category": _content_category(relative),
                "generator_paradigm": "",
                "generator_name": "",
                "generator_checkpoint": "",
                "decoder_family": "",
            }
        )
    return rows, rejected


def _generator_metadata(folder_name: str) -> tuple[str, str, str]:
    normalized = _slug(folder_name)
    for key, metadata in GENERATOR_METADATA.items():
        if key in normalized or normalized in key:
            return metadata
    return "unknown", folder_name, "unknown"


def _genimage_rows(
    task8b_root: Path,
    supported_extensions: set[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    source_root = task8b_root / "genimage_ai"
    if not source_root.is_dir():
        raise Task8BPreparationError(f"Missing extracted GenImage directory: {source_root}")
    rows: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in supported_extensions:
            continue
        relative = path.relative_to(task8b_root)
        parts_lower = [part.lower() for part in relative.parts]
        if "nature" in parts_lower:
            rejected.append(
                {"relative_path": relative.as_posix(), "reason": "genimage_nature_excluded"}
            )
            continue
        if "ai" not in parts_lower:
            rejected.append(
                {"relative_path": relative.as_posix(), "reason": "genimage_path_has_no_ai_branch"}
            )
            continue
        if len(relative.parts) < 3:
            rejected.append(
                {"relative_path": relative.as_posix(), "reason": "missing_generator_folder"}
            )
            continue
        generator_folder = relative.parts[1]
        paradigm, generator_name, decoder = _generator_metadata(generator_folder)
        ai_index = parts_lower.index("ai")
        category = relative.parts[ai_index + 1] if ai_index + 1 < len(relative.parts) - 1 else "unknown"
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "dataset_name": (
                    "tiny_genimage"
                    if path.name.startswith("tiny_genimage_")
                    else "genimage"
                ),
                "source_subset": generator_name,
                "label": "ai_generated",
                "license_status": "cc-by-nc-sa-4.0",
                "processing_state": "native_generator_export",
                "device_id": "",
                "camera_make": "",
                "camera_model": "",
                "lens_model": "",
                "focal_length": "",
                "content_category": category,
                "generator_paradigm": paradigm,
                "generator_name": generator_name,
                "generator_checkpoint": generator_folder,
                "decoder_family": decoder,
            }
        )
    return rows, rejected


def _cap_generators(
    rows: list[dict[str, str]],
    max_per_generator: int,
    seed: int,
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["generator_name"]].append(row)
    selected: list[dict[str, str]] = []
    for generator, generator_rows in sorted(grouped.items()):
        selected.extend(
            _stable_order(generator_rows, f"{seed}:generator:{generator}")[
                :max_per_generator
            ]
        )
    return selected


def _balanced_authentic_sample(
    rows: list[dict[str, str]],
    target: int,
    seed: int,
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["device_id"]].append(row)
    queues = {
        device: _stable_order(device_rows, f"{seed}:device:{device}")
        for device, device_rows in grouped.items()
    }
    selected: list[dict[str, str]] = []
    devices = sorted(queues)
    while len(selected) < target:
        added = False
        for device in devices:
            if queues[device] and len(selected) < target:
                selected.append(queues[device].pop())
                added = True
        if not added:
            break
    return selected


def _balanced_generator_sample(
    rows: list[dict[str, str]],
    target: int,
    seed: int,
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["generator_name"]].append(row)
    queues = {
        generator: _stable_order(generator_rows, f"{seed}:generator-final:{generator}")
        for generator, generator_rows in grouped.items()
    }
    selected: list[dict[str, str]] = []
    generators = sorted(queues)
    while len(selected) < target:
        added = False
        for generator in generators:
            if queues[generator] and len(selected) < target:
                selected.append(queues[generator].pop())
                added = True
        if not added:
            break
    return selected


def prepare_task8b_inventory(
    *,
    task8b_root: Path,
    output_path: Path,
    report_path: Path,
    supported_extensions: set[str],
    max_per_generator: int,
    seed: int,
    allowed_generators: set[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Discover, filter, balance, and write a canonical Task 8B inventory."""

    if output_path.exists() and not overwrite:
        raise Task8BPreparationError(
            f"Inventory already exists: {output_path}; pass --overwrite after review"
        )
    if max_per_generator < 1:
        raise Task8BPreparationError("max_per_generator must be positive")
    normalized_extensions = {value.lower() for value in supported_extensions}
    premier_rows, premier_rejected = _premier_rows(task8b_root, normalized_extensions)
    genimage_rows, genimage_rejected = _genimage_rows(task8b_root, normalized_extensions)
    normalized_allowed = (
        {_slug(value) for value in allowed_generators} if allowed_generators else None
    )
    excluded_generators: Counter[str] = Counter()
    if normalized_allowed is not None:
        retained_rows = []
        for row in genimage_rows:
            if _slug(row["generator_name"]) in normalized_allowed:
                retained_rows.append(row)
            else:
                excluded_generators[row["generator_name"]] += 1
        genimage_rows = retained_rows
    if not premier_rows:
        raise Task8BPreparationError("No eligible PREMIER images were discovered")
    if not genimage_rows:
        raise Task8BPreparationError("No eligible GenImage AI images were discovered")

    selected_ai = _cap_generators(genimage_rows, max_per_generator, seed)
    selected_authentic = _balanced_authentic_sample(
        premier_rows,
        target=min(len(premier_rows), len(selected_ai)),
        seed=seed,
    )
    balanced_count = min(len(selected_authentic), len(selected_ai))
    selected_authentic = selected_authentic[:balanced_count]
    selected_ai = _balanced_generator_sample(selected_ai, balanced_count, seed)
    selected = sorted(
        [*selected_authentic, *selected_ai],
        key=lambda row: row["relative_path"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(TASK8B_REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerows(selected)
    temporary.replace(output_path)

    report = {
        "task8b_root": str(task8b_root.resolve()),
        "output": str(output_path.resolve()),
        "supported_extensions": sorted(normalized_extensions),
        "seed": seed,
        "max_per_generator": max_per_generator,
        "allowed_generators": sorted(allowed_generators) if allowed_generators else "all",
        "stress_only_generator_counts": dict(sorted(excluded_generators.items())),
        "discovered": {
            "premier": len(premier_rows),
            "genimage_ai": len(genimage_rows),
        },
        "selected": {
            "authentic": len(selected_authentic),
            "ai_generated": len(selected_ai),
            "devices": len({row["device_id"] for row in selected_authentic}),
            "generator_counts": dict(
                sorted(Counter(row["generator_name"] for row in selected_ai).items())
            ),
        },
        "rejected_count": len(premier_rejected) + len(genimage_rejected),
        "rejected": [*premier_rejected, *genimage_rejected],
        "review_required": True,
        "review_instructions": [
            "Inspect every rejected PREMIER path and supply subset/device metadata before inclusion.",
            "Confirm inferred camera and generator metadata against upstream CSV files.",
            "Do not run task8b-prepare until sources.csv has been reviewed.",
        ],
    }
    write_json(report_path, report)
    return report
