"""Bounded extraction of native images from PREMIER tar archives."""

from __future__ import annotations

import hashlib
import re
import shutil
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from cya_detector.data.manifest import write_json


DEVICE_PATTERN = re.compile(r"^[A-Z]\d{2,}(?:[_-].+)?$", re.IGNORECASE)


class Task8BPremierArchiveError(ValueError):
    """Raised when a PREMIER archive cannot be extracted safely."""


def _member_identity(
    member: tarfile.TarInfo,
    supported_extensions: set[str],
) -> tuple[str, Path] | None:
    path = PurePosixPath(member.name)
    if (
        not member.isfile()
        or path.is_absolute()
        or ".." in path.parts
        or path.suffix.lower() not in supported_extensions
    ):
        return None
    device_index = next(
        (index for index, part in enumerate(path.parts) if DEVICE_PATTERN.match(part)),
        None,
    )
    if device_index is None or "images" not in {part.lower() for part in path.parts}:
        return None
    return path.parts[device_index].split("_", maxsplit=1)[0].upper(), Path(
        *path.parts[device_index:]
    )


def extract_premier_image_sample(
    *,
    archive_path: Path,
    subset: str,
    output_root: Path,
    report_path: Path,
    limit_per_device: int,
    seed: int,
    supported_extensions: set[str],
    maximum_member_bytes: int = 50_000_000,
) -> dict[str, Any]:
    """Extract a deterministic per-device sample while skipping videos and RAW files."""

    subset = subset.upper()
    if subset not in {"N1", "N2", "N3"}:
        raise Task8BPremierArchiveError("subset must be N1, N2, or N3")
    if limit_per_device < 1:
        raise Task8BPremierArchiveError("limit_per_device must be positive")
    if not archive_path.is_file():
        raise Task8BPremierArchiveError(f"Archive does not exist: {archive_path}")
    normalized_extensions = {suffix.lower() for suffix in supported_extensions}
    grouped: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            identity = _member_identity(member, normalized_extensions)
            if identity is not None:
                device, relative = identity
                grouped[device].append((member.name, relative))
    if not grouped:
        raise Task8BPremierArchiveError("No supported PREMIER images found")

    selected: dict[str, tuple[str, Path]] = {}
    selected_counts: Counter[str] = Counter()
    for device, candidates in sorted(grouped.items()):
        ordered = sorted(
            candidates,
            key=lambda item: hashlib.sha256(
                f"{seed}:{subset}:{device}:{item[0]}".encode()
            ).hexdigest(),
        )
        for member_name, relative in ordered[:limit_per_device]:
            selected[member_name] = (device, relative)
            selected_counts[device] += 1

    destination = output_root / subset
    existing_images = [
        path
        for path in destination.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized_extensions
    ] if destination.exists() else []
    if existing_images:
        raise Task8BPremierArchiveError(
            f"Destination already contains images: {destination}; review before replacing"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"task8b-{subset}-", dir=output_root) as temporary:
        temporary_root = Path(temporary)
        extracted_names: set[str] = set()
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive:
                selection = selected.get(member.name)
                if selection is None:
                    continue
                if member.size > maximum_member_bytes:
                    raise Task8BPremierArchiveError(
                        f"Archive member exceeds size limit: {member.name}"
                    )
                _, relative = selection
                target = temporary_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise Task8BPremierArchiveError(f"Cannot read member: {member.name}")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                try:
                    with Image.open(target) as image:
                        image.verify()
                except Exception as exc:
                    raise Task8BPremierArchiveError(
                        f"Extracted member is not a valid image: {member.name}: {exc}"
                    ) from exc
                extracted_names.add(member.name)
        missing = set(selected) - extracted_names
        if missing:
            raise Task8BPremierArchiveError(
                f"Archive changed between listing and extraction; missing {len(missing)} members"
            )
        destination.mkdir(parents=True, exist_ok=True)
        for device_directory in temporary_root.iterdir():
            target = destination / device_directory.name
            if target.exists():
                raise Task8BPremierArchiveError(f"Device destination exists: {target}")
            device_directory.replace(target)

    report = {
        "archive": str(archive_path.resolve()),
        "archive_preserved": True,
        "subset": subset,
        "destination": str(destination.resolve()),
        "discovered_device_counts": {
            device: len(rows) for device, rows in sorted(grouped.items())
        },
        "selected_device_counts": dict(sorted(selected_counts.items())),
        "selected_count": len(selected),
        "limit_per_device": limit_per_device,
        "seed": seed,
    }
    write_json(report_path, report)
    return report
