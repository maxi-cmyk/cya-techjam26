"""Bounded extraction of AI-only samples from GenImage ZIP archives."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from cya_detector.data.manifest import write_json


class Task8BArchiveError(ValueError):
    """Raised when a GenImage archive cannot be extracted safely."""


def _eligible_member(
    member: zipfile.ZipInfo,
    supported_extensions: set[str],
) -> tuple[zipfile.ZipInfo, Path] | None:
    path = PurePosixPath(member.filename)
    parts = path.parts
    lowered = [part.lower() for part in parts]
    if member.is_dir() or path.suffix.lower() not in supported_extensions:
        return None
    if path.is_absolute() or ".." in parts or "nature" in lowered or "ai" not in lowered:
        return None
    ai_index = lowered.index("ai")
    start = ai_index - 1 if ai_index and lowered[ai_index - 1] in {"train", "val"} else ai_index
    relative = Path(*parts[start:])
    return member, relative


def extract_genimage_ai_sample(
    *,
    archive_path: Path,
    generator_name: str,
    output_root: Path,
    report_path: Path,
    limit: int,
    seed: int,
    supported_extensions: set[str],
    maximum_member_bytes: int = 25_000_000,
) -> dict[str, Any]:
    """Extract a deterministic bounded sample without expanding the whole ZIP."""

    if limit < 1:
        raise Task8BArchiveError("limit must be positive")
    if (
        not generator_name.strip()
        or Path(generator_name).name != generator_name
        or generator_name in {".", ".."}
    ):
        raise Task8BArchiveError("generator_name must be one safe directory name")
    if not archive_path.is_file():
        raise Task8BArchiveError(f"Archive does not exist: {archive_path}")
    destination = output_root / generator_name
    if destination.exists():
        raise Task8BArchiveError(
            f"Destination already exists: {destination}; review it before replacing"
        )
    normalized_extensions = {suffix.lower() for suffix in supported_extensions}
    with zipfile.ZipFile(archive_path) as archive:
        total_member_count = len(archive.infolist())
        eligible = [
            result
            for member in archive.infolist()
            if (result := _eligible_member(member, normalized_extensions)) is not None
        ]
        eligible.sort(
            key=lambda item: hashlib.sha256(
                f"{seed}:{generator_name}:{item[0].filename}".encode()
            ).hexdigest()
        )
        selected = eligible[:limit]
        if len(selected) < limit:
            raise Task8BArchiveError(
                f"Archive has only {len(selected)} eligible AI images; requested {limit}"
            )
        relative_paths = [relative for _, relative in selected]
        if len(set(relative_paths)) != len(relative_paths):
            raise Task8BArchiveError("Selected archive members collide after path normalization")

        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="task8b-", dir=output_root) as temporary:
            temporary_root = Path(temporary) / generator_name
            for member, relative in selected:
                if member.file_size > maximum_member_bytes:
                    raise Task8BArchiveError(
                        f"Archive member exceeds size limit: {member.filename}"
                    )
                target = temporary_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                try:
                    with Image.open(target) as image:
                        image.verify()
                except Exception as exc:
                    raise Task8BArchiveError(
                        f"Extracted member is not a valid image: {member.filename}: {exc}"
                    ) from exc
            temporary_root.replace(destination)

    report = {
        "archive": str(archive_path.resolve()),
        "archive_preserved": True,
        "generator_name": generator_name,
        "eligible_ai_members": len(eligible),
        "selected_count": len(selected),
        "excluded_member_count": total_member_count - len(eligible),
        "seed": seed,
        "destination": str(destination.resolve()),
        "selected_paths": [
            str((destination / relative).resolve()) for relative in relative_paths
        ],
    }
    write_json(report_path, report)
    return report
