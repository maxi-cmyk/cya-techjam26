"""Deterministic image discovery and the decode/validate/normalize boundary.

Discovery never opens file content, only inspects filenames/extensions and
refuses to follow symlinks. Validation is the only place that decodes bytes,
so the empty-discovery and path-collision fatal checks fire before any
expensive decode work begins.
"""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from cya_detector.inference.contracts import ValidationError

SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"})
MAX_PIXELS = 64_000_000


class DiscoveryError(RuntimeError):
    """Raised for fatal discovery-time conditions."""


def _normalized_relative_posix(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    return unicodedata.normalize("NFC", relative.as_posix())


def discover_images(root: Path) -> list[str]:
    """Recursively discover supported images under ``root``, deterministically
    ordered by the UTF-8 bytes of their NFC-normalized relative POSIX path.

    Raises ``DiscoveryError`` if nothing is found or if two files normalize
    to the same relative path (fatal in both cases; the caller publishes
    nothing).
    """

    root = Path(root)
    if not root.is_dir():
        raise DiscoveryError(f"Input directory does not exist: {root}")

    discovered: dict[str, Path] = {}
    collisions: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        dirnames[:] = [name for name in dirnames if not (current / name).is_symlink()]
        for filename in filenames:
            candidate = current / filename
            if candidate.is_symlink():
                continue
            if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            normalized = _normalized_relative_posix(candidate, root)
            if normalized in discovered:
                collisions.add(normalized)
            else:
                discovered[normalized] = candidate

    if collisions:
        raise DiscoveryError(
            "Normalized relative path collision(s): " + ", ".join(sorted(collisions))
        )
    if not discovered:
        raise DiscoveryError(f"No supported images discovered under {root}")

    return sorted(discovered.keys(), key=lambda value: value.encode("utf-8"))


def load_and_validate_image(
    path: Path, *, relative_image_path: str
) -> Image.Image | ValidationError:
    """Decode, bomb-guard, and normalize one image to owned RGB, or report why not.

    Any exception outside the deliberately caught set here is not silently
    reclassified as a validation error — it propagates as a fatal run
    failure, matching the fail-closed pattern used throughout this pipeline.
    """

    try:
        handle = Image.open(path)
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return ValidationError(
            relative_image_path, "file_unreadable", "File could not be opened."
        )
    except UnidentifiedImageError:
        return ValidationError(
            relative_image_path, "unsupported_image",
            "Unsupported or unrecognized image format.",
        )
    except OSError:
        # The format was recognized well enough to start parsing (e.g. a
        # valid header) but reading failed partway through — a truncated or
        # otherwise corrupt file, not an access or format-recognition issue.
        return ValidationError(
            relative_image_path, "decode_failed", "Image data could not be decoded."
        )

    with handle:
        try:
            width, height = handle.size
        except (OSError, ValueError):
            return ValidationError(
                relative_image_path, "decode_failed", "Image header could not be read."
            )
        if width <= 0 or height <= 0:
            return ValidationError(
                relative_image_path, "invalid_dimensions",
                "Image has invalid (zero or negative) dimensions.",
            )
        if width * height > MAX_PIXELS:
            return ValidationError(
                relative_image_path, "decompression_bomb",
                "Image exceeds the maximum allowed pixel count.",
            )
        try:
            return handle.convert("RGB")
        except (OSError, ValueError, UnidentifiedImageError):
            return ValidationError(
                relative_image_path, "decode_failed", "Image data could not be decoded."
            )
