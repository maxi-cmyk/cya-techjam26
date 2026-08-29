"""Independent, deterministic benchmark image operations."""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from PIL import __version__ as pillow_version


class TransformContractError(ValueError):
    """Raised when benchmark inputs violate the independent-transform contract."""


@dataclass(frozen=True)
class TransformCell:
    """One configured benchmark operation and parameter setting."""

    name: str
    parameter: int | float
    cell_id: str
    output_format: str
    stochastic: bool = False


@dataclass(frozen=True)
class TransformResult:
    """An in-memory transformed image and its realized settings."""

    image: Image.Image
    realized: dict[str, Any]


def benchmark_cells(config: dict[str, Any]) -> tuple[TransformCell, ...]:
    """Expand the frozen benchmark configuration in stable contract order."""

    transforms = config["benchmark_transforms"]
    cells: list[TransformCell] = []

    cells.extend(
        TransformCell("jpeg", quality, f"jpeg_q{quality}", "JPEG")
        for quality in transforms["jpeg_quality"]
    )
    cells.extend(
        TransformCell("blur", sigma, f"blur_sigma_{sigma}", "PNG")
        for sigma in transforms["gaussian_blur_sigma"]
    )
    cells.extend(
        TransformCell("resize", scale, f"resize_scale_{scale}", "PNG")
        for scale in transforms["resize_scale"]
    )
    cells.extend(
        TransformCell("noise", sigma, f"noise_sigma_{sigma}", "PNG", stochastic=True)
        for sigma in transforms["gaussian_noise_sigma"]
    )

    jitter_fraction = transforms["color_jitter_fraction"]
    cells.append(
        TransformCell(
            "color_jitter",
            jitter_fraction,
            f"color_jitter_{jitter_fraction}",
            "PNG",
            stochastic=True,
        )
    )
    crop_fraction = transforms["center_crop_fraction"]
    cells.append(
        TransformCell(
            "center_crop",
            crop_fraction,
            f"center_crop_{crop_fraction}",
            "PNG",
        )
    )
    return tuple(cells)


def derive_seed(project_seed: int, sample_id: str, cell_id: str) -> int:
    """Derive a local 64-bit seed without consuming shared random state."""

    digest = hashlib.sha256(f"{project_seed}:{sample_id}:{cell_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def validate_parent_record(record: dict[str, Any]) -> None:
    """Require a canonical clean parent so benchmark transforms cannot chain."""

    if record.get("image_view") != "matched_clean" or record.get("transform") != "clean":
        raise TransformContractError(
            "Benchmark parents must have image_view='matched_clean' and transform='clean'"
        )


def _require_output_format(cell: TransformCell, expected: str) -> None:
    if cell.output_format != expected:
        raise TransformContractError(
            f"Transform {cell.name!r} requires output_format={expected!r}, "
            f"not {cell.output_format!r}"
        )


def _positive_number(cell: TransformCell, *, allow_zero: bool = False) -> float:
    value = cell.parameter
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TransformContractError(f"Invalid parameter for {cell.name!r}: {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric) or (numeric < 0 if allow_zero else numeric <= 0):
        raise TransformContractError(f"Invalid parameter for {cell.name!r}: {value!r}")
    return numeric


def _jpeg(rgb: Image.Image, cell: TransformCell) -> TransformResult:
    _require_output_format(cell, "JPEG")
    quality = cell.parameter
    if isinstance(quality, bool) or not isinstance(quality, int) or not 1 <= quality <= 100:
        raise TransformContractError(f"Invalid JPEG quality: {quality!r}")

    encoded = io.BytesIO()
    rgb.save(
        encoded,
        format="JPEG",
        quality=quality,
        subsampling=0,
        optimize=False,
        progressive=False,
        exif=b"",
    )
    encoded.seek(0)
    with Image.open(encoded) as decoded:
        image = decoded.convert("RGB")
    return TransformResult(
        image,
        {
            "quality": quality,
            "subsampling": "4:4:4",
            "optimize": False,
            "progressive": False,
            "exif": "",
            "output_format": "JPEG",
        },
    )


def _blur(rgb: Image.Image, cell: TransformCell) -> TransformResult:
    _require_output_format(cell, "PNG")
    sigma = _positive_number(cell, allow_zero=True)
    return TransformResult(
        rgb.filter(ImageFilter.GaussianBlur(radius=sigma)),
        {"sigma": cell.parameter, "output_format": "PNG"},
    )


def _resize(rgb: Image.Image, cell: TransformCell) -> TransformResult:
    _require_output_format(cell, "PNG")
    scale = _positive_number(cell)
    if scale > 1:
        raise TransformContractError(f"Invalid resize scale: {cell.parameter!r}")

    width, height = rgb.size
    intermediate_size = (
        max(1, math.floor(width * scale + 0.5)),
        max(1, math.floor(height * scale + 0.5)),
    )
    downsampled = rgb.resize(intermediate_size, resample=Image.Resampling.BILINEAR)
    restored = downsampled.resize(rgb.size, resample=Image.Resampling.BILINEAR)
    return TransformResult(
        restored,
        {
            "scale": cell.parameter,
            "intermediate_size": list(intermediate_size),
            "output_size": list(rgb.size),
            "interpolation": "bilinear",
            "filtering": "pillow_bilinear_fixed",
            "dimension_rounding": "floor(d * scale + 0.5)",
            "resize_library": "Pillow",
            "resize_library_version": pillow_version,
            "mode": "RGB",
            "dtype": "uint8",
            "output_format": "PNG",
        },
    )


def _noise(
    rgb: Image.Image,
    cell: TransformCell,
    seed: int,
) -> TransformResult:
    _require_output_format(cell, "PNG")
    sigma = _positive_number(cell, allow_zero=True)
    normalized = np.asarray(rgb, dtype=np.float64) / 255.0
    rng = np.random.default_rng(seed)
    noisy = normalized + rng.normal(0.0, sigma, normalized.shape)
    rounded = np.floor(np.clip(noisy, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return TransformResult(
        Image.fromarray(rounded),
        {
            "sigma": cell.parameter,
            "seed": seed,
            "noise_space": "normalized_float_rgb",
            "clip_range": [0.0, 1.0],
            "rounding": "floor(value + 0.5)",
            "output_dtype": "uint8",
            "output_format": "PNG",
        },
    )


def _color_jitter(
    rgb: Image.Image,
    cell: TransformCell,
    seed: int,
) -> TransformResult:
    _require_output_format(cell, "PNG")
    fraction = _positive_number(cell, allow_zero=True)
    if fraction > 1:
        raise TransformContractError(f"Invalid color jitter fraction: {cell.parameter!r}")

    rng = np.random.default_rng(seed)
    brightness, contrast, saturation = (
        float(value) for value in rng.uniform(1.0 - fraction, 1.0 + fraction, size=3)
    )
    image = ImageEnhance.Brightness(rgb).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Color(image).enhance(saturation)
    return TransformResult(
        image,
        {
            "fraction": cell.parameter,
            "seed": seed,
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "order": ["brightness", "contrast", "saturation"],
            "output_format": "PNG",
        },
    )


def _center_crop(rgb: Image.Image, cell: TransformCell) -> TransformResult:
    _require_output_format(cell, "PNG")
    fraction = _positive_number(cell)
    if fraction > 1:
        raise TransformContractError(f"Invalid center crop fraction: {cell.parameter!r}")

    width, height = rgb.size
    crop_width = max(1, math.floor(width * fraction + 0.5))
    crop_height = max(1, math.floor(height * fraction + 0.5))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    bounds = (left, top, left + crop_width, top + crop_height)
    return TransformResult(
        rgb.crop(bounds),
        {
            "fraction": cell.parameter,
            "crop_bounds": list(bounds),
            "output_size": [crop_width, crop_height],
            "output_format": "PNG",
        },
    )


def apply_benchmark(
    image: Image.Image,
    cell: TransformCell,
    sample_id: str,
    project_seed: int,
) -> TransformResult:
    """Apply exactly one declared benchmark operation to an RGB-normalized image."""

    try:
        rgb = image.convert("RGB")
    except (AttributeError, OSError, ValueError) as exc:
        raise TransformContractError("Benchmark parent cannot be converted to RGB") from exc

    seed = derive_seed(project_seed, sample_id, cell.cell_id)
    if cell.name == "jpeg":
        return _jpeg(rgb, cell)
    if cell.name == "blur":
        return _blur(rgb, cell)
    if cell.name == "resize":
        return _resize(rgb, cell)
    if cell.name == "noise":
        return _noise(rgb, cell, seed)
    if cell.name == "color_jitter":
        return _color_jitter(rgb, cell, seed)
    if cell.name == "center_crop":
        return _center_crop(rgb, cell)
    raise TransformContractError(f"Unknown benchmark transform: {cell.name!r}")
