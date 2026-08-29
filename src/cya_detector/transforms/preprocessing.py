"""Deterministic preparation of Pillow images for model input."""

from __future__ import annotations

import random

from PIL import Image


def _validate_size(size: int) -> None:
    if size <= 0:
        raise ValueError(f"input size must be positive, got {size!r}")


def to_rgb(image: Image.Image) -> Image.Image:
    """Convert an image to RGB, compositing any alpha channel over black."""

    rgba = image.convert("RGBA")
    black = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    return Image.alpha_composite(black, rgba).convert("RGB")


def pad_to_minimum(image: Image.Image, size: int) -> Image.Image:
    """Pad an image to at least ``size`` in each dimension without resizing it."""

    _validate_size(size)
    rgb = to_rgb(image)
    width, height = rgb.size
    output_width = max(width, size)
    output_height = max(height, size)
    left = (output_width - width) // 2
    top = (output_height - height) // 2

    padded = Image.new("RGB", (output_width, output_height), (0, 0, 0))
    padded.paste(rgb, (left, top))
    return padded


def center_crop_input(image: Image.Image, size: int) -> Image.Image:
    """Pad as needed, then take a deterministic centered ``size`` square crop."""

    _validate_size(size)
    padded = pad_to_minimum(image, size)
    width, height = padded.size
    left = (width - size) // 2
    top = (height - size) // 2
    return padded.crop((left, top, left + size, top + size))


def random_crop_input(image: Image.Image, size: int, *, seed: int) -> Image.Image:
    """Pad as needed, then take a crop selected by a local seeded RNG."""

    _validate_size(size)
    padded = pad_to_minimum(image, size)
    width, height = padded.size
    generator = random.Random(seed)
    left = generator.randint(0, width - size)
    top = generator.randint(0, height - size)
    return padded.crop((left, top, left + size, top + size))
