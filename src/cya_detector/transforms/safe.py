"""Deterministic, training-only SAFE augmentation."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from PIL import Image, ImageEnhance

from cya_detector.transforms.benchmark import TransformResult
from cya_detector.transforms.preprocessing import pad_to_minimum

_SUBSEED_NAMES = ("crop", "flip", "jitter", "rotation", "mask")


class SafePolicyError(ValueError):
    """Raised when SAFE is requested outside its isolated training policy."""


@dataclass(frozen=True)
class SafeSettings:
    """Explicit settings for one SAFE training ablation."""

    input_size: int = 336
    flip_probability: float = 0.5
    color_jitter_fraction: float = 0.5
    rotation_degrees: float = 180.0
    mask_patch_size: int = 16
    mask_max_fraction: float = 0.75
    mask_probability: float = 0.5

    def __post_init__(self) -> None:
        for name in ("input_size", "mask_patch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise SafePolicyError(f"{name} must be a positive integer")

        for name in (
            "flip_probability",
            "color_jitter_fraction",
            "mask_max_fraction",
            "mask_probability",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= value <= 1.0
            ):
                raise SafePolicyError(f"{name} must be numeric and between 0.0 and 1.0")

        rotation = self.rotation_degrees
        if (
            isinstance(rotation, bool)
            or not isinstance(rotation, (int, float))
            or not math.isfinite(float(rotation))
            or rotation < 0
        ):
            raise SafePolicyError("rotation_degrees must be a nonnegative number")


def validate_training_policy(config: Mapping[str, Any], *, phase: str) -> str:
    """Select exactly one training policy and keep SAFE out of non-training phases."""

    policies = config.get("training_policy", {})
    controlled_config = policies.get("controlled", {})
    safe_config = policies.get("safe", {})
    controlled = controlled_config.get("enabled") is True
    safe = safe_config.get("enabled") is True

    if controlled and safe:
        raise SafePolicyError("Controlled and SAFE training policies are mutually exclusive")
    if not controlled and not safe:
        raise SafePolicyError("Exactly one training policy must be enabled")
    if safe and phase != "seed_train":
        raise SafePolicyError("SAFE is training-only and requires phase='seed_train'")
    return "safe" if safe else "controlled"


def _derive_subseeds(
    project_seed: int,
    epoch: int,
    sample_id: str,
) -> dict[str, int]:
    base = f"safe:{project_seed}:{epoch}:{sample_id}"
    return {
        name: int.from_bytes(
            hashlib.sha256(f"{base}:{name}".encode()).digest()[:8], "big"
        )
        for name in _SUBSEED_NAMES
    }


def _grid_boxes(size: int, patch_size: int) -> list[tuple[int, int, int, int]]:
    return [
        (left, top, min(left + patch_size, size), min(top + patch_size, size))
        for top in range(0, size, patch_size)
        for left in range(0, size, patch_size)
    ]


def _pad_and_random_crop(
    image: Image.Image,
    size: int,
    *,
    seed: int,
) -> tuple[Image.Image, dict[str, list[int]]]:
    pre_pad_width, pre_pad_height = image.size
    padded = pad_to_minimum(image, size)
    padded_width, padded_height = padded.size
    pad_left = (padded_width - pre_pad_width) // 2
    pad_top = (padded_height - pre_pad_height) // 2
    padding = [
        pad_left,
        pad_top,
        padded_width - pre_pad_width - pad_left,
        padded_height - pre_pad_height - pad_top,
    ]

    generator = random.Random(seed)
    crop_left = generator.randint(0, padded_width - size)
    crop_top = generator.randint(0, padded_height - size)
    crop_box = [crop_left, crop_top, crop_left + size, crop_top + size]
    return padded.crop(tuple(crop_box)), {
        "pre_pad_size": [pre_pad_width, pre_pad_height],
        "padded_size": [padded_width, padded_height],
        "padding": padding,
        "crop_box": crop_box,
    }


def _apply_mask(
    image: Image.Image,
    settings: SafeSettings,
    *,
    seed: int,
) -> tuple[Image.Image, bool, float, list[list[int]], float]:
    generator = random.Random(seed)
    applied = generator.random() < settings.mask_probability
    if not applied:
        return image, False, 0.0, [], 0.0

    target_fraction = generator.uniform(0.0, settings.mask_max_fraction)
    boxes = _grid_boxes(settings.input_size, settings.mask_patch_size)
    generator.shuffle(boxes)
    target_area = target_fraction * settings.input_size * settings.input_size
    selected: list[tuple[int, int, int, int]] = []
    selected_area = 0
    for box in boxes:
        left, top, right, bottom = box
        box_area = (right - left) * (bottom - top)
        if selected_area + box_area <= target_area:
            selected.append(box)
            selected_area += box_area

    masked = image.copy()
    for box in selected:
        masked.paste((0, 0, 0), box)
    realized_fraction = selected_area / (settings.input_size * settings.input_size)
    return masked, True, target_fraction, [list(box) for box in selected], realized_fraction


def apply_safe(
    image: Image.Image,
    settings: SafeSettings,
    sample_id: str,
    project_seed: int,
    epoch: int,
    *,
    phase: str,
) -> TransformResult:
    """Apply the isolated SAFE sequence to one seed-training image."""

    if phase != "seed_train":
        raise SafePolicyError("SAFE is training-only and requires phase='seed_train'")

    seeds = _derive_subseeds(project_seed, epoch, sample_id)
    transformed, crop_provenance = _pad_and_random_crop(
        image,
        settings.input_size,
        seed=seeds["crop"],
    )

    flipped = random.Random(seeds["flip"]).random() < settings.flip_probability
    if flipped:
        transformed = transformed.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    jitter = random.Random(seeds["jitter"])
    jitter_lower = max(0.0, 1.0 - settings.color_jitter_fraction)
    jitter_upper = 1.0 + settings.color_jitter_fraction
    brightness = jitter.uniform(jitter_lower, jitter_upper)
    contrast = jitter.uniform(jitter_lower, jitter_upper)
    saturation = jitter.uniform(jitter_lower, jitter_upper)
    transformed = ImageEnhance.Brightness(transformed).enhance(brightness)
    transformed = ImageEnhance.Contrast(transformed).enhance(contrast)
    transformed = ImageEnhance.Color(transformed).enhance(saturation)

    rotation_angle = random.Random(seeds["rotation"]).uniform(
        -settings.rotation_degrees,
        settings.rotation_degrees,
    )
    transformed = transformed.rotate(
        rotation_angle,
        resample=Image.Resampling.BILINEAR,
        expand=False,
        fillcolor=(0, 0, 0),
    )

    transformed, mask_applied, mask_target_fraction, mask_boxes, mask_fraction = _apply_mask(
        transformed,
        settings,
        seed=seeds["mask"],
    )
    return TransformResult(
        transformed,
        {
            "order": [
                "pad",
                "random_crop",
                "horizontal_flip",
                "color_jitter",
                "rotation",
                "mask",
            ],
            "settings": asdict(settings),
            "seeds": seeds,
            **crop_provenance,
            "flipped": flipped,
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "rotation_angle": rotation_angle,
            "rotation_interpolation": "bilinear",
            "rotation_fill": [0, 0, 0],
            "mask_applied": mask_applied,
            "mask_target_fraction": mask_target_fraction,
            "mask_boxes": mask_boxes,
            "mask_fraction": mask_fraction,
        },
    )
