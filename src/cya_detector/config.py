"""Configuration loading and validation for reproducible runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a project configuration violates the frozen schema."""


REQUIRED_SECTIONS = {
    "project",
    "runtime",
    "paths",
    "dataset",
    "model",
    "preprocessing",
    "transform_engine",
    "training_policy",
    "benchmark_transforms",
    "features",
    "optimization",
    "evaluation",
}

EXPECTED_TRANSFORMS = {
    "jpeg_quality": [90, 70, 50, 30],
    "gaussian_blur_sigma": [0.5, 1.0, 2.0],
    "resize_scale": [0.5, 0.25],
    "gaussian_noise_sigma": [0.02, 0.05, 0.1],
    "color_jitter_fraction": 0.2,
    "center_crop_fraction": 0.8,
}

EXPECTED_TRANSFORM_ENGINE = {
    "version": "task3-v1",
    "preprocessing_version": "clip-crop-v1",
    "resize_library": "Pillow",
    "resize_interpolation": "bilinear",
    "resize_filtering": "pillow_bilinear_fixed",
    "dimension_rounding": "floor(d * scale + 0.5)",
    "jpeg_storage": "JPEG",
    "non_jpeg_storage": "PNG",
    "padding": "symmetric_zero",
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a version-1 JSON configuration."""

    config_path = Path(path)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {config_path}: {exc}") from exc

    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate invariants shared by training, evaluation, and inference."""

    if config.get("schema_version") != 1:
        raise ConfigError("Only schema_version 1 is supported")

    missing = sorted(REQUIRED_SECTIONS - config.keys())
    if missing:
        raise ConfigError(f"Missing configuration sections: {', '.join(missing)}")

    if config["dataset"].get("labels") != ["authentic", "ai_generated"]:
        raise ConfigError("Dataset labels must remain ['authentic', 'ai_generated']")

    split_fractions = config["dataset"].get("split_fractions", {})
    expected_splits = {"seed_train", "self_train_pool", "selection_val", "final_test"}
    if set(split_fractions) != expected_splits:
        raise ConfigError(f"Dataset splits must be {sorted(expected_splits)}")
    if abs(sum(split_fractions.values()) - 1.0) > 1e-9:
        raise ConfigError("Dataset split fractions must sum to 1.0")
    if config["dataset"].get("c2pa_scan_required_before_derivation") is not True:
        raise ConfigError("C2PA source scanning must be required before derivation")

    transforms = config["benchmark_transforms"]
    if transforms.get("allow_chaining") is not False:
        raise ConfigError("Benchmark transform chaining must remain disabled")

    for name, expected in EXPECTED_TRANSFORMS.items():
        if transforms.get(name) != expected:
            raise ConfigError(f"Unexpected {name}: expected {expected!r}")

    transform_engine = config["transform_engine"]
    for name, expected in EXPECTED_TRANSFORM_ENGINE.items():
        if transform_engine.get(name) != expected:
            raise ConfigError(f"Unexpected transform_engine.{name}: expected {expected!r}")

    training_policy = config["training_policy"]
    controlled = training_policy.get("controlled", {})
    safe = training_policy.get("safe", {})
    if not isinstance(controlled.get("enabled"), bool) or not isinstance(
        safe.get("enabled"), bool
    ):
        raise ConfigError("Training policy enabled fields must be booleans")
    if controlled.get("enabled") == safe.get("enabled"):
        raise ConfigError("Training policies must be mutually exclusive")

    if (
        controlled.get("clean_fraction") != 0.5
        or controlled.get("transformed_fraction") != 0.5
    ):
        raise ConfigError("Controlled training fractions must remain 50/50")

    if controlled.get("balance_labels") is not True:
        raise ConfigError("Controlled training must balance labels")
    if controlled.get("uniform_transform_cells") is not True:
        raise ConfigError("Controlled training must sample transform cells uniformly")

    for name in (
        "horizontal_flip_probability",
        "color_jitter_fraction",
        "mask_max_fraction",
        "mask_probability",
    ):
        value = safe.get(name)
        if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise ConfigError(f"Safe training policy {name} must be between 0.0 and 1.0")

    evaluation = config["evaluation"]
    if evaluation.get("clean_weight") != 0.5 or evaluation.get("robustness_weight") != 0.5:
        raise ConfigError("Evaluation weights must remain 50/50")

    if config["features"].get("frequency_fast_track") is not False:
        raise ConfigError("Frequency fast-track must be disabled in the base configuration")

    model = config["model"]
    if model.get("input_size") != 336:
        raise ConfigError("The CLIP input size must remain 336")
    if model.get("freeze_backbone") is not True:
        raise ConfigError("The base configuration must freeze the CLIP backbone")
    if model.get("input_size") != config["preprocessing"].get("train_crop_size"):
        raise ConfigError("Training crop size must match the CLIP input size")
