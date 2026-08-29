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

    evaluation = config["evaluation"]
    if evaluation.get("clean_weight") != 0.5 or evaluation.get("robustness_weight") != 0.5:
        raise ConfigError("Evaluation weights must remain 50/50")

    if config["features"].get("frequency_fast_track") is not False:
        raise ConfigError("Frequency fast-track must be disabled in the base configuration")

    model = config["model"]
    if model.get("freeze_backbone") is not True:
        raise ConfigError("The base configuration must freeze the CLIP backbone")
    if model.get("input_size") != config["preprocessing"].get("train_crop_size"):
        raise ConfigError("Training crop size must match the CLIP input size")
    if not model.get("revision"):
        raise ConfigError("A requested model revision is required")
    rine_layers = model.get("rine_layers", [])
    if not rine_layers or rine_layers != sorted(set(rine_layers)):
        raise ConfigError("RINE layers must be non-empty, unique, and increasing")
    if rine_layers[0] < 1:
        raise ConfigError("RINE layer indices must be positive")
    if not model.get("rine_representation_version"):
        raise ConfigError("A RINE representation version is required")
    if not config["preprocessing"].get("version"):
        raise ConfigError("A preprocessing version is required for embedding caches")
    if config["evaluation"].get("bootstrap_iterations", 0) < 2:
        raise ConfigError("At least two bootstrap iterations are required")
    regression = config["evaluation"].get("max_per_class_accuracy_regression")
    if regression is None or not 0.0 <= regression < 1.0:
        raise ConfigError("Per-class regression tolerance must be in [0, 1)")
    warmup_fraction = config["optimization"].get("warmup_fraction")
    if warmup_fraction is None or not 0.0 <= warmup_fraction < 1.0:
        raise ConfigError("Warmup fraction must be in [0, 1)")
