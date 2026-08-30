"""Configuration loading and validation for reproducible runs."""

from __future__ import annotations

import json
import math
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
    "frequency",
    "auxiliary",
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

BENCHMARK_TRANSFORM_KEYS = frozenset({"allow_chaining", *EXPECTED_TRANSFORMS})
TRANSFORM_ENGINE_KEYS = frozenset(EXPECTED_TRANSFORM_ENGINE)
TRAINING_POLICY_KEYS = frozenset({"controlled", "safe"})
CONTROLLED_POLICY_KEYS = frozenset(
    {
        "enabled",
        "clean_fraction",
        "transformed_fraction",
        "balance_labels",
        "uniform_transform_cells",
    }
)
SAFE_POLICY_KEYS = frozenset(
    {
        "enabled",
        "horizontal_flip_probability",
        "color_jitter_fraction",
        "rotation_degrees",
        "mask_patch_size",
        "mask_max_fraction",
        "mask_probability",
    }
)


def _require_exact_keys(
    value: Any,
    expected_keys: frozenset[str],
    *,
    section: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"Configuration section {section} must be an object")
    actual_keys = set(value)
    missing = sorted(expected_keys - actual_keys)
    if missing:
        raise ConfigError(f"Missing {section} key(s): {', '.join(missing)}")
    unknown = sorted(actual_keys - expected_keys)
    if unknown:
        raise ConfigError(f"Unknown {section} key(s): {', '.join(unknown)}")
    return value


def _require_probability(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be numeric and between 0.0 and 1.0")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ConfigError(f"{name} must be between 0.0 and 1.0")
    return numeric


def _matches_frozen_numeric(
    actual: Any,
    expected: float | list[int | float],
) -> bool:
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _matches_frozen_numeric(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected, strict=True)
            )
        )
    return (
        not isinstance(actual, bool)
        and isinstance(actual, (int, float))
        and actual == expected
    )


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

    if not isinstance(config, dict):
        raise ConfigError("Configuration root must be an object")
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

    transforms = _require_exact_keys(
        config["benchmark_transforms"],
        BENCHMARK_TRANSFORM_KEYS,
        section="benchmark_transforms",
    )
    if transforms.get("allow_chaining") is not False:
        raise ConfigError("Benchmark transform chaining must remain disabled")

    for name, expected in EXPECTED_TRANSFORMS.items():
        if not _matches_frozen_numeric(transforms[name], expected):
            raise ConfigError(f"Unexpected {name}: expected {expected!r}")

    transform_engine = _require_exact_keys(
        config["transform_engine"],
        TRANSFORM_ENGINE_KEYS,
        section="transform_engine",
    )
    for name, expected in EXPECTED_TRANSFORM_ENGINE.items():
        if transform_engine.get(name) != expected:
            raise ConfigError(f"Unexpected transform_engine.{name}: expected {expected!r}")

    training_policy = _require_exact_keys(
        config["training_policy"],
        TRAINING_POLICY_KEYS,
        section="training_policy",
    )
    controlled = _require_exact_keys(
        training_policy["controlled"],
        CONTROLLED_POLICY_KEYS,
        section="training_policy.controlled",
    )
    safe = _require_exact_keys(
        training_policy["safe"],
        SAFE_POLICY_KEYS,
        section="training_policy.safe",
    )
    if not isinstance(controlled.get("enabled"), bool) or not isinstance(
        safe.get("enabled"), bool
    ):
        raise ConfigError("Training policy enabled fields must be booleans")
    if controlled.get("enabled") == safe.get("enabled"):
        raise ConfigError("Training policies must be mutually exclusive")

    clean_fraction = _require_probability(
        controlled["clean_fraction"],
        name="Controlled training clean_fraction",
    )
    transformed_fraction = _require_probability(
        controlled["transformed_fraction"],
        name="Controlled training transformed_fraction",
    )
    if clean_fraction != 0.5 or transformed_fraction != 0.5:
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
        _require_probability(safe[name], name=f"Safe training policy {name}")

    rotation_degrees = safe["rotation_degrees"]
    if (
        isinstance(rotation_degrees, bool)
        or not isinstance(rotation_degrees, (int, float))
        or not math.isfinite(float(rotation_degrees))
        or rotation_degrees < 0
    ):
        raise ConfigError("Safe training policy rotation_degrees must be nonnegative")

    mask_patch_size = safe["mask_patch_size"]
    if (
        isinstance(mask_patch_size, bool)
        or not isinstance(mask_patch_size, int)
        or mask_patch_size <= 0
    ):
        raise ConfigError("Safe training policy mask_patch_size must be a positive integer")

    evaluation = config["evaluation"]
    if evaluation.get("clean_weight") != 0.5 or evaluation.get("robustness_weight") != 0.5:
        raise ConfigError("Evaluation weights must remain 50/50")

    if config["features"].get("frequency_fast_track") is not False:
        raise ConfigError("Frequency fast-track must be disabled in the base configuration")

    frequency = config["frequency"]
    if not frequency.get("extractor_version"):
        raise ConfigError("Frequency extractor version is required")
    for name in ("radial_bins", "angular_bins", "dct_bins", "phase_bins"):
        if frequency.get(name, 0) <= 1:
            raise ConfigError(f"Frequency {name} must exceed one")
    if frequency.get("max_analysis_size", 0) < 32:
        raise ConfigError("Frequency max_analysis_size must be at least 32")
    if frequency.get("workers", 0) <= 0:
        raise ConfigError("Frequency workers must be positive")

    auxiliary = config["auxiliary"]
    if not auxiliary.get("extractor_version"):
        raise ConfigError("Auxiliary extractor version is required")
    for name in ("color_window_size", "prnu_block_size", "prnu_min_dimension", "optics_min_dimension"):
        if auxiliary.get(name, 0) < 16:
            raise ConfigError(f"Auxiliary {name} must be at least 16")
    if auxiliary.get("low_variance_epsilon", 0) <= 0:
        raise ConfigError("Auxiliary low_variance_epsilon must be positive")
    if auxiliary.get("prnu_denoise_sigma", 0) <= 0:
        raise ConfigError("Auxiliary PRNU denoise sigma must be positive")
    if auxiliary.get("ca_scale_limit", 0) <= 0 or auxiliary.get("ca_scale_steps", 0) < 3:
        raise ConfigError("Auxiliary CA scale search is invalid")
    if not 0.0 <= auxiliary.get("ca_min_edge_fraction", -1) <= 1.0:
        raise ConfigError("Auxiliary CA edge fraction must be in [0, 1]")
    if auxiliary.get("max_analysis_size", 0) < 32 or auxiliary.get("optics_max_analysis_size", 0) < 32:
        raise ConfigError("Auxiliary analysis sizes must be at least 32")
    if not 0.0 <= auxiliary.get("max_eligibility_rate_gap", -1) <= 1.0:
        raise ConfigError("Auxiliary eligibility gap must be in [0, 1]")
    if auxiliary.get("workers", 0) <= 0:
        raise ConfigError("Auxiliary workers must be positive")

    model = config["model"]
    if model.get("input_size") != 336:
        raise ConfigError("The CLIP input size must remain 336")
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
