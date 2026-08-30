"""Deterministic clean-only promotion gate for the Task 9 texture pilot."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from cya_detector.evaluation.metrics import binary_metrics
from cya_detector.predictions import PredictionRecord, read_predictions
from cya_detector.training.texture_stage_d import (
    APPROVED_MATCHING_POLICY, LOCKED_TEXTURE_SEEDS, LOCKED_TEXTURE_VARIANTS, REQUIRED_RUN_ARTIFACTS,
)


_VARIANTS = LOCKED_TEXTURE_VARIANTS
_REQUIRED_RUN_ARTIFACTS = REQUIRED_RUN_ARTIFACTS


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _prediction_path(root: Path, variant: str, seed: int) -> Path:
    return root / variant / f"seed_{seed}" / "predictions" / "selection_val.csv"


def _require_finite(value: Any, *, name: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite(item, name=f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite(item, name=f"{name}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Texture gate refused non-finite {name}")


def _require_measurement(report: dict[str, Any], *, name: str, keys: tuple[str, ...]) -> None:
    for key in keys:
        value = report.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"Texture gate requires finite nonnegative {name} measurement: {key}")


def _load_completed_run(root: Path, variant: str, seed: int) -> tuple[list[PredictionRecord], dict[str, Any], list[Path]]:
    run_root = root / variant / f"seed_{seed}"
    paths = [run_root / relative for relative in _REQUIRED_RUN_ARTIFACTS]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"Texture gate requires completed run artifact: {missing[0]}")
    try:
        metadata = json.loads((run_root / "metadata" / "run_metadata.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_root / "reports" / "metrics.json").read_text(encoding="utf-8"))
        json.loads((run_root / "reports" / "training_history.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Texture gate requires valid completed run artifacts: {run_root}") from exc
    if (
        metadata.get("status") != "completed" or metadata.get("variant") != variant
        or metadata.get("seed") != seed or metadata.get("matching_policy") != APPROVED_MATCHING_POLICY
    ):
        raise ValueError(f"Texture gate requires completed fixed_q96 run metadata: {run_root}")
    committed_hashes = metadata.get("artifact_sha256")
    expected_paths = [path.relative_to(run_root) for path in paths if path.name != "run_metadata.json"]
    if (
        not isinstance(committed_hashes, dict)
        or set(committed_hashes) != {str(path).replace("\\", "/") for path in expected_paths}
        or any(
            not isinstance(digest, str)
            or hashlib.sha256((run_root / relative).read_bytes()).hexdigest() != digest
            for relative, digest in committed_hashes.items()
        )
    ):
        raise ValueError(f"Texture gate requires one transactionally committed artifact set: {run_root}")
    if metrics.get("selection_split") != "selection_val" or metrics.get("matching_policy") != APPROVED_MATCHING_POLICY:
        raise ValueError(f"Texture gate requires fixed_q96 selection metrics: {run_root}")
    if not isinstance(metrics.get("inference"), dict) or not isinstance(metrics.get("task4_extraction"), dict):
        raise ValueError(f"Texture gate requires measured inference and Task 4 extraction reports: {run_root}")
    _require_measurement(
        metrics["inference"], name="inference", keys=("latency_seconds", "peak_memory_bytes")
    )
    _require_measurement(
        metrics["task4_extraction"], name="Task 4 extraction",
        keys=("elapsed_seconds", "peak_gpu_memory_bytes"),
    )
    _require_finite(metrics, name="run_metrics")
    return _load_clean(run_root / "predictions" / "selection_val.csv"), metrics, paths


def _load_clean(path: Path) -> list[PredictionRecord]:
    if not path.is_file():
        raise ValueError(f"Required texture prediction file is missing: {path}")
    rows = read_predictions(path)
    if not rows or any(row.split != "selection_val" or row.transform != "clean" for row in rows):
        raise ValueError("Texture gate requires clean selection_val prediction rows")
    if len({row.sample_id for row in rows}) != len(rows):
        raise ValueError("Texture gate requires unique prediction sample IDs")
    if any(row.matching_policy != APPROVED_MATCHING_POLICY for row in rows):
        raise ValueError("Texture gate requires fixed_q96 prediction provenance")
    if any(not math.isfinite(row.logit) or not math.isfinite(row.probability) for row in rows):
        raise ValueError("Texture gate refused non-finite prediction inputs")
    return rows


def _aligned(rows: list[PredictionRecord], reference: list[PredictionRecord]) -> list[PredictionRecord]:
    by_id = {row.sample_id: row for row in rows}
    reference_by_id = {row.sample_id: row for row in reference}
    if set(by_id) != set(reference_by_id):
        raise ValueError("Texture gate prediction sample sets must match")
    if any(by_id[sample_id].label != row.label for sample_id, row in reference_by_id.items()):
        raise ValueError("Texture gate prediction labels must match")
    return [by_id[row.sample_id] for row in reference]


def _correct(row: PredictionRecord) -> bool:
    return int(row.probability >= 0.5) == row.target


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_texture_pilot(
    *, experiment_root: Path, seeds: tuple[int, ...], max_per_class_regression: float,
) -> dict[str, Any]:
    """Compare global-only and global-local clean heads over the fixed nine-run pilot."""

    root = Path(experiment_root)
    if tuple(seeds) != LOCKED_TEXTURE_SEEDS:
        raise ValueError("Texture gate requires locked seeds 42, 43, 44")
    if not 0.0 <= max_per_class_regression < 1.0:
        raise ValueError("Texture gate regression tolerance must be in [0, 1)")
    per_seed: list[dict[str, Any]] = []
    input_paths: list[Path] = []
    corrected_total = introduced_total = 0
    latency_rows: list[dict[str, Any]] = []
    for seed in seeds:
        completed = {variant: _load_completed_run(root, variant, seed) for variant in _VARIANTS}
        loaded = {variant: value[0] for variant, value in completed.items()}
        input_paths.extend(path for _, _, paths in completed.values() for path in paths)
        for variant, (_, metrics, _) in completed.items():
            latency_rows.append({
                "variant": variant, "seed": seed,
                "inference_latency_seconds": metrics["inference"]["latency_seconds"],
                "inference_peak_memory_bytes": metrics["inference"]["peak_memory_bytes"],
                "task4_extraction_elapsed_seconds": metrics["task4_extraction"].get("elapsed_seconds"),
                "task4_extraction_peak_memory_bytes": metrics["task4_extraction"].get("peak_gpu_memory_bytes"),
            })
        reference = loaded["global_only"]
        aligned = {variant: _aligned(records, reference) for variant, records in loaded.items()}
        global_metrics = binary_metrics(aligned["global_only"])
        fused_metrics = binary_metrics(aligned["global_local"])
        local_metrics = binary_metrics(aligned["local_only"])
        _require_finite(global_metrics, name="global_only_metrics")
        _require_finite(fused_metrics, name="global_local_metrics")
        _require_finite(local_metrics, name="local_only_metrics")
        corrected = sum(not _correct(left) and _correct(right) for left, right in zip(aligned["global_only"], aligned["global_local"], strict=True))
        introduced = sum(_correct(left) and not _correct(right) for left, right in zip(aligned["global_only"], aligned["global_local"], strict=True))
        corrected_total += corrected
        introduced_total += introduced
        per_seed.append({
            "seed": seed, "global_only_accuracy": global_metrics["accuracy"],
            "local_only_accuracy": local_metrics["accuracy"],
            "global_local_accuracy": fused_metrics["accuracy"],
            "clean_accuracy_delta": fused_metrics["accuracy"] - global_metrics["accuracy"],
            "authentic_accuracy_delta": fused_metrics["authentic_accuracy"] - global_metrics["authentic_accuracy"],
            "ai_generated_accuracy_delta": fused_metrics["ai_generated_accuracy"] - global_metrics["ai_generated_accuracy"],
            "corrected_global_errors": corrected, "introduced_global_errors": introduced,
        })
    aggregate = {
        "global_only_accuracy_mean": statistics.fmean(row["global_only_accuracy"] for row in per_seed),
        "local_only_accuracy_mean": statistics.fmean(row["local_only_accuracy"] for row in per_seed),
        "global_local_accuracy_mean": statistics.fmean(row["global_local_accuracy"] for row in per_seed),
        "clean_accuracy_mean_delta": statistics.fmean(row["clean_accuracy_delta"] for row in per_seed),
        "authentic_accuracy_mean_delta": statistics.fmean(row["authentic_accuracy_delta"] for row in per_seed),
        "ai_generated_accuracy_mean_delta": statistics.fmean(row["ai_generated_accuracy_delta"] for row in per_seed),
        "corrected_global_errors": corrected_total, "introduced_global_errors": introduced_total,
    }
    passes = (
        aggregate["clean_accuracy_mean_delta"] > 0.0
        and aggregate["authentic_accuracy_mean_delta"] >= -max_per_class_regression
        and aggregate["ai_generated_accuracy_mean_delta"] >= -max_per_class_regression
        and corrected_total > 0
    )
    decision = "continue_to_robustness_design" if passes else "reject_texture_clean_gate"
    report = {
        "status": "completed", "decision": decision, "selection_split": "selection_val",
        "seeds": list(seeds), "max_per_class_regression": max_per_class_regression,
        "per_seed": per_seed, "aggregate": aggregate,
    }
    comparison = root / "comparison"
    comparison_path = comparison / "global_local_comparison.json"
    per_seed_path = comparison / "per_seed_metrics.csv"
    latency_path = comparison / "latency_comparison.json"
    _atomic_json(comparison_path, report)
    _atomic_csv(per_seed_path, per_seed)
    _atomic_json(latency_path, {"status": "completed", "measurements": latency_rows})
    manifest = {
        "status": "completed", "decision": decision,
        "files": [
            {"path": str(path.relative_to(root)).replace("\\", "/"), "sha256": _sha256(path)}
            for path in sorted([*input_paths, comparison_path, per_seed_path, latency_path])
        ],
    }
    _atomic_json(root / "metadata" / "artifact_manifest.json", manifest)
    return report
