"""Deterministic clean-only promotion gate for the Task 9 texture pilot."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from cya_detector.evaluation.metrics import binary_metrics
from cya_detector.predictions import PredictionRecord, read_predictions


_VARIANTS = ("global_only", "local_only", "global_local")


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


def _load_clean(path: Path) -> list[PredictionRecord]:
    if not path.is_file():
        raise ValueError(f"Required texture prediction file is missing: {path}")
    rows = read_predictions(path)
    if not rows or any(row.split != "selection_val" or row.transform != "clean" for row in rows):
        raise ValueError("Texture gate requires clean selection_val prediction rows")
    if len({row.sample_id for row in rows}) != len(rows):
        raise ValueError("Texture gate requires unique prediction sample IDs")
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
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Texture gate requires unique configured seeds")
    if not 0.0 <= max_per_class_regression < 1.0:
        raise ValueError("Texture gate regression tolerance must be in [0, 1)")
    per_seed: list[dict[str, Any]] = []
    input_paths: list[Path] = []
    corrected_total = introduced_total = 0
    for seed in seeds:
        loaded = {variant: _load_clean(_prediction_path(root, variant, seed)) for variant in _VARIANTS}
        input_paths.extend(_prediction_path(root, variant, seed) for variant in _VARIANTS)
        reference = loaded["global_only"]
        aligned = {variant: _aligned(records, reference) for variant, records in loaded.items()}
        global_metrics = binary_metrics(aligned["global_only"])
        fused_metrics = binary_metrics(aligned["global_local"])
        corrected = sum(not _correct(left) and _correct(right) for left, right in zip(aligned["global_only"], aligned["global_local"], strict=True))
        introduced = sum(_correct(left) and not _correct(right) for left, right in zip(aligned["global_only"], aligned["global_local"], strict=True))
        corrected_total += corrected
        introduced_total += introduced
        per_seed.append({
            "seed": seed, "global_only_accuracy": global_metrics["accuracy"],
            "global_local_accuracy": fused_metrics["accuracy"],
            "clean_accuracy_delta": fused_metrics["accuracy"] - global_metrics["accuracy"],
            "authentic_accuracy_delta": fused_metrics["authentic_accuracy"] - global_metrics["authentic_accuracy"],
            "ai_generated_accuracy_delta": fused_metrics["ai_generated_accuracy"] - global_metrics["ai_generated_accuracy"],
            "corrected_global_errors": corrected, "introduced_global_errors": introduced,
        })
    aggregate = {
        "global_only_accuracy_mean": statistics.fmean(row["global_only_accuracy"] for row in per_seed),
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
    _atomic_json(latency_path, {"status": "not_measured_from_prediction_artifacts", "variants": list(_VARIANTS)})
    manifest = {
        "status": "completed", "decision": decision,
        "files": [
            {"path": str(path.relative_to(root)).replace("\\", "/"), "sha256": _sha256(path)}
            for path in sorted([*input_paths, comparison_path, per_seed_path, latency_path])
        ],
    }
    _atomic_json(root / "metadata" / "artifact_manifest.json", manifest)
    return report
