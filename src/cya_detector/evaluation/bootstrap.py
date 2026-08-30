"""Deterministic, stratified confidence intervals for retained comparisons."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Callable, Iterable

from cya_detector.evaluation.metrics import evaluate_predictions
from cya_detector.predictions import PredictionRecord


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot take a percentile of no values")
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _stratified_resample(
    rows: list[PredictionRecord], random_state: random.Random
) -> list[PredictionRecord]:
    units: dict[str, list[PredictionRecord]] = defaultdict(list)
    for row in rows:
        unit = row.source_id or row.parent_id or row.sample_id
        units[unit].append(row)
    strata: dict[str, list[list[PredictionRecord]]] = defaultdict(list)
    for unit_rows in units.values():
        labels = {row.label for row in unit_rows}
        if len(labels) != 1:
            raise ValueError("A bootstrap source unit cannot contain both labels")
        strata[next(iter(labels))].append(unit_rows)
    sample: list[PredictionRecord] = []
    for stratum in strata.values():
        for _ in range(len(stratum)):
            sample.extend(random_state.choice(stratum))
    return sample


def bootstrap_intervals(
    records: Iterable[PredictionRecord],
    *,
    threshold: float = 0.5,
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    rows = list(records)
    if iterations < 2:
        raise ValueError("At least two bootstrap iterations are required")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")

    extractors: dict[str, Callable[[dict[str, Any]], float | None]] = {
        "clean_accuracy": lambda report: (
            report["clean"]["accuracy"] if report["clean"] is not None else None
        ),
        "robustness_mean_accuracy": lambda report: report["robustness"]["mean_accuracy"],
        "selection_score": lambda report: report["selection_score"],
    }
    distributions: dict[str, list[float]] = {name: [] for name in extractors}
    random_state = random.Random(seed)
    for _ in range(iterations):
        report = evaluate_predictions(
            _stratified_resample(rows, random_state),
            threshold=threshold,
            allow_resampled_duplicates=True,
        )
        for name, extractor in extractors.items():
            value = extractor(report)
            if value is not None:
                distributions[name].append(value)

    tail = (1.0 - confidence) / 2.0
    intervals: dict[str, Any] = {}
    for name, values in distributions.items():
        intervals[name] = (
            {
                "lower": _percentile(values, tail),
                "upper": _percentile(values, 1.0 - tail),
                "iterations": len(values),
                "confidence": confidence,
            }
            if values
            else None
        )
    return intervals


def paired_bootstrap_difference(
    left_records: Iterable[PredictionRecord],
    right_records: Iterable[PredictionRecord],
    *,
    metric: str = "clean_accuracy",
    threshold: float = 0.5,
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Estimate a paired left-minus-right interval using shared source units."""

    extractors: dict[str, Callable[[dict[str, Any]], float | None]] = {
        "clean_accuracy": lambda report: (
            report["clean"]["accuracy"] if report["clean"] is not None else None
        ),
        "robustness_mean_accuracy": lambda report: report["robustness"]["mean_accuracy"],
        "selection_score": lambda report: report["selection_score"],
    }
    if metric not in extractors:
        raise ValueError(f"Unsupported paired-bootstrap metric: {metric}")
    if iterations < 2:
        raise ValueError("At least two bootstrap iterations are required")

    def units(rows: Iterable[PredictionRecord]) -> dict[str, list[PredictionRecord]]:
        grouped: dict[str, list[PredictionRecord]] = defaultdict(list)
        for row in rows:
            grouped[row.source_id or row.parent_id or row.sample_id].append(row)
        return grouped

    left_units_rows = list(left_records)
    right_units_rows = list(right_records)
    left_units = units(left_units_rows)
    right_units = units(right_units_rows)
    if set(left_units) != set(right_units):
        raise ValueError("Paired model reports must contain the same source units")
    strata: dict[str, list[str]] = defaultdict(list)
    for unit in left_units:
        left_signature = {(row.label, row.evaluation_cell) for row in left_units[unit]}
        right_signature = {(row.label, row.evaluation_cell) for row in right_units[unit]}
        if left_signature != right_signature:
            raise ValueError(f"Evaluation cells differ for paired source unit {unit}")
        labels = {label for label, _ in left_signature}
        if len(labels) != 1:
            raise ValueError(f"Paired source unit {unit} contains multiple labels")
        strata[next(iter(labels))].append(unit)

    extractor = extractors[metric]
    left_point = extractor(evaluate_predictions(left_units_rows, threshold=threshold))
    right_point = extractor(evaluate_predictions(right_units_rows, threshold=threshold))
    if left_point is None or right_point is None:
        raise ValueError(f"Metric {metric} is unavailable in the paired reports")

    random_state = random.Random(seed)
    differences: list[float] = []
    for _ in range(iterations):
        sampled_units: list[str] = []
        for stratum in strata.values():
            sampled_units.extend(random_state.choice(stratum) for _ in range(len(stratum)))
        left_sample = [row for unit in sampled_units for row in left_units[unit]]
        right_sample = [row for unit in sampled_units for row in right_units[unit]]
        left_value = extractor(
            evaluate_predictions(
                left_sample, threshold=threshold, allow_resampled_duplicates=True
            )
        )
        right_value = extractor(
            evaluate_predictions(
                right_sample, threshold=threshold, allow_resampled_duplicates=True
            )
        )
        if left_value is None or right_value is None:
            raise ValueError(f"Metric {metric} disappeared during paired bootstrap")
        differences.append(left_value - right_value)

    tail = (1.0 - confidence) / 2.0
    return {
        "metric": metric,
        "point_difference": left_point - right_point,
        "lower": _percentile(differences, tail),
        "upper": _percentile(differences, 1.0 - tail),
        "iterations": iterations,
        "confidence": confidence,
        "left_sample_count": len(left_units_rows),
        "right_sample_count": len(right_units_rows),
    }
