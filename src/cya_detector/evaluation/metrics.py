"""Dependency-free binary metrics and the locked 50/50 challenge score."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from cya_detector.predictions import PredictionRecord


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def binary_metrics(
    records: Iterable[PredictionRecord], *, threshold: float = 0.5, ece_bins: int = 10
) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        raise ValueError("Cannot evaluate an empty prediction set")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")
    if ece_bins <= 0:
        raise ValueError("ece_bins must be positive")

    tn = fp = fn = tp = 0
    bins: list[list[PredictionRecord]] = [[] for _ in range(ece_bins)]
    for row in rows:
        predicted = int(row.probability >= threshold)
        target = row.target
        if target == 0 and predicted == 0:
            tn += 1
        elif target == 0 and predicted == 1:
            fp += 1
        elif target == 1 and predicted == 0:
            fn += 1
        else:
            tp += 1
        bin_index = min(int(row.probability * ece_bins), ece_bins - 1)
        bins[bin_index].append(row)

    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        mean_probability = sum(row.probability for row in bucket) / len(bucket)
        empirical_positive_rate = sum(row.target for row in bucket) / len(bucket)
        ece += len(bucket) / len(rows) * abs(mean_probability - empirical_positive_rate)

    correct = tn + tp
    authentic_count = tn + fp
    ai_count = tp + fn
    return {
        "sample_count": len(rows),
        "accuracy": correct / len(rows),
        "authentic_accuracy": _safe_ratio(tn, authentic_count),
        "ai_generated_accuracy": _safe_ratio(tp, ai_count),
        "false_positive_rate": _safe_ratio(fp, authentic_count),
        "false_negative_rate": _safe_ratio(fn, ai_count),
        "ece": ece,
        "threshold": threshold,
        "confusion_matrix": {
            "true_authentic_pred_authentic": tn,
            "true_authentic_pred_ai_generated": fp,
            "true_ai_generated_pred_authentic": fn,
            "true_ai_generated_pred_ai_generated": tp,
        },
    }


def _normalized_group_value(value: str) -> str:
    value = value.strip()
    return value if value else "unknown"


def evaluate_predictions(
    records: Iterable[PredictionRecord],
    *,
    threshold: float = 0.5,
    allow_resampled_duplicates: bool = False,
) -> dict[str, Any]:
    """Evaluate clean and every independent transform cell without pooling cells."""

    rows = list(records)
    if not rows:
        raise ValueError("Cannot evaluate an empty prediction set")
    seeds = {row.seed for row in rows}
    policies = {row.matching_policy for row in rows}
    if len(seeds) != 1 or len(policies) != 1:
        raise ValueError("One evaluation report must contain exactly one seed and matching policy")
    seen_units: set[tuple[str, str]] = set()
    for row in rows:
        unit = row.source_id or row.sample_id
        key = (unit, row.evaluation_cell)
        if key in seen_units and not allow_resampled_duplicates:
            raise ValueError(f"Duplicate prediction for source/cell: {key}")
        seen_units.add(key)

    by_cell: dict[str, list[PredictionRecord]] = defaultdict(list)
    for row in rows:
        by_cell[row.evaluation_cell].append(row)

    cell_metrics = {
        cell: binary_metrics(cell_rows, threshold=threshold)
        for cell, cell_rows in sorted(by_cell.items())
    }
    clean = cell_metrics.get("clean")
    robustness_cells = {key: value for key, value in cell_metrics.items() if key != "clean"}
    robustness_accuracy = (
        sum(value["accuracy"] for value in robustness_cells.values()) / len(robustness_cells)
        if robustness_cells
        else None
    )
    selection_score = (
        0.5 * clean["accuracy"] + 0.5 * robustness_accuracy
        if clean is not None and robustness_accuracy is not None
        else None
    )

    group_fields = {
        "dataset_name": lambda row: row.dataset_name,
        "generator_name": lambda row: row.generator_name,
        "generator_checkpoint": lambda row: row.generator_checkpoint,
        "capture_source": lambda row: row.capture_source,
        "matching_policy": lambda row: row.matching_policy,
    }
    breakdowns: dict[str, dict[str, Any]] = {}
    for field, getter in group_fields.items():
        groups: dict[str, list[PredictionRecord]] = defaultdict(list)
        for row in rows:
            groups[_normalized_group_value(getter(row))].append(row)
        breakdowns[field] = {
            value: binary_metrics(group, threshold=threshold)
            for value, group in sorted(groups.items())
        }

    return {
        "score_contract": {
            "clean_weight": 0.5,
            "robustness_weight": 0.5,
            "robustness_aggregation": "unweighted_mean_of_independent_transform_cells",
        },
        "clean": clean,
        "robustness": {
            "cell_count": len(robustness_cells),
            "mean_accuracy": robustness_accuracy,
            "cells": robustness_cells,
        },
        "selection_score": selection_score,
        "all_records": binary_metrics(rows, threshold=threshold),
        "breakdowns": breakdowns,
    }
