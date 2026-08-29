"""Evaluation access control and machine-readable report generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from cya_detector.evaluation.bootstrap import bootstrap_intervals
from cya_detector.evaluation.metrics import evaluate_predictions
from cya_detector.predictions import PredictionRecord


class FinalTestLockedError(PermissionError):
    """Raised when a normal experiment attempts to inspect final_test."""


def enforce_evaluation_boundary(
    records: Iterable[PredictionRecord],
    *,
    final_evaluation: bool,
    architecture_frozen: bool,
) -> list[PredictionRecord]:
    rows = list(records)
    has_final = any(row.split == "final_test" for row in rows)
    if has_final and not (final_evaluation and architecture_frozen):
        raise FinalTestLockedError(
            "final_test requires both final_evaluation=True and architecture_frozen=True"
        )
    if final_evaluation and not architecture_frozen:
        raise FinalTestLockedError("Final evaluation is forbidden before architecture freeze")
    if final_evaluation:
        unexpected = sorted({row.split for row in rows if row.split != "final_test"})
        if unexpected:
            raise ValueError(
                "Final evaluation accepts final_test only; found " + ", ".join(unexpected)
            )
    if not final_evaluation:
        unexpected = sorted({row.split for row in rows if row.split != "selection_val"})
        if unexpected:
            raise ValueError(
                "Normal model evaluation accepts selection_val only; found "
                + ", ".join(unexpected)
            )
    return rows


def build_report(
    records: Iterable[PredictionRecord],
    *,
    threshold: float,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    final_evaluation: bool = False,
    architecture_frozen: bool = False,
) -> dict[str, Any]:
    rows = enforce_evaluation_boundary(
        records,
        final_evaluation=final_evaluation,
        architecture_frozen=architecture_frozen,
    )
    report = evaluate_predictions(rows, threshold=threshold)
    report["bootstrap_intervals"] = bootstrap_intervals(
        rows,
        threshold=threshold,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    report["evaluation_mode"] = "final_test" if final_evaluation else "selection_val"
    return report


def write_report(output_directory: Path, report: dict[str, Any]) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    if report["clean"] is not None:
        rows.append({"cell": "clean", **report["clean"]})
    for cell, metrics in report["robustness"]["cells"].items():
        rows.append({"cell": cell, **metrics})
    scalar_fields = [
        "sample_count",
        "accuracy",
        "authentic_accuracy",
        "ai_generated_accuracy",
        "false_positive_rate",
        "false_negative_rate",
        "ece",
        "threshold",
    ]
    with (output_directory / "robustness_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=["cell", *scalar_fields])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in writer.fieldnames})
