from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from cya_detector.evaluation.bootstrap import bootstrap_intervals, paired_bootstrap_difference
from cya_detector.evaluation.metrics import binary_metrics, evaluate_predictions
from cya_detector.evaluation.reporting import (
    FinalTestLockedError,
    build_report,
    write_report,
)
from cya_detector.predictions import PredictionRecord


def prediction(
    sample_id: str,
    label: str,
    probability: float,
    *,
    split: str = "selection_val",
    transform: str = "clean",
    parameter: str = "",
) -> PredictionRecord:
    return PredictionRecord(
        sample_id=sample_id,
        source_id=sample_id.split("-")[0],
        parent_id=sample_id.split("-")[0],
        split=split,
        label=label,
        logit=0.0,
        probability=probability,
        checkpoint="test",
        seed=42,
        matching_policy="fixed_q96",
        transform=transform,
        transform_parameter=parameter,
    )


class EvaluationTests(unittest.TestCase):
    def test_perfect_clean_and_robust_score(self) -> None:
        rows = [
            prediction("a-clean", "authentic", 0.1),
            prediction("b-clean", "ai_generated", 0.9),
            prediction("a-jpeg", "authentic", 0.2, transform="jpeg", parameter="90"),
            prediction("b-jpeg", "ai_generated", 0.8, transform="jpeg", parameter="90"),
        ]
        report = evaluate_predictions(rows)
        self.assertEqual(report["clean"]["accuracy"], 1.0)
        self.assertEqual(report["robustness"]["mean_accuracy"], 1.0)
        self.assertEqual(report["selection_score"], 1.0)

    def test_class_collapsed_predictor_is_exposed(self) -> None:
        rows = [
            prediction("a", "authentic", 0.1),
            prediction("b", "authentic", 0.2),
            prediction("c", "ai_generated", 0.1),
            prediction("d", "ai_generated", 0.2),
        ]
        metrics = binary_metrics(rows)
        self.assertEqual(metrics["authentic_accuracy"], 1.0)
        self.assertEqual(metrics["ai_generated_accuracy"], 0.0)
        self.assertEqual(metrics["false_negative_rate"], 1.0)

    def test_robustness_cells_are_unweighted(self) -> None:
        rows = [
            prediction("a-clean", "authentic", 0.1),
            prediction("b-clean", "ai_generated", 0.9),
            prediction("a-blur", "authentic", 0.9, transform="blur", parameter="1"),
            prediction("b-blur", "ai_generated", 0.1, transform="blur", parameter="1"),
            prediction("a-jpeg", "authentic", 0.1, transform="jpeg", parameter="90"),
            prediction("b-jpeg", "ai_generated", 0.9, transform="jpeg", parameter="90"),
        ]
        report = evaluate_predictions(rows)
        self.assertEqual(report["robustness"]["mean_accuracy"], 0.5)
        self.assertEqual(report["selection_score"], 0.75)

    def test_duplicate_source_in_one_cell_is_rejected(self) -> None:
        rows = [
            prediction("a-clean", "authentic", 0.1),
            prediction("a-copy", "authentic", 0.2),
        ]
        with self.assertRaises(ValueError):
            evaluate_predictions(rows)

    def test_final_test_is_locked(self) -> None:
        rows = [prediction("a", "authentic", 0.1, split="final_test")]
        with self.assertRaises(FinalTestLockedError):
            build_report(rows, threshold=0.5, bootstrap_iterations=2, bootstrap_seed=42)
        report = build_report(
            rows,
            threshold=0.5,
            bootstrap_iterations=2,
            bootstrap_seed=42,
            final_evaluation=True,
            architecture_frozen=True,
        )
        self.assertEqual(report["evaluation_mode"], "final_test")

    def test_normal_evaluation_accepts_selection_only(self) -> None:
        rows = [prediction("a", "authentic", 0.1, split="seed_train")]
        with self.assertRaises(ValueError):
            build_report(rows, threshold=0.5, bootstrap_iterations=2, bootstrap_seed=42)

    def test_bootstrap_is_deterministic_and_reports_can_be_written(self) -> None:
        rows = [
            prediction("a", "authentic", 0.1),
            prediction("b", "ai_generated", 0.9),
        ]
        first = bootstrap_intervals(rows, iterations=10, seed=7)
        second = bootstrap_intervals(rows, iterations=10, seed=7)
        self.assertEqual(first, second)
        report = build_report(
            rows, threshold=0.5, bootstrap_iterations=10, bootstrap_seed=7
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_report(output, report)
            self.assertTrue((output / "metrics.json").is_file())
            self.assertTrue((output / "robustness_table.csv").is_file())

    def test_paired_bootstrap_uses_matching_source_units(self) -> None:
        left = [
            prediction("a", "authentic", 0.1),
            prediction("b", "ai_generated", 0.9),
        ]
        right = [
            replace(left[0], probability=0.9, matching_policy="uniform_q95_q100"),
            replace(left[1], probability=0.1, matching_policy="uniform_q95_q100"),
        ]
        comparison = paired_bootstrap_difference(left, right, iterations=10, seed=7)
        self.assertEqual(comparison["point_difference"], 1.0)


if __name__ == "__main__":
    unittest.main()
