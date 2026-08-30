from __future__ import annotations

import csv
import hashlib
import json
import shutil
import unittest
import uuid
from pathlib import Path

from cya_detector.predictions import PredictionRecord, write_predictions


class TextureGateTests(unittest.TestCase):
    """The breaks caught here are unsafe promotion and incomplete comparison artifacts."""

    def setUp(self) -> None:
        self.root = Path(".tmp") / f"texture-gate-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    @staticmethod
    def _records(seed: int, probabilities: tuple[float, ...], *, transform: str = "clean") -> list[PredictionRecord]:
        return [
            PredictionRecord(
                sample_id=sample_id, source_id=sample_id, parent_id=sample_id,
                split="selection_val", label=label, logit=probability * 2 - 1,
                probability=probability, checkpoint="best_clean", seed=seed,
                matching_policy="fixed_q96", transform=transform,
            )
            for sample_id, label, probability in zip(
                ("auth-1", "auth-2", "auth-3", "ai-1", "ai-2", "ai-3"),
                ("authentic", "authentic", "authentic", "ai_generated", "ai_generated", "ai_generated"), probabilities,
                strict=True,
            )
        ]

    def _write_nine_runs(
        self, *, global_only: tuple[float, ...] = (0.8, 0.7, 0.9, 0.2, 0.8, 0.9),
        global_local: tuple[float, ...] = (0.2, 0.3, 0.1, 0.8, 0.9, 0.9), transform: str = "clean",
    ) -> None:
        for variant, probabilities in (("global_only", global_only), ("local_only", global_only), ("global_local", global_local)):
            for seed in (42, 43, 44):
                path = self.root / variant / f"seed_{seed}" / "predictions" / "selection_val.csv"
                write_predictions(path, self._records(seed, probabilities, transform=transform))

    def _compare(self):
        from cya_detector.evaluation.texture_gate import compare_texture_pilot

        return compare_texture_pilot(
            experiment_root=self.root, seeds=(42, 43, 44), max_per_class_regression=0.01,
        )

    def test_promotes_only_when_global_local_improves_clean_accuracy_without_class_regression(self) -> None:
        self._write_nine_runs()
        decision = self._compare()
        self.assertEqual(decision["decision"], "continue_to_robustness_design")
        self.assertGreater(decision["aggregate"]["clean_accuracy_mean_delta"], 0.0)
        self.assertGreater(decision["aggregate"]["corrected_global_errors"], 0)
        expected = {
            self.root / "comparison" / "global_local_comparison.json",
            self.root / "comparison" / "per_seed_metrics.csv",
            self.root / "comparison" / "latency_comparison.json",
            self.root / "metadata" / "artifact_manifest.json",
        }
        self.assertTrue(all(path.is_file() for path in expected))
        manifest = json.loads((self.root / "metadata" / "artifact_manifest.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertTrue(all(len(item["sha256"]) == 64 for item in manifest["files"]))
        self.assertFalse(any(".tmp" in path.name for path in self.root.rglob("*")))

    def test_rejects_no_mean_accuracy_improvement(self) -> None:
        self._write_nine_runs(global_local=(0.8, 0.7, 0.9, 0.2, 0.8, 0.9))
        self.assertEqual(self._compare()["decision"], "reject_texture_clean_gate")

    def test_rejects_authentic_or_ai_generated_regression_over_tolerance(self) -> None:
        for name, baseline, values in (
            ("authentic", (0.2, 0.3, 0.1, 0.2, 0.2, 0.2), (0.8, 0.3, 0.1, 0.8, 0.9, 0.9)),
            ("ai_generated", (0.8, 0.7, 0.9, 0.2, 0.8, 0.9), (0.2, 0.3, 0.1, 0.2, 0.8, 0.2)),
        ):
            with self.subTest(label=name):
                self._write_nine_runs(global_only=baseline, global_local=values)
                self.assertEqual(self._compare()["decision"], "reject_texture_clean_gate")
                shutil.rmtree(self.root / "comparison", ignore_errors=True)
                shutil.rmtree(self.root / "metadata", ignore_errors=True)

    def test_rejects_missing_or_mismatched_or_nonclean_prediction_inputs(self) -> None:
        self._write_nine_runs()
        (self.root / "global_local" / "seed_44" / "predictions" / "selection_val.csv").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self._compare()
        self._write_nine_runs()
        path = self.root / "global_local" / "seed_42" / "predictions" / "selection_val.csv"
        records = self._records(42, (0.2, 0.3, 0.1, 0.8, 0.9, 0.9))
        records[0] = PredictionRecord(**{**records[0].__dict__, "sample_id": "different"})
        write_predictions(path, records)
        with self.assertRaisesRegex(ValueError, "sample"):
            self._compare()
        self._write_nine_runs(transform="jpeg")
        with self.assertRaisesRegex(ValueError, "clean"):
            self._compare()

    def test_rejects_promotion_when_global_errors_are_not_corrected(self) -> None:
        self._write_nine_runs(global_local=(0.8, 0.7, 0.9, 0.2, 0.8, 0.9))
        report = self._compare()
        self.assertEqual(report["aggregate"]["corrected_global_errors"], 0)
        self.assertEqual(report["decision"], "reject_texture_clean_gate")


if __name__ == "__main__":
    unittest.main()
