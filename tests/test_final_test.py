from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from PIL import Image

from cya_detector.data.manifest import MANIFEST_FIELDS, sha256_file, write_manifest
from cya_detector.evaluation.final_test import FinalTestError, evaluate_final_test


class FinalTestEvaluationTests(unittest.TestCase):
    """The break caught here is a leak in the final_test boundary: reading a
    forbidden split, running twice, or publishing on any validation failure."""

    def setUp(self) -> None:
        self.root = Path(".tmp") / f"final-test-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.manifest_path = self.root / "fixed_q96_manifest.csv"
        self.output_root = self.root / "output"
        self.checkpoint_identity = {"seed": 42, "checkpoint_sha256": "a" * 64}

    def _write_rows(self, specs: list[tuple[str, str, str]]) -> None:
        """specs: list of (sample_id_prefix, label, split)."""

        rows = []
        for prefix, label, split in specs:
            image_path = self.root / f"{prefix}.jpg"
            Image.new("RGB", (8, 8), (10, 20, 30)).save(image_path, format="JPEG")
            row = {field: "" for field in MANIFEST_FIELDS}
            row.update(
                {
                    "sample_id": f"{prefix}__matched_clean__fixed_q96",
                    "source_id": prefix,
                    "parent_id": f"{prefix}__source_original",
                    "image_path": str(image_path.resolve()),
                    "image_view": "matched_clean",
                    "sha256": sha256_file(image_path),
                    "label": label,
                    "split": split,
                    "transform": "clean",
                    "dataset_name": "fixture",
                }
            )
            rows.append(row)
        write_manifest(self.manifest_path, rows)

    def _predict_by_label(self, expected: dict[str, float] | None = None):
        def predictor(image):
            return 0.5

        return predictor

    def test_rejects_manifest_with_no_final_test_rows(self) -> None:
        self._write_rows([("a", "authentic", "selection_val")])

        with self.assertRaises(FinalTestError):
            evaluate_final_test(
                manifest_path=self.manifest_path,
                predict_probability=self._predict_by_label(),
                threshold=0.5,
                output_root=self.output_root,
                checkpoint_identity=self.checkpoint_identity,
                confirm_final_test_read=True,
            )

    def test_refuses_without_explicit_confirmation(self) -> None:
        self._write_rows([("a", "authentic", "final_test"), ("b", "ai_generated", "final_test")])

        with self.assertRaises(FinalTestError):
            evaluate_final_test(
                manifest_path=self.manifest_path,
                predict_probability=self._predict_by_label(),
                threshold=0.5,
                output_root=self.output_root,
                checkpoint_identity=self.checkpoint_identity,
                confirm_final_test_read=False,
            )
        self.assertFalse((self.output_root / "final_test_report.json").exists())

    def test_scores_only_final_test_rows_and_publishes_a_report(self) -> None:
        self._write_rows(
            [
                ("a", "authentic", "final_test"),
                ("b", "ai_generated", "final_test"),
                ("c", "authentic", "selection_val"),
                ("d", "ai_generated", "seed_train"),
            ]
        )

        report = evaluate_final_test(
            manifest_path=self.manifest_path,
            predict_probability=self._predict_by_label(),
            threshold=0.5,
            output_root=self.output_root,
            checkpoint_identity=self.checkpoint_identity,
            confirm_final_test_read=True,
        )

        self.assertTrue(report["final_test_read"])
        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(report["metrics"]["sample_count"], 2)
        self.assertEqual(report["checkpoint_identity"], self.checkpoint_identity)
        published = json.loads((self.output_root / "final_test_report.json").read_text())
        self.assertEqual(published, report)

    def test_refuses_to_run_twice(self) -> None:
        self._write_rows([("a", "authentic", "final_test"), ("b", "ai_generated", "final_test")])
        evaluate_final_test(
            manifest_path=self.manifest_path,
            predict_probability=self._predict_by_label(),
            threshold=0.5,
            output_root=self.output_root,
            checkpoint_identity=self.checkpoint_identity,
            confirm_final_test_read=True,
        )

        with self.assertRaises(FinalTestError):
            evaluate_final_test(
                manifest_path=self.manifest_path,
                predict_probability=self._predict_by_label(),
                threshold=0.5,
                output_root=self.output_root,
                checkpoint_identity=self.checkpoint_identity,
                confirm_final_test_read=True,
            )

    def test_rejects_a_final_test_row_that_is_not_matched_clean(self) -> None:
        self._write_rows([("a", "authentic", "final_test"), ("b", "ai_generated", "final_test")])
        rows = []
        import csv

        with self.manifest_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["image_view"] = "benchmark"
        write_manifest(self.manifest_path, rows)

        with self.assertRaises(FinalTestError):
            evaluate_final_test(
                manifest_path=self.manifest_path,
                predict_probability=self._predict_by_label(),
                threshold=0.5,
                output_root=self.output_root,
                checkpoint_identity=self.checkpoint_identity,
                confirm_final_test_read=True,
            )
        self.assertFalse((self.output_root / "final_test_report.json").exists())

    def test_rejects_an_image_hash_mismatch(self) -> None:
        self._write_rows([("a", "authentic", "final_test"), ("b", "ai_generated", "final_test")])
        import csv

        with self.manifest_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["sha256"] = "0" * 64
        write_manifest(self.manifest_path, rows)

        with self.assertRaises(FinalTestError):
            evaluate_final_test(
                manifest_path=self.manifest_path,
                predict_probability=self._predict_by_label(),
                threshold=0.5,
                output_root=self.output_root,
                checkpoint_identity=self.checkpoint_identity,
                confirm_final_test_read=True,
            )
        self.assertFalse((self.output_root / "final_test_report.json").exists())

    def test_rejects_an_out_of_range_prediction(self) -> None:
        self._write_rows([("a", "authentic", "final_test"), ("b", "ai_generated", "final_test")])

        with self.assertRaises(FinalTestError):
            evaluate_final_test(
                manifest_path=self.manifest_path,
                predict_probability=lambda image: 1.5,
                threshold=0.5,
                output_root=self.output_root,
                checkpoint_identity=self.checkpoint_identity,
                confirm_final_test_read=True,
            )
        self.assertFalse((self.output_root / "final_test_report.json").exists())

    def test_computes_correct_accuracy_from_real_predictions(self) -> None:
        self._write_rows(
            [
                ("a", "authentic", "final_test"),
                ("b", "ai_generated", "final_test"),
                ("c", "authentic", "final_test"),
            ]
        )

        # Manifest order is a, b, c. Correctly predict a (authentic) and b
        # (ai_generated), then wrongly predict c (authentic) as ai_generated.
        def predictor(image):
            predictor.calls += 1
            return 0.1 if predictor.calls == 1 else 0.9

        predictor.calls = 0

        report = evaluate_final_test(
            manifest_path=self.manifest_path,
            predict_probability=predictor,
            threshold=0.5,
            output_root=self.output_root,
            checkpoint_identity=self.checkpoint_identity,
            confirm_final_test_read=True,
        )

        self.assertAlmostEqual(report["metrics"]["accuracy"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
