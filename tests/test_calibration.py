from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

from cya_detector.data.manifest import sha256_file
from cya_detector.evaluation.calibration import CalibrationError, fit_temperature
from cya_detector.predictions import PredictionRecord, write_predictions

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _record(
    index: int, *, logit: float, label: str, split: str = "selection_val",
    seed: int = 42, transform: str = "clean", transform_parameter: str = "",
) -> PredictionRecord:
    return PredictionRecord(
        sample_id=f"sample-{index}", source_id=f"sample-{index}", parent_id=f"sample-{index}",
        split=split, label=label, logit=logit, probability=_sigmoid(logit),
        checkpoint="controlled_rine", seed=seed, matching_policy="fixed_q96",
        transform=transform, transform_parameter=transform_parameter,
    )


class CalibrationTests(unittest.TestCase):
    """The break caught here is a calibration fit that overfits confidence,
    silently mixes seeds/cells/splits, or drifts on rerun."""

    def _synthetic_records(self, *, overconfidence: float) -> list[PredictionRecord]:
        rng = random.Random(1234)
        records = []
        for index in range(400):
            true_logit = rng.uniform(-3.0, 3.0)
            label = "ai_generated" if rng.random() < _sigmoid(true_logit) else "authentic"
            records.append(_record(index, logit=true_logit * overconfidence, label=label))
        return records

    def test_fit_recovers_the_known_overconfidence_factor(self) -> None:
        records = self._synthetic_records(overconfidence=3.0)

        result = fit_temperature(records)

        self.assertAlmostEqual(result["temperature"], 3.0, delta=0.6)
        self.assertEqual(result["sample_count"], 400)
        self.assertEqual(result["seed"], 42)
        self.assertLessEqual(result["nll_after"], result["nll_before"] + 1e-9)

    def test_fit_never_increases_negative_log_likelihood(self) -> None:
        # T=1 (the unscaled input) is always inside the search bounds, so the
        # fitted temperature can never make the likelihood worse.
        for overconfidence in (0.3, 1.0, 5.0):
            with self.subTest(overconfidence=overconfidence):
                records = self._synthetic_records(overconfidence=overconfidence)
                result = fit_temperature(records)
                self.assertLessEqual(result["nll_after"], result["nll_before"] + 1e-9)

    def test_rerun_is_deterministic(self) -> None:
        records = self._synthetic_records(overconfidence=2.0)

        first = fit_temperature(records)
        second = fit_temperature(records)

        self.assertEqual(first, second)

    def test_rejects_empty_input(self) -> None:
        with self.assertRaises(CalibrationError):
            fit_temperature([])

    def test_rejects_non_selection_val_split(self) -> None:
        records = [
            _record(0, logit=1.0, label="ai_generated", split="final_test"),
            _record(1, logit=-1.0, label="authentic", split="final_test"),
        ]
        with self.assertRaises(CalibrationError):
            fit_temperature(records)

    def test_rejects_non_clean_cell(self) -> None:
        records = [
            _record(
                0, logit=1.0, label="ai_generated", transform="benchmark",
                transform_parameter='{"cell_id": "jpeg_q90"}',
            ),
            _record(
                1, logit=-1.0, label="authentic", transform="benchmark",
                transform_parameter='{"cell_id": "jpeg_q90"}',
            ),
        ]
        with self.assertRaises(CalibrationError):
            fit_temperature(records)

    def test_rejects_mixed_seeds(self) -> None:
        records = [
            _record(0, logit=1.0, label="ai_generated", seed=42),
            _record(1, logit=-1.0, label="authentic", seed=43),
        ]
        with self.assertRaises(CalibrationError):
            fit_temperature(records)

    def test_rejects_single_class_input(self) -> None:
        records = [
            _record(0, logit=1.0, label="ai_generated"),
            _record(1, logit=2.0, label="ai_generated"),
        ]
        with self.assertRaises(CalibrationError):
            fit_temperature(records)

    def test_rejects_non_finite_logits(self) -> None:
        records = [
            _record(0, logit=float("inf"), label="ai_generated"),
            _record(1, logit=-1.0, label="authentic"),
        ]
        with self.assertRaises(CalibrationError):
            fit_temperature(records)


class CalibrationCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(".tmp") / f"calibration-cli-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_help_exits_zero(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/fit_temperature_calibration.py", "--help"],
            cwd=REPO_ROOT, capture_output=True, check=False, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fits_and_publishes_a_hash_verified_report(self) -> None:
        rng = random.Random(99)
        records = []
        for index in range(60):
            true_logit = rng.uniform(-3.0, 3.0)
            label = "ai_generated" if rng.random() < _sigmoid(true_logit) else "authentic"
            records.append(_record(index, logit=true_logit * 2.0, label=label))
        # A non-clean, non-matching-seed row must be filtered out rather than
        # crashing the CLI or contaminating the fit.
        records.append(_record(999, logit=5.0, label="ai_generated", transform="benchmark"))
        records.append(_record(998, logit=5.0, label="ai_generated", seed=43))
        predictions_path = self.root / "best_50_50_predictions.csv"
        write_predictions(predictions_path, records)
        output_path = self.root / "calibration.json"

        result = subprocess.run(
            [
                sys.executable, "scripts/fit_temperature_calibration.py",
                "--predictions", str(predictions_path),
                "--seed", "42",
                "--output", str(output_path),
            ],
            cwd=REPO_ROOT, capture_output=True, check=False, text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(report["sample_count"], 60)
        self.assertEqual(report["seed"], 42)
        self.assertEqual(report["source_predictions_sha256"], sha256_file(predictions_path))


if __name__ == "__main__":
    unittest.main()
