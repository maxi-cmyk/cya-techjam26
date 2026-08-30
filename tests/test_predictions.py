from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cya_detector.predictions import PredictionRecord, read_predictions, write_predictions


class PredictionContractTests(unittest.TestCase):
    def test_csv_round_trip_and_unknown_metadata(self) -> None:
        record = PredictionRecord(
            sample_id="sample-1",
            parent_id="parent-1",
            split="selection_val",
            label="authentic",
            logit=-2.0,
            probability=0.1,
            checkpoint="best_clean",
            seed=42,
            matching_policy="fixed_q96",
            dataset_name="",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            write_predictions(path, [record])
            loaded = read_predictions(path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].sample_id, "sample-1")
        self.assertEqual(loaded[0].dataset_name, "unknown")

    def test_invalid_probability_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PredictionRecord(
                sample_id="sample-1",
                parent_id="",
                split="selection_val",
                label="ai_generated",
                logit=2.0,
                probability=1.1,
                checkpoint="latest",
                seed=42,
                matching_policy="fixed_q96",
            )


if __name__ == "__main__":
    unittest.main()
