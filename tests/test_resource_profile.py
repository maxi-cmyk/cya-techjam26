from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from PIL import Image

from cya_detector.evaluation.resource_profile import (
    ResourceProfileError,
    checkpoint_disk_footprint,
    profile_predictor,
)


class CheckpointDiskFootprintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(".tmp") / f"resource-profile-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_reports_size_per_file_and_total(self) -> None:
        first = self.root / "checkpoint.pt"
        first.write_bytes(b"0" * 128)
        second = self.root / "layer_importance.json"
        second.write_bytes(b"1" * 32)

        report = checkpoint_disk_footprint([first, second])

        self.assertEqual(report["file_sizes_bytes"][str(first)], 128)
        self.assertEqual(report["file_sizes_bytes"][str(second)], 32)
        self.assertEqual(report["total_bytes"], 160)

    def test_rejects_empty_list(self) -> None:
        with self.assertRaises(ResourceProfileError):
            checkpoint_disk_footprint([])

    def test_rejects_missing_file(self) -> None:
        with self.assertRaises(ResourceProfileError):
            checkpoint_disk_footprint([self.root / "does-not-exist.pt"])


class ProfilePredictorTests(unittest.TestCase):
    def _images(self, count: int) -> list[Image.Image]:
        return [Image.new("RGB", (8, 8), (index, index, index)) for index in range(count)]

    def test_reports_latency_statistics_and_zero_gpu_memory_on_cpu(self) -> None:
        report = profile_predictor(lambda image: 0.5, self._images(5), warmup=1)

        self.assertEqual(report["sample_count"], 5)
        self.assertEqual(report["warmup_count"], 1)
        latency = report["latency_seconds"]
        for key in ("mean", "median", "p95", "max", "total"):
            with self.subTest(key=key):
                self.assertGreaterEqual(latency[key], 0.0)
        self.assertGreaterEqual(latency["total"], latency["mean"])
        self.assertEqual(report["peak_gpu_memory_bytes"], 0)

    def test_calls_predictor_once_per_image_plus_warmup(self) -> None:
        calls: list[int] = []

        def counting_predictor(image: Image.Image) -> float:
            calls.append(1)
            return 0.5

        profile_predictor(counting_predictor, self._images(4), warmup=2)

        self.assertEqual(len(calls), 4 + 2)

    def test_warmup_count_is_capped_at_sample_count(self) -> None:
        report = profile_predictor(lambda image: 0.5, self._images(2), warmup=10)

        self.assertEqual(report["warmup_count"], 2)

    def test_rejects_empty_image_set(self) -> None:
        with self.assertRaises(ResourceProfileError):
            profile_predictor(lambda image: 0.5, [])

    def test_rejects_negative_warmup(self) -> None:
        with self.assertRaises(ResourceProfileError):
            profile_predictor(lambda image: 0.5, self._images(1), warmup=-1)

    def test_rejects_a_non_numeric_prediction(self) -> None:
        with self.assertRaises(ResourceProfileError):
            profile_predictor(lambda image: "not-a-number", self._images(1))

    def test_rejects_a_boolean_prediction(self) -> None:
        with self.assertRaises(ResourceProfileError):
            profile_predictor(lambda image: True, self._images(1))


if __name__ == "__main__":
    unittest.main()
