from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from cya_detector.features.frequency import (
    extract_frequency_features,
    frequency_cache_key,
)


class FrequencyFeatureTests(unittest.TestCase):
    def test_cache_key_is_configuration_specific(self) -> None:
        configuration = {"radial_bins": 24, "phase_bins": 12}
        first = frequency_cache_key(
            image_sha256="abc", extractor_version="v1", configuration=configuration
        )
        second = frequency_cache_key(
            image_sha256="abc", extractor_version="v1", configuration=configuration
        )
        changed = frequency_cache_key(
            image_sha256="abc",
            extractor_version="v2",
            configuration=configuration,
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_extraction_is_deterministic_and_finite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gradient.png"
            gradient = np.tile(np.arange(64, dtype=np.uint8), (64, 1)) * 4
            Image.fromarray(gradient, mode="L").save(path)
            first = extract_frequency_features(path)
            second = extract_frequency_features(path)
        self.assertEqual(first.names, second.names)
        np.testing.assert_array_equal(first.values, second.values)
        self.assertTrue(np.all(np.isfinite(first.values)))
        self.assertEqual(len(first.names), len(set(first.names)))
        self.assertFalse(first.metadata["resize_applied"])

    def test_phase_features_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checker.png"
            yy, xx = np.indices((64, 64))
            checker = (((xx + yy) % 2) * 255).astype(np.uint8)
            Image.fromarray(checker, mode="L").save(path)
            result = extract_frequency_features(path)
        phase_values = [
            value
            for name, value in zip(result.names, result.values, strict=True)
            if name.startswith("phase_")
        ]
        self.assertTrue(phase_values)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in phase_values))

    def test_tiny_images_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.png"
            Image.new("RGB", (8, 8)).save(path)
            with self.assertRaises(ValueError):
                extract_frequency_features(path)


if __name__ == "__main__":
    unittest.main()
