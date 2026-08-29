from __future__ import annotations

import unittest

import numpy as np

from cya_detector.features.frequency import extract_frequency_features


class FrequencyFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

    def test_runs_on_a_clean_rgb_image(self) -> None:
        image = self.rng.random((64, 64, 3)).astype(np.float32)
        result = extract_frequency_features(image)
        self.assertTrue(result.valid)
        self.assertGreater(result.confidence, 0.0)
        self.assertGreater(len(result.values), 0)
        for value in result.values.values():
            self.assertTrue(np.isfinite(value))

    def test_same_image_gives_identical_output(self) -> None:
        image = self.rng.random((64, 64, 3)).astype(np.float32)
        first = extract_frequency_features(image)
        second = extract_frequency_features(image)
        self.assertEqual(first.values, second.values)

    def test_handles_tiny_image_without_crashing(self) -> None:
        image = self.rng.random((4, 4, 3)).astype(np.float32)
        result = extract_frequency_features(image)
        self.assertFalse(result.valid)
        self.assertEqual(result.confidence, 0.0)

    def test_flat_image_is_invalid_not_a_crash(self) -> None:
        image = np.full((64, 64, 3), 0.5, dtype=np.float32)
        result = extract_frequency_features(image)
        self.assertFalse(result.valid)

    def test_synthetic_checkerboard_produces_a_strong_periodic_peak(self) -> None:
        size = 64
        checker = (np.indices((size, size)).sum(axis=0) % 8 < 4).astype(np.float32)
        image = np.repeat(checker[:, :, None], 3, axis=2)
        result = extract_frequency_features(image)
        self.assertTrue(result.valid)
        self.assertGreater(result.values["radial_peak_prominence_max"], 0.0)

    def test_checkerboard_peak_count_beats_natural_gradient(self) -> None:
        size = 128
        y, x = np.mgrid[0:size, 0:size]
        gradient = ((np.sin(x / 15) + np.cos(y / 20)) * 0.25 + 0.5).astype(np.float32)
        natural = np.repeat(gradient[:, :, None], 3, axis=2)

        yy, xx = np.indices((size, size))
        checker = ((xx // 4 + yy // 4) % 2) * 0.05
        artifact = np.clip(gradient + checker, 0, 1).astype(np.float32)
        artifact = np.repeat(artifact[:, :, None], 3, axis=2)

        natural_result = extract_frequency_features(natural)
        artifact_result = extract_frequency_features(artifact)

        # DC/near-DC bins must not swamp the significance threshold and hide
        # the artifact's periodic peak (regression check for that bug).
        self.assertGreater(
            artifact_result.values["radial_peak_count"],
            natural_result.values["radial_peak_count"],
        )
        self.assertGreater(
            artifact_result.values["radial_peak_prominence_max"],
            natural_result.values["radial_peak_prominence_max"],
        )


if __name__ == "__main__":
    unittest.main()
