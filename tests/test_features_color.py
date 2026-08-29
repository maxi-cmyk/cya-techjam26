from __future__ import annotations

import unittest

import numpy as np

from cya_detector.features.color import extract_color_features


class ColorFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

    def test_runs_on_a_clean_rgb_image(self) -> None:
        image = self.rng.random((64, 64, 3)).astype(np.float32)
        result = extract_color_features(image)
        self.assertTrue(result.valid)
        self.assertGreater(len(result.values), 0)
        for value in result.values.values():
            self.assertTrue(np.isfinite(value))

    def test_low_variance_channel_is_masked_not_a_forced_zero(self) -> None:
        flat_image = np.full((32, 32, 3), 0.5, dtype=np.float32)
        result = extract_color_features(flat_image)
        self.assertFalse(result.valid)
        self.assertEqual(result.confidence, 0.0)

    def test_same_image_gives_identical_output(self) -> None:
        image = self.rng.random((64, 64, 3)).astype(np.float32)
        first = extract_color_features(image)
        second = extract_color_features(image)
        self.assertEqual(first.values, second.values)

    def test_perfectly_correlated_channels_give_correlation_near_one(self) -> None:
        base = self.rng.random((48, 48)).astype(np.float32)
        image = np.stack([base, base, base], axis=-1)
        result = extract_color_features(image)
        self.assertAlmostEqual(result.values["rgb_rg_global_corr"], 1.0, places=3)
        self.assertAlmostEqual(result.values["rgb_rb_global_corr"], 1.0, places=3)

    def test_handles_small_image_without_crashing(self) -> None:
        image = self.rng.random((8, 8, 3)).astype(np.float32)
        result = extract_color_features(image)
        self.assertFalse(result.valid)


if __name__ == "__main__":
    unittest.main()
