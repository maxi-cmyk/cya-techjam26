from __future__ import annotations

import unittest

import numpy as np

from cya_detector.features.prnu import extract_prnu_features


class PrnuFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

    def test_runs_on_a_clean_rgb_image(self) -> None:
        image = self.rng.random((64, 64, 3)).astype(np.float32)
        result = extract_prnu_features(image)
        self.assertTrue(result.valid)
        self.assertGreater(len(result.values), 0)
        for value in result.values.values():
            self.assertTrue(np.isfinite(value))

    def test_same_image_gives_identical_output(self) -> None:
        image = self.rng.random((64, 64, 3)).astype(np.float32)
        first = extract_prnu_features(image)
        second = extract_prnu_features(image)
        self.assertEqual(first.values, second.values)

    def test_flat_image_is_invalid_not_a_crash(self) -> None:
        image = np.full((64, 64, 3), 0.5, dtype=np.float32)
        result = extract_prnu_features(image)
        self.assertFalse(result.valid)
        self.assertEqual(result.confidence, 0.0)

    def test_handles_tiny_image_without_crashing(self) -> None:
        image = self.rng.random((8, 8, 3)).astype(np.float32)
        result = extract_prnu_features(image)
        self.assertFalse(result.valid)

    def test_residual_energy_increases_with_added_noise(self) -> None:
        y, x = np.mgrid[0:64, 0:64]
        base = ((np.sin(x / 10) + np.cos(y / 12)) * 0.25 + 0.5).astype(np.float32)
        clean = np.stack([base, base, base], axis=-1)
        noisy = np.clip(clean + self.rng.normal(0, 0.1, clean.shape), 0, 1).astype(np.float32)
        clean_result = extract_prnu_features(clean)
        noisy_result = extract_prnu_features(noisy)
        self.assertGreater(noisy_result.values["residual_energy"], clean_result.values["residual_energy"])


if __name__ == "__main__":
    unittest.main()
