from __future__ import annotations

import unittest

import numpy as np

from cya_detector.features.optics import extract_optics_features


def _make_grid_image(size: int = 96, spacing: int = 12) -> np.ndarray:
    channel = np.zeros((size, size), dtype=np.float32)
    channel[::spacing, :] = 1.0
    channel[:, ::spacing] = 1.0
    return channel


class OpticsFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

    def test_runs_on_an_edge_rich_image(self) -> None:
        green = _make_grid_image()
        image = np.stack([green, green, green], axis=-1)
        result = extract_optics_features(image)
        self.assertTrue(result.valid)
        self.assertGreater(len(result.values), 0)
        for value in result.values.values():
            self.assertTrue(np.isfinite(value))

    def test_low_edge_support_produces_neutral_masked_result(self) -> None:
        flat_image = np.full((64, 64, 3), 0.5, dtype=np.float32)
        result = extract_optics_features(flat_image)
        self.assertFalse(result.valid)
        self.assertEqual(result.confidence, 0.0)

    def test_shifted_red_channel_is_detected_as_chromatic_shift(self) -> None:
        green = _make_grid_image()
        red = np.roll(green, shift=2, axis=1)
        blue = green.copy()
        image = np.stack([red, green, blue], axis=-1)
        result = extract_optics_features(image)
        self.assertTrue(result.valid)
        self.assertGreater(abs(result.values["chromatic_shift_r_dx"]), 1.0)

    def test_handles_small_image_without_crashing(self) -> None:
        image = self.rng.random((8, 8, 3)).astype(np.float32)
        result = extract_optics_features(image)
        self.assertFalse(result.valid)


if __name__ == "__main__":
    unittest.main()
