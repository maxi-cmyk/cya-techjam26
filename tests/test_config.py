from __future__ import annotations

import copy
import unittest
from pathlib import Path

from cya_detector.config import ConfigError, load_config, validate_config


REPO_ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(REPO_ROOT / "configs/colab.json")

    def test_colab_config_is_valid(self) -> None:
        self.assertEqual(self.config["schema_version"], 1)
        self.assertEqual(self.config["runtime"]["platform"], "google_colab")

    def test_transform_chaining_cannot_be_enabled(self) -> None:
        candidate = copy.deepcopy(self.config)
        candidate["benchmark_transforms"]["allow_chaining"] = True
        with self.assertRaises(ConfigError):
            validate_config(candidate)

    def test_score_weights_remain_equal(self) -> None:
        candidate = copy.deepcopy(self.config)
        candidate["evaluation"]["clean_weight"] = 0.6
        with self.assertRaises(ConfigError):
            validate_config(candidate)

    def test_fast_track_is_disabled_in_base_config(self) -> None:
        candidate = copy.deepcopy(self.config)
        candidate["features"]["frequency_fast_track"] = True
        with self.assertRaises(ConfigError):
            validate_config(candidate)

    def test_split_fractions_must_sum_to_one(self) -> None:
        candidate = copy.deepcopy(self.config)
        candidate["dataset"]["split_fractions"]["seed_train"] = 0.5
        with self.assertRaises(ConfigError):
            validate_config(candidate)


if __name__ == "__main__":
    unittest.main()
