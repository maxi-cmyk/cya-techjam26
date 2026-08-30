from __future__ import annotations

import copy
import unittest
from pathlib import Path

from cya_detector.config import ConfigError, load_config, validate_config
from cya_detector.data.manifest import MANIFEST_FIELDS

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/colab.json"


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CONFIG_PATH)

    def test_colab_config_is_valid(self) -> None:
        self.assertEqual(self.config["schema_version"], 1)
        self.assertEqual(self.config["runtime"]["platform"], "google_colab")

    def test_transform_chaining_cannot_be_enabled(self) -> None:
        candidate = copy.deepcopy(self.config)
        candidate["benchmark_transforms"]["allow_chaining"] = True
        with self.assertRaises(ConfigError):
            validate_config(candidate)

    def test_benchmark_scalar_numeric_fields_reject_booleans(self) -> None:
        for field in ("color_jitter_fraction", "center_crop_fraction"):
            for value in (False, True):
                with self.subTest(field=field, value=value):
                    candidate = copy.deepcopy(self.config)
                    candidate["benchmark_transforms"][field] = value
                    with self.assertRaises(ConfigError):
                        validate_config(candidate)

    def test_benchmark_list_numeric_elements_reject_booleans(self) -> None:
        fields = (
            "jpeg_quality",
            "gaussian_blur_sigma",
            "resize_scale",
            "gaussian_noise_sigma",
        )
        for field in fields:
            for index in range(len(self.config["benchmark_transforms"][field])):
                for value in (False, True):
                    with self.subTest(field=field, index=index, value=value):
                        candidate = copy.deepcopy(self.config)
                        candidate["benchmark_transforms"][field][index] = value
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

    def test_task3_contract_is_frozen(self) -> None:
        config = load_config(CONFIG_PATH)
        engine = config["transform_engine"]
        self.assertEqual(engine["version"], "task3-v1")
        self.assertEqual(engine["preprocessing_version"], "clip-crop-v1")
        self.assertEqual(engine["resize_library"], "Pillow")
        self.assertEqual(engine["resize_interpolation"], "bilinear")
        self.assertEqual(engine["resize_filtering"], "pillow_bilinear_fixed")
        self.assertEqual(engine["dimension_rounding"], "floor(d * scale + 0.5)")
        self.assertEqual(engine["jpeg_storage"], "JPEG")
        self.assertEqual(engine["non_jpeg_storage"], "PNG")
        self.assertEqual(engine["padding"], "symmetric_zero")

    def test_training_policies_are_mutually_exclusive(self) -> None:
        config = load_config(CONFIG_PATH)
        config["training_policy"]["controlled"]["enabled"] = True
        config["training_policy"]["safe"]["enabled"] = True
        with self.assertRaisesRegex(ConfigError, "mutually exclusive"):
            validate_config(config)

    def test_controlled_training_fractions_remain_50_50(self) -> None:
        candidate = copy.deepcopy(self.config)
        candidate["training_policy"]["controlled"]["clean_fraction"] = 0.2
        candidate["training_policy"]["controlled"]["transformed_fraction"] = 0.8
        with self.assertRaises(ConfigError):
            validate_config(candidate)

    def test_controlled_training_requires_label_balancing(self) -> None:
        candidate = copy.deepcopy(self.config)
        candidate["training_policy"]["controlled"]["balance_labels"] = False
        with self.assertRaises(ConfigError):
            validate_config(candidate)

    def test_controlled_training_requires_uniform_transform_cells(self) -> None:
        candidate = copy.deepcopy(self.config)
        candidate["training_policy"]["controlled"]["uniform_transform_cells"] = False
        with self.assertRaises(ConfigError):
            validate_config(candidate)

    def test_training_policy_enabled_fields_must_be_booleans(self) -> None:
        for policy_name in ("controlled", "safe"):
            with self.subTest(policy=policy_name):
                candidate = copy.deepcopy(self.config)
                candidate["training_policy"][policy_name]["enabled"] = "true"
                with self.assertRaises(ConfigError):
                    validate_config(candidate)

    def test_task3_sections_reject_unknown_keys(self) -> None:
        section_paths = (
            ("benchmark_transforms",),
            ("transform_engine",),
            ("training_policy",),
            ("training_policy", "controlled"),
            ("training_policy", "safe"),
        )
        for path in section_paths:
            with self.subTest(path=path):
                candidate = copy.deepcopy(self.config)
                section = candidate
                for key in path:
                    section = section[key]
                section["unexpected_key"] = "not allowed"
                with self.assertRaisesRegex(ConfigError, "Unknown.*key"):
                    validate_config(candidate)

    def test_task3_sections_reject_missing_keys(self) -> None:
        required_fields = (
            (("benchmark_transforms",), "allow_chaining"),
            (("transform_engine",), "padding"),
            (("training_policy",), "controlled"),
            (("training_policy", "controlled"), "clean_fraction"),
            (("training_policy", "safe"), "mask_probability"),
        )
        for path, field in required_fields:
            with self.subTest(path=path, field=field):
                candidate = copy.deepcopy(self.config)
                section = candidate
                for key in path:
                    section = section[key]
                del section[field]
                with self.assertRaisesRegex(ConfigError, "Missing.*key"):
                    validate_config(candidate)

    def test_safe_requires_rotation_and_mask_patch_size(self) -> None:
        for field in ("rotation_degrees", "mask_patch_size"):
            with self.subTest(field=field):
                candidate = copy.deepcopy(self.config)
                del candidate["training_policy"]["safe"][field]
                with self.assertRaisesRegex(ConfigError, "Missing.*key"):
                    validate_config(candidate)

    def test_task3_numeric_fields_reject_booleans_and_invalid_ranges(self) -> None:
        invalid_values = (
            (("training_policy", "controlled", "clean_fraction"), True),
            (("training_policy", "safe", "horizontal_flip_probability"), True),
            (("training_policy", "safe", "mask_max_fraction"), True),
            (("training_policy", "safe", "rotation_degrees"), True),
            (("training_policy", "safe", "rotation_degrees"), -1.0),
            (("training_policy", "safe", "mask_patch_size"), True),
            (("training_policy", "safe", "mask_patch_size"), 0),
        )
        for path, value in invalid_values:
            with self.subTest(path=path, value=value):
                candidate = copy.deepcopy(self.config)
                section = candidate
                for key in path[:-1]:
                    section = section[key]
                section[path[-1]] = value
                with self.assertRaises(ConfigError):
                    validate_config(candidate)

    def test_task3_provenance_fields_are_frozen(self) -> None:
        for field in (
            "parent_sha256",
            "realized_parameters",
            "transform_version",
            "preprocessing_version",
        ):
            self.assertIn(field, MANIFEST_FIELDS)


if __name__ == "__main__":
    unittest.main()
