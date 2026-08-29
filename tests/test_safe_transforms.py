from __future__ import annotations

import hashlib
import random
import unittest

from PIL import Image, ImageEnhance

from cya_detector.transforms import (
    SafePolicyError,
    SafeSettings,
    apply_safe,
    validate_training_policy,
)


class SafeTransformTests(unittest.TestCase):
    @staticmethod
    def gradient(width: int = 40, height: int = 36) -> Image.Image:
        image = Image.new("RGB", (width, height))
        image.putdata(
            [
                ((x * 17 + y * 3) % 256, (x * 5 + y * 11) % 256, (x + y * 19) % 256)
                for y in range(height)
                for x in range(width)
            ]
        )
        return image

    def setUp(self) -> None:
        self.image = self.gradient()
        self.settings = SafeSettings(
            input_size=32,
            flip_probability=0.5,
            color_jitter_fraction=0.5,
            rotation_degrees=180.0,
            mask_patch_size=8,
            mask_max_fraction=0.75,
            mask_probability=1.0,
        )

    def test_safe_requires_seed_train_phase_exactly(self) -> None:
        for phase in ("selection_val", "final_test", "train", "seed_train "):
            with self.subTest(phase=phase), self.assertRaisesRegex(
                SafePolicyError, "training-only"
            ):
                apply_safe(self.image, self.settings, "sample", 42, 0, phase=phase)

    def test_safe_repeats_for_same_sample_and_epoch_with_named_local_seeds(self) -> None:
        original_state = random.getstate()
        self.addCleanup(random.setstate, original_state)
        random.seed(7)
        state_before = random.getstate()

        first = apply_safe(
            self.image, self.settings, "sample", 42, 3, phase="seed_train"
        )
        second = apply_safe(
            self.image, self.settings, "sample", 42, 3, phase="seed_train"
        )

        self.assertEqual(first.image.tobytes(), second.image.tobytes())
        self.assertEqual(first.realized, second.realized)
        self.assertEqual(random.getstate(), state_before)
        self.assertEqual(
            set(first.realized["seeds"]),
            {"crop", "flip", "jitter", "rotation", "mask"},
        )
        self.assertEqual(len(set(first.realized["seeds"].values())), 5)
        expected_crop_seed = int.from_bytes(
            hashlib.sha256(b"safe:42:3:sample:crop").digest()[:8], "big"
        )
        self.assertEqual(first.realized["seeds"]["crop"], expected_crop_seed)

        next_epoch = apply_safe(
            self.image, self.settings, "sample", 42, 4, phase="seed_train"
        )
        self.assertNotEqual(first.realized["seeds"], next_epoch.realized["seeds"])

    def test_mask_boxes_are_unique_grid_patches_and_never_exceed_limit(self) -> None:
        result = apply_safe(
            self.image, self.settings, "sample", 42, 0, phase="seed_train"
        )
        boxes = result.realized["mask_boxes"]

        self.assertTrue(result.realized["mask_applied"])
        self.assertGreater(len(boxes), 0)
        self.assertEqual(len(boxes), len({tuple(box) for box in boxes}))
        for left, top, right, bottom in boxes:
            self.assertEqual((left % 8, top % 8), (0, 0))
            self.assertEqual((right - left, bottom - top), (8, 8))
        expected_fraction = sum(
            (right - left) * (bottom - top) for left, top, right, bottom in boxes
        ) / (32 * 32)
        self.assertEqual(result.realized["mask_fraction"], expected_fraction)
        self.assertLessEqual(result.realized["mask_fraction"], 0.75)
        for box in boxes:
            self.assertEqual(result.image.crop(box).getbbox(), None)

    def test_safe_applies_the_approved_order_and_records_realized_settings(self) -> None:
        result = apply_safe(
            self.image, self.settings, "sample", 42, 0, phase="seed_train"
        )
        realized = result.realized

        self.assertEqual(
            realized["order"],
            [
                "pad",
                "random_crop",
                "horizontal_flip",
                "color_jitter",
                "rotation",
                "mask",
            ],
        )
        self.assertEqual(
            realized["settings"],
            {
                "input_size": 32,
                "flip_probability": 0.5,
                "color_jitter_fraction": 0.5,
                "rotation_degrees": 180.0,
                "mask_patch_size": 8,
                "mask_max_fraction": 0.75,
                "mask_probability": 1.0,
            },
        )
        self.assertEqual(realized["rotation_interpolation"], "bilinear")
        self.assertEqual(realized["rotation_fill"], [0, 0, 0])

        expected = self.image.crop((5, 4, 37, 36))
        if realized["flipped"]:
            expected = expected.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        expected = ImageEnhance.Brightness(expected).enhance(realized["brightness"])
        expected = ImageEnhance.Contrast(expected).enhance(realized["contrast"])
        expected = ImageEnhance.Color(expected).enhance(realized["saturation"])
        expected = expected.rotate(
            realized["rotation_angle"],
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=(0, 0, 0),
        )
        for box in realized["mask_boxes"]:
            expected.paste((0, 0, 0), box)
        self.assertEqual(result.image.tobytes(), expected.tobytes())

    def test_realized_padding_and_crop_bounds_match_literal_fixture_geometry(self) -> None:
        source = self.gradient(width=24, height=40)
        settings = SafeSettings(
            input_size=32,
            flip_probability=0.0,
            color_jitter_fraction=0.0,
            rotation_degrees=0.0,
            mask_patch_size=8,
            mask_max_fraction=0.75,
            mask_probability=0.0,
        )

        result = apply_safe(source, settings, "geometry", 42, 0, phase="seed_train")

        self.assertEqual(result.realized.get("pre_pad_size"), [24, 40])
        self.assertEqual(result.realized.get("padded_size"), [32, 40])
        self.assertEqual(result.realized.get("padding"), [4, 0, 4, 0])
        self.assertEqual(result.realized.get("crop_box"), [0, 6, 32, 38])

        expected_padded = Image.new("RGB", (32, 40), (0, 0, 0))
        expected_padded.paste(source, (4, 0))
        expected_crop = expected_padded.crop((0, 6, 32, 38))
        self.assertEqual(result.image.tobytes(), expected_crop.tobytes())


class TrainingPolicyValidationTests(unittest.TestCase):
    @staticmethod
    def config(*, controlled: bool, safe: bool) -> dict[str, object]:
        return {
            "training_policy": {
                "controlled": {"enabled": controlled},
                "safe": {"enabled": safe},
            }
        }

    def test_controlled_policy_is_selected_without_safe_augmentation(self) -> None:
        config = self.config(controlled=True, safe=False)

        self.assertEqual(
            validate_training_policy(config, phase="seed_train"),
            "controlled",
        )
        self.assertEqual(
            validate_training_policy(config, phase="selection_val"),
            "controlled",
        )

    def test_safe_policy_is_selected_only_for_seed_training(self) -> None:
        config = self.config(controlled=False, safe=True)

        self.assertEqual(validate_training_policy(config, phase="seed_train"), "safe")
        for phase in ("selection_val", "final_test", "train"):
            with self.subTest(phase=phase), self.assertRaisesRegex(
                SafePolicyError, "training-only"
            ):
                validate_training_policy(config, phase=phase)

    def test_both_training_policies_are_rejected(self) -> None:
        with self.assertRaisesRegex(SafePolicyError, "mutually exclusive"):
            validate_training_policy(
                self.config(controlled=True, safe=True),
                phase="seed_train",
            )

    def test_missing_training_policy_is_rejected(self) -> None:
        with self.assertRaisesRegex(SafePolicyError, "Exactly one"):
            validate_training_policy(
                self.config(controlled=False, safe=False),
                phase="seed_train",
            )


if __name__ == "__main__":
    unittest.main()
