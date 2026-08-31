from __future__ import annotations

import io
import shutil
import unicodedata
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from cya_detector.inference.contracts import RunSummary
from cya_detector.inference.inputs import DiscoveryError, discover_images, load_and_validate_image
from cya_detector.inference.runner import InferenceRunFailure, run_inference


class DiscoverImagesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(".tmp") / f"discovery-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_discovers_supported_extensions_case_insensitively_and_recursively(self) -> None:
        (self.root / "nested").mkdir()
        # Discovery only inspects filenames/extensions, never file content, so
        # placeholder bytes are enough here.
        for name in ("a.JPG", "b.png", "nested/c.WebP", "d.tif", "e.txt", "f.gif"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")

        discovered = discover_images(self.root)

        self.assertEqual(set(discovered), {"a.JPG", "b.png", "nested/c.WebP", "d.tif"})

    def test_deterministic_ordering_by_utf8_bytes(self) -> None:
        for name in ("b.png", "a.png", "c.png"):
            (self.root / name).write_bytes(b"x")

        first = discover_images(self.root)
        second = discover_images(self.root)

        self.assertEqual(first, ["a.png", "b.png", "c.png"])
        self.assertEqual(first, second)

    def test_never_follows_symlinked_files_or_directories(self) -> None:
        outside = self.root.parent / f"outside-{uuid.uuid4().hex}"
        outside.mkdir()
        self.addCleanup(shutil.rmtree, outside, ignore_errors=True)
        (outside / "secret.png").write_bytes(b"x")
        try:
            (self.root / "linked_dir").symlink_to(outside, target_is_directory=True)
            (self.root / "linked_file.png").symlink_to(outside / "secret.png")
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are not supported in this environment")
        (self.root / "real.png").write_bytes(b"x")

        discovered = discover_images(self.root)

        self.assertEqual(discovered, ["real.png"])

    def test_empty_discovery_is_fatal(self) -> None:
        with self.assertRaises(DiscoveryError):
            discover_images(self.root)

    def test_normalized_path_collision_is_fatal(self) -> None:
        nfc_name = unicodedata.normalize("NFC", "café.png")
        nfd_name = unicodedata.normalize("NFD", "café.png")
        if nfc_name == nfd_name:
            self.skipTest("platform does not distinguish NFC/NFD byte sequences")
        (self.root / nfc_name).write_bytes(b"x")
        try:
            (self.root / nfd_name).write_bytes(b"y")
        except OSError:
            self.skipTest("filesystem collapses NFC/NFD variants to one file")
        if len(list(self.root.iterdir())) < 2:
            self.skipTest("filesystem collapses NFC/NFD variants to one file")

        with self.assertRaises(DiscoveryError):
            discover_images(self.root)


class LoadAndValidateImageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(".tmp") / f"load-validate-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_valid_image_is_normalized_to_owned_rgb(self) -> None:
        path = self.root / "rgba.png"
        Image.new("RGBA", (4, 4), (10, 20, 30, 128)).save(path)

        result = load_and_validate_image(path, relative_image_path="rgba.png")

        self.assertEqual(result.mode, "RGB")

    def test_truncated_but_recognized_file_is_decode_failed(self) -> None:
        buffer = io.BytesIO()
        Image.new("RGB", (64, 64), (1, 2, 3)).save(buffer, format="JPEG")
        truncated = buffer.getvalue()[: len(buffer.getvalue()) // 2]
        path = self.root / "truncated.jpg"
        path.write_bytes(truncated)

        result = load_and_validate_image(path, relative_image_path="truncated.jpg")

        self.assertEqual(result.code, "decode_failed")

    def test_unrecognized_bytes_are_unsupported_image(self) -> None:
        path = self.root / "garbage.png"
        path.write_bytes(b"not an image at all")

        result = load_and_validate_image(path, relative_image_path="garbage.png")

        self.assertEqual(result.code, "unsupported_image")

    def test_oversized_image_is_a_decompression_bomb(self) -> None:
        from cya_detector.inference import inputs as inputs_module

        path = self.root / "huge.png"
        Image.new("RGB", (4, 4)).save(path)
        with patch.object(inputs_module, "MAX_PIXELS", 1):
            result = load_and_validate_image(path, relative_image_path="huge.png")

        self.assertEqual(result.code, "decompression_bomb")

    def test_error_message_never_contains_the_absolute_path(self) -> None:
        path = self.root / "garbage.png"
        path.write_bytes(b"not an image at all")

        result = load_and_validate_image(path, relative_image_path="garbage.png")

        self.assertNotIn(str(self.root), result.message)


class RunInferenceTests(unittest.TestCase):
    """The break caught here is a pipeline that invents a probability for an
    invalid image, calls the real predictor when C2PA already answered, or
    silently swallows a predictor failure instead of treating it as fatal."""

    def setUp(self) -> None:
        self.root = Path(".tmp") / f"run-inference-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def _fake_c2pa(self, claimed: set[str]):
        def check(path: Path) -> bool:
            return path.name in claimed
        return check

    def test_scores_valid_images_and_records_invalid_ones_as_partial_success(self) -> None:
        Image.new("RGB", (4, 4), (10, 20, 30)).save(self.root / "good.png")
        (self.root / "bad.png").write_bytes(b"not an image")

        result = run_inference(
            self.root, predict_probability=lambda image: 0.7,
            has_verified_ai_generation_claim=self._fake_c2pa(set()),
        )

        self.assertEqual(result.summary, RunSummary(discovered=2, predicted=1, invalid=1))
        self.assertEqual(result.predictions[0].image_path, "good.png")
        self.assertEqual(result.predictions[0].pred, 0.7)
        self.assertEqual(result.errors[0].image_path, "bad.png")
        self.assertEqual(result.errors[0].code, "unsupported_image")

    def test_verified_c2pa_claim_short_circuits_without_calling_the_predictor(self) -> None:
        Image.new("RGB", (4, 4)).save(self.root / "ai.png")
        calls: list[str] = []

        def tracking_predictor(image) -> float:
            calls.append("called")
            return 0.1

        result = run_inference(
            self.root, predict_probability=tracking_predictor,
            has_verified_ai_generation_claim=self._fake_c2pa({"ai.png"}),
        )

        self.assertEqual(calls, [])
        self.assertEqual(result.predictions[0].pred, 1.0)

    def test_unverified_c2pa_falls_through_to_the_real_predictor(self) -> None:
        Image.new("RGB", (4, 4)).save(self.root / "plain.png")

        result = run_inference(
            self.root, predict_probability=lambda image: 0.33,
            has_verified_ai_generation_claim=self._fake_c2pa(set()),
        )

        self.assertEqual(result.predictions[0].pred, 0.33)

    def test_predictor_exception_is_a_fatal_run_failure(self) -> None:
        Image.new("RGB", (4, 4)).save(self.root / "plain.png")

        def failing_predictor(image) -> float:
            raise RuntimeError("boom")

        with self.assertRaises(InferenceRunFailure):
            run_inference(
                self.root, predict_probability=failing_predictor,
                has_verified_ai_generation_claim=self._fake_c2pa(set()),
            )

    def test_out_of_range_prediction_is_a_fatal_run_failure(self) -> None:
        Image.new("RGB", (4, 4)).save(self.root / "plain.png")

        with self.assertRaises(InferenceRunFailure):
            run_inference(
                self.root, predict_probability=lambda image: 1.5,
                has_verified_ai_generation_claim=self._fake_c2pa(set()),
            )

    def test_nan_prediction_is_a_fatal_run_failure(self) -> None:
        Image.new("RGB", (4, 4)).save(self.root / "plain.png")

        with self.assertRaises(InferenceRunFailure):
            run_inference(
                self.root, predict_probability=lambda image: float("nan"),
                has_verified_ai_generation_claim=self._fake_c2pa(set()),
            )

    def test_boolean_prediction_is_a_fatal_run_failure(self) -> None:
        Image.new("RGB", (4, 4)).save(self.root / "plain.png")

        with self.assertRaises(InferenceRunFailure):
            run_inference(
                self.root, predict_probability=lambda image: True,
                has_verified_ai_generation_claim=self._fake_c2pa(set()),
            )

    def test_empty_discovery_is_a_fatal_run_failure(self) -> None:
        with self.assertRaises(InferenceRunFailure):
            run_inference(
                self.root, predict_probability=lambda image: 0.5,
                has_verified_ai_generation_claim=self._fake_c2pa(set()),
            )

    def test_progress_callback_receives_one_line_per_image(self) -> None:
        Image.new("RGB", (4, 4)).save(self.root / "good.png")
        (self.root / "bad.png").write_bytes(b"not an image")
        lines: list[str] = []

        run_inference(
            self.root, predict_probability=lambda image: 0.5,
            has_verified_ai_generation_claim=self._fake_c2pa(set()),
            print_progress=lines.append,
        )

        self.assertEqual(len(lines), 2)
        self.assertTrue(any(line.startswith("predicted") for line in lines))
        self.assertTrue(any(line.startswith("invalid") for line in lines))


if __name__ == "__main__":
    unittest.main()
