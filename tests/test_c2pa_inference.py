from __future__ import annotations

import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from cya_detector.inference.c2pa import (
    _has_created_trained_algorithmic_media_action,
    has_verified_ai_generation_claim,
)


class TrainedAlgorithmicMediaAssertionParsingTests(unittest.TestCase):
    """The break caught here is trusting an unverified, malformed, or
    authenticity-only claim as a verified AI-generation claim."""

    def _manifest(self, *, actions: list[dict]) -> dict:
        return {"assertions": [{"label": "c2pa.actions", "data": {"actions": actions}}]}

    def test_true_for_a_created_trained_algorithmic_media_action(self) -> None:
        manifest = self._manifest(
            actions=[
                {
                    "action": "c2pa.created",
                    "digitalSourceType": (
                        "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
                    ),
                }
            ]
        )

        self.assertTrue(_has_created_trained_algorithmic_media_action(manifest))

    def test_false_for_a_created_action_with_a_different_source_type(self) -> None:
        manifest = self._manifest(
            actions=[
                {
                    "action": "c2pa.created",
                    "digitalSourceType": (
                        "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"
                    ),
                }
            ]
        )

        self.assertFalse(_has_created_trained_algorithmic_media_action(manifest))

    def test_false_for_an_authenticity_only_claim(self) -> None:
        # e.g. c2pa.edited, no c2pa.created action anywhere.
        manifest = self._manifest(
            actions=[{"action": "c2pa.edited", "digitalSourceType": "irrelevant"}]
        )

        self.assertFalse(_has_created_trained_algorithmic_media_action(manifest))

    def test_false_for_missing_or_malformed_assertions(self) -> None:
        for manifest in (
            {},
            {"assertions": "not-a-list"},
            {"assertions": [{"label": "c2pa.actions", "data": "not-a-dict"}]},
            {"assertions": [{"label": "c2pa.actions", "data": {"actions": "not-a-list"}}]},
            {"assertions": [{"label": "com.other.assertion", "data": {}}]},
        ):
            with self.subTest(manifest=manifest):
                self.assertFalse(_has_created_trained_algorithmic_media_action(manifest))


class HasVerifiedAiGenerationClaimTests(unittest.TestCase):
    """The break caught here is treating absence of a claim, or a parser
    failure, as anything other than a safe False that falls through."""

    def setUp(self) -> None:
        self.root = Path(".tmp") / f"c2pa-inference-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_false_when_the_c2pa_dependency_is_missing(self) -> None:
        image_path = self.root / "plain.png"
        Image.new("RGB", (4, 4)).save(image_path)

        with patch.dict(sys.modules, {"c2pa": None}):
            self.assertFalse(has_verified_ai_generation_claim(image_path))

    def test_false_for_a_plain_image_with_no_manifest(self) -> None:
        image_path = self.root / "plain.png"
        Image.new("RGB", (4, 4)).save(image_path)

        self.assertFalse(has_verified_ai_generation_claim(image_path))

    def test_false_for_a_nonexistent_file_rather_than_raising(self) -> None:
        self.assertFalse(has_verified_ai_generation_claim(self.root / "missing.jpg"))

    def test_false_for_a_corrupt_file_rather_than_raising(self) -> None:
        corrupt_path = self.root / "corrupt.jpg"
        corrupt_path.write_bytes(b"not a real image or manifest")

        self.assertFalse(has_verified_ai_generation_claim(corrupt_path))

    def test_never_returns_true_when_reader_creation_itself_raises(self) -> None:
        image_path = self.root / "plain.png"
        Image.new("RGB", (4, 4)).save(image_path)

        with patch("c2pa.Reader.try_create", side_effect=RuntimeError("boom")):
            self.assertFalse(has_verified_ai_generation_claim(image_path))


if __name__ == "__main__":
    unittest.main()
