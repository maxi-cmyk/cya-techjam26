from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from cya_detector.features.color import extract_color_features
from cya_detector.features.common import auxiliary_cache_key
from cya_detector.features.optics import extract_optics_features
from cya_detector.features.prnu import extract_prnu_features
from cya_detector.training.auxiliary_stage_c import physical_feature_eligible


class AuxiliaryFeatureTests(unittest.TestCase):
    def _image(self, directory: str, name: str = "pattern.png") -> Path:
        path = Path(directory) / name
        yy, xx = np.indices((256, 320))
        rgb = np.stack(
            (
                (xx % 256).astype(np.uint8),
                (yy % 256).astype(np.uint8),
                ((xx + yy) % 256).astype(np.uint8),
            ),
            axis=-1,
        )
        Image.fromarray(rgb, mode="RGB").save(path)
        return path

    def test_cache_key_changes_with_configuration(self) -> None:
        first = auxiliary_cache_key(
            image_sha256="abc", extractor_version="v1", configuration={"size": 64}
        )
        changed = auxiliary_cache_key(
            image_sha256="abc", extractor_version="v1", configuration={"size": 32}
        )
        self.assertNotEqual(first, changed)

    def test_color_features_are_deterministic_and_finite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._image(directory)
            first = extract_color_features(path)
            second = extract_color_features(path)
        self.assertEqual(first.names, second.names)
        np.testing.assert_array_equal(first.values, second.values)
        self.assertTrue(np.all(np.isfinite(first.values)))
        self.assertEqual(set(first.families), {"rgb", "lab"})

    def test_prnu_proxy_is_bounded_and_not_a_camera_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = extract_prnu_features(self._image(directory))
        coherence = result.as_dict()["prnu_coherence"]
        self.assertTrue(0.0 <= coherence <= 1.0)
        self.assertIn("prnu", result.valid)

    def test_optics_emits_confidence_and_defers_distortion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = extract_optics_features(
                self._image(directory), scale_steps=5, max_analysis_size=256
            )
        self.assertTrue(0.0 <= result.confidence["ca"] <= 1.0)
        self.assertFalse(result.valid["radial_distortion"])
        self.assertIn("deferred", result.metadata["radial_distortion_reason"])

    def test_physical_eligibility_is_label_independent_and_rejects_matched_view(self) -> None:
        authentic = {
            "label": "authentic", "image_view": "source_original",
            "width": "512", "height": "512",
        }
        generated = {**authentic, "label": "ai_generated"}
        self.assertTrue(physical_feature_eligible(authentic, min_dimension=256))
        self.assertTrue(physical_feature_eligible(generated, min_dimension=256))
        authentic["image_view"] = "matched_clean"
        self.assertFalse(physical_feature_eligible(authentic, min_dimension=256))

    def test_task8b_physical_eligibility_requires_verified_provenance(self) -> None:
        row = {
            "label": "authentic",
            "dataset_name": "premier",
            "image_view": "source_original",
            "width": "512",
            "height": "512",
            "license_verified": "true",
            "physical_source_status": "native_camera",
        }
        self.assertTrue(physical_feature_eligible(row, min_dimension=256))
        row["license_verified"] = "false"
        self.assertFalse(physical_feature_eligible(row, min_dimension=256))
        row["license_verified"] = "true"
        row["physical_source_status"] = "social_media_derivative"
        self.assertFalse(physical_feature_eligible(row, min_dimension=256))


if __name__ == "__main__":
    unittest.main()
