from __future__ import annotations

import unittest

from cya_detector.models.clip_baseline import embedding_cache_key


class ClipBaselineContractTests(unittest.TestCase):
    def test_embedding_key_is_stable_and_view_specific(self) -> None:
        arguments = {
            "image_sha256": "abc",
            "model_identifier": "openai/clip-vit-large-patch14-336",
            "resolved_revision": "commit-1",
            "preprocessing_version": "processor-v1",
            "view_identifier": "fixed_q96:matched_clean:clean:default",
        }
        first = embedding_cache_key(**arguments)
        second = embedding_cache_key(**arguments)
        self.assertEqual(first, second)
        changed = embedding_cache_key(**{**arguments, "resolved_revision": "commit-2"})
        self.assertNotEqual(first, changed)

    def test_incomplete_cache_key_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            embedding_cache_key(
                image_sha256="",
                model_identifier="model",
                resolved_revision="revision",
                preprocessing_version="v1",
                view_identifier="clean",
            )


if __name__ == "__main__":
    unittest.main()
