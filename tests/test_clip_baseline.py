from __future__ import annotations

import unittest

from cya_detector.models.clip_baseline import embedding_cache_key, resolve_hugging_face_revision


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

    def test_commit_revision_does_not_require_network_resolution(self) -> None:
        commit = "0123456789abcdef0123456789abcdef01234567"
        self.assertEqual(
            resolve_hugging_face_revision("organization/model", revision=commit), commit
        )


if __name__ == "__main__":
    unittest.main()
