from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from cya_detector.data.dataset import ManifestExample
from cya_detector.models.clip_baseline import LoadedClip


class _Processor:
    def __call__(self, *, images, return_tensors: str):
        image = torch.as_tensor(__import__("numpy").array(images.convert("RGB"), copy=True), dtype=torch.float32)
        pixels = image.permute(2, 0, 1).unsqueeze(0) / 255.0
        return {"pixel_values": torch.nn.functional.interpolate(pixels, size=(16, 16))}


class _Encoder:
    def __init__(self, *, nonfinite: bool = False) -> None:
        self.config = SimpleNamespace(num_hidden_layers=2, hidden_size=3, projection_dim=2)
        self.calls = 0
        self.nonfinite = nonfinite

    def __call__(self, *, pixel_values, output_hidden_states=False, return_dict=False):
        self.calls += 1
        count = pixel_values.shape[0]
        embeds = torch.arange(count * 2, dtype=torch.float32).reshape(count, 2)
        if self.nonfinite:
            embeds[0, 0] = float("nan")
        states = tuple(torch.full((count, 1, 3), float(layer)) for layer in range(3))
        return SimpleNamespace(image_embeds=embeds, hidden_states=states)


def _example(
    path: Path, *, sample_id: str, split: str = "seed_train", sha256: str = "sha",
    image_view: str = "matched_clean", transform: str = "clean",
) -> ManifestExample:
    return ManifestExample(
        sample_id=sample_id, source_id="source", parent_id="parent", image_path=path,
        sha256=sha256, label="authentic", split=split, image_view=image_view,
        transform=transform, transform_parameter="", metadata={
            "dataset_name": "fixture", "generator_name": "unknown",
            "generator_checkpoint": "unknown", "capture_source": "unknown",
        },
    )


class TextureExtractionTests(unittest.TestCase):
    def _root(self) -> Path:
        root = Path(".tmp") / f"texture-extraction-{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        return root

    def _extract(self, root: Path, encoder: _Encoder, examples: list[ManifestExample], **changes):
        from cya_detector.training.texture_stage_d import extract_texture_features

        return extract_texture_features(
            loaded_clip=LoadedClip(encoder, _Processor(), "fixture", "requested", changes.pop("revision", "resolved"), 2),
            examples=examples, global_cache_root=root / "global", patch_cache_root=root / "patches",
            matching_policy="fixed", preprocessing_version="prep", rine_representation_version="rine-v1",
            texture_extractor_version=changes.pop("version", "texture-v1"), layers=(1, 2),
            patch_size=16, patch_count=4, batch_size=2, device="cpu", **changes,
        )

    def test_extracts_fixed_patch_cache_and_reuses_valid_caches(self) -> None:
        with self.subTest("first extraction"):
            root = self._root()
            first_image, second_image = root / "one.png", root / "two.png"
            Image.new("RGB", (16, 16), (10, 20, 30)).save(first_image)
            Image.new("RGB", (8, 9), (30, 40, 50)).save(second_image)
            examples = [_example(first_image, sample_id="one"), _example(second_image, sample_id="two", split="selection_val", sha256="sha2")]
            encoder = _Encoder()
            rows, report = self._extract(root, encoder, examples)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row.global_cache_path.is_file() and row.patch_cache_path.is_file() for row in rows))
            patch = torch.load(rows[1].patch_cache_path, weights_only=True)
            self.assertEqual(tuple(patch["patch_features"].shape), (4, 2))
            self.assertEqual(patch["patch_mask"].tolist(), [True, False, False, False])
            self.assertEqual(len(patch["patch_boxes"]), 1)
            self.assertEqual(report["extracted_count"], 2)
            self.assertEqual(report["patch_count"], 4)
            self.assertEqual(report["projection_dimension"], 2)
            self.assertIn("model_revision", report)
            self.assertIn("cache_total_bytes", report)
            first_calls = encoder.calls
            _, cached_report = self._extract(root, encoder, examples)
            self.assertEqual(encoder.calls, first_calls)
            self.assertEqual(cached_report["cache_hit_count"], 2)

    def test_rejects_disallowed_splits_and_missing_sha_before_encoder(self) -> None:
        with self.subTest("input validation"):
            root = self._root()
            image = root / "image.png"
            Image.new("RGB", (16, 16)).save(image)
            encoder = _Encoder()
            with self.assertRaises(ValueError):
                self._extract(root, encoder, [_example(image, sample_id="bad", split="final_test")])
            with self.assertRaises(ValueError):
                self._extract(root, encoder, [_example(image, sample_id="missing", sha256="")])
            self.assertEqual(encoder.calls, 0)

    def test_rejects_non_matched_clean_views_before_encoder(self) -> None:
        root = self._root()
        image = root / "image.png"
        Image.new("RGB", (16, 16)).save(image)
        encoder = _Encoder()
        for image_view, transform in (("source_original", "clean"), ("matched_clean", "jpeg")):
            with self.subTest(image_view=image_view, transform=transform):
                with self.assertRaises(ValueError):
                    self._extract(root, encoder, [_example(
                        image, sample_id=f"{image_view}-{transform}", image_view=image_view,
                        transform=transform,
                    )])
        self.assertEqual(encoder.calls, 0)

    def test_corrupt_or_stale_global_cache_is_reextracted_before_reuse(self) -> None:
        root = self._root()
        image = root / "image.png"
        Image.new("RGB", (16, 16)).save(image)
        encoder = _Encoder()
        rows, _ = self._extract(root, encoder, [_example(image, sample_id="sample")])
        torch.save(torch.zeros(1), rows[0].global_cache_path)
        calls = encoder.calls
        self._extract(root, encoder, [_example(image, sample_id="sample")])
        self.assertGreater(encoder.calls, calls)
        self.assertEqual(tuple(torch.load(rows[0].global_cache_path, weights_only=True).shape), (2, 3))
        metadata_path = rows[0].global_cache_path.with_suffix(".meta.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["resolved_revision"] = "stale"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        calls = encoder.calls
        self._extract(root, encoder, [_example(image, sample_id="sample")])
        self.assertGreater(encoder.calls, calls)

    def test_coordinates_version_and_revision_invalidate_patch_cache(self) -> None:
        with self.subTest("invalidation"):
            root = self._root()
            image = root / "image.png"
            Image.new("RGB", (32, 16), (10, 20, 30)).save(image)
            example = _example(image, sample_id="sample")
            encoder = _Encoder()
            rows, _ = self._extract(root, encoder, [example])
            payload = torch.load(rows[0].patch_cache_path, weights_only=True)
            payload["patch_boxes"] = [[1, 1, 16, 16]]
            torch.save(payload, rows[0].patch_cache_path)
            calls = encoder.calls
            self._extract(root, encoder, [example])
            self.assertGreater(encoder.calls, calls)
            calls = encoder.calls
            self._extract(root, encoder, [example], version="texture-v2")
            self.assertGreater(encoder.calls, calls)
            calls = encoder.calls
            self._extract(root, encoder, [example], revision="different")
            self.assertGreater(encoder.calls, calls)

    def test_nonfinite_patch_output_is_not_published(self) -> None:
        with self.subTest("nonfinite"):
            root = self._root()
            image = root / "image.png"
            Image.new("RGB", (16, 16)).save(image)
            encoder = _Encoder(nonfinite=True)
            with self.assertRaises(ValueError):
                self._extract(root, encoder, [_example(image, sample_id="sample")])
            self.assertFalse(any((root / "patches").rglob("*.pt")))


if __name__ == "__main__":
    unittest.main()
