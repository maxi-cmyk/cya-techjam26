from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from cya_detector.models.rine import (
    build_rine_auxiliary_fusion,
    build_rine_head,
    validate_rine_layers,
)


class RineTests(unittest.TestCase):
    def test_layer_contract(self) -> None:
        self.assertEqual(validate_rine_layers([6, 12, 18, 24], layer_count=24), (6, 12, 18, 24))
        with self.assertRaises(ValueError):
            validate_rine_layers([12, 6], layer_count=24)
        with self.assertRaises(ValueError):
            validate_rine_layers([6, 25], layer_count=24)

    def test_fusion_shape_and_importance_weights(self) -> None:
        import torch

        model = build_rine_head(layer_count=4, hidden_dimension=8)
        output = model(torch.randn(3, 4, 8))
        self.assertEqual(tuple(output.shape), (3, 1))
        weights = model.importance_weights()
        self.assertEqual(len(weights), 4)
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_fusion_rejects_wrong_shape(self) -> None:
        import torch

        model = build_rine_head(layer_count=4, hidden_dimension=8)
        with self.assertRaises(ValueError):
            model(torch.randn(3, 8))

    def test_auxiliary_fusion_freezes_global_rine(self) -> None:
        import torch

        global_model = build_rine_head(layer_count=4, hidden_dimension=8)
        model = build_rine_auxiliary_fusion(
            global_model=global_model,
            auxiliary_dimension=5,
        )
        model.train()
        output = model(torch.randn(3, 4, 8), torch.randn(3, 5))

        self.assertEqual(tuple(output.shape), (3, 1))
        self.assertFalse(model.global_model.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.global_model.parameters())
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in model.auxiliary_projection.parameters())
        )

    def test_tiny_clip_intermediate_extraction(self) -> None:
        import torch
        from PIL import Image
        from transformers import (
            CLIPImageProcessor,
            CLIPVisionConfig,
            CLIPVisionModelWithProjection,
        )

        from cya_detector.data.dataset import ManifestExample
        from cya_detector.models.clip_baseline import LoadedClip
        from cya_detector.training.rine_stage_b import extract_rine_features

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.png"
            Image.new("RGB", (32, 32), (100, 120, 140)).save(image_path)
            config = CLIPVisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=4,
                num_attention_heads=4,
                image_size=32,
                patch_size=16,
                projection_dim=16,
            )
            loaded = LoadedClip(
                model=CLIPVisionModelWithProjection(config).eval(),
                processor=CLIPImageProcessor(
                    size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}
                ),
                identifier="fixture",
                requested_revision="commit",
                resolved_revision="commit",
                embedding_dimension=16,
            )
            example = ManifestExample(
                sample_id="sample",
                source_id="source",
                parent_id="parent",
                image_path=image_path,
                sha256="abc",
                label="authentic",
                split="seed_train",
                image_view="matched_clean",
                transform="clean",
                transform_parameter="",
                metadata={
                    "dataset_name": "fixture",
                    "generator_name": "unknown",
                    "generator_checkpoint": "unknown",
                    "capture_source": "unknown",
                },
            )
            rows, report = extract_rine_features(
                loaded_clip=loaded,
                examples=[example],
                cache_root=root / "cache",
                matching_policy="fixed_q96",
                preprocessing_version="v1",
                representation_version="rine-v1",
                layers=[1, 2, 3, 4],
                batch_size=1,
                device="cpu",
            )
            tensor = torch.load(rows[0].cache_path, weights_only=True)
            self.assertEqual(tuple(tensor.shape), (4, 32))
            self.assertEqual(report["extracted_count"], 1)


if __name__ == "__main__":
    unittest.main()
