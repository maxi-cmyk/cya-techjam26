from __future__ import annotations

import unittest

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
