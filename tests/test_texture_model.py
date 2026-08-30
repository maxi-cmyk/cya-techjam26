from __future__ import annotations

import re
import unittest


class TextureModelTests(unittest.TestCase):
    @staticmethod
    def inputs():
        import torch

        global_features = torch.randn(2, 4, 8)
        patch_features = torch.randn(2, 4, 6)
        patch_mask = torch.tensor([[True, True, True, True], [True, False, False, False]])
        return global_features, patch_features, patch_mask

    def test_all_variants_return_a_logit_per_sample(self) -> None:
        from cya_detector.models.texture import TEXTURE_VARIANTS, build_texture_head

        global_features, patch_features, patch_mask = self.inputs()
        self.assertEqual(TEXTURE_VARIANTS, ("global_only", "local_only", "global_local"))
        for variant in TEXTURE_VARIANTS:
            model = build_texture_head(
                variant=variant,
                layer_count=4,
                global_dimension=8,
                patch_dimension=6,
                fusion_dimension=5,
            )
            self.assertEqual(tuple(model(global_features, patch_features, patch_mask).shape), (2, 1))

    def test_masked_patch_weights_are_valid_probability_distributions(self) -> None:
        import torch
        from cya_detector.models.texture import masked_patch_weights

        scores = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 8.0, -4.0, 7.0]])
        mask = torch.tensor([[True, True, True, True], [True, False, False, False]])
        weights = masked_patch_weights(scores, mask)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue((weights >= 0).all())
        self.assertTrue((weights[~mask] == 0).all())
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(2)))

    def test_masked_patch_values_do_not_change_local_logits(self) -> None:
        import torch
        from cya_detector.models.texture import build_texture_head

        torch.manual_seed(7)
        global_features, patch_features, patch_mask = self.inputs()
        changed_features = patch_features.clone()
        changed_features[~patch_mask] = 1000000.0
        for variant in ("local_only", "global_local"):
            model = build_texture_head(
                variant=variant,
                layer_count=4,
                global_dimension=8,
                patch_dimension=6,
                fusion_dimension=5,
            )
            self.assertTrue(
                torch.allclose(
                    model(global_features, patch_features, patch_mask),
                    model(global_features, changed_features, patch_mask),
                )
            )

    def test_masked_nonfinite_patch_values_do_not_change_local_logits(self) -> None:
        import torch
        from cya_detector.models.texture import build_texture_head

        torch.manual_seed(7)
        global_features, patch_features, patch_mask = self.inputs()
        changed_features = patch_features.clone()
        changed_features[1, 1] = torch.nan
        changed_features[1, 2] = torch.inf
        changed_features[1, 3] = -torch.inf
        for variant in ("local_only", "global_local"):
            model = build_texture_head(
                variant=variant,
                layer_count=4,
                global_dimension=8,
                patch_dimension=6,
                fusion_dimension=5,
            )
            self.assertTrue(
                torch.allclose(
                    model(global_features, patch_features, patch_mask),
                    model(global_features, changed_features, patch_mask),
                )
            )

    def test_masked_patch_weights_reject_nonfinite_available_scores(self) -> None:
        import torch
        from cya_detector.models.texture import masked_patch_weights

        mask = torch.tensor([[True, True]])
        for score in (torch.nan, torch.inf):
            with self.subTest(score=score):
                with self.assertRaisesRegex(
                    ValueError, "^Patch scores must be finite at available positions$"
                ):
                    masked_patch_weights(torch.tensor([[0.0, score]]), mask)

    def test_all_false_patch_mask_is_rejected(self) -> None:
        import torch
        from cya_detector.models.texture import build_texture_head

        global_features, patch_features, _ = self.inputs()
        model = build_texture_head(
            variant="global_only",
            layer_count=4,
            global_dimension=8,
            patch_dimension=6,
            fusion_dimension=5,
        )
        with self.assertRaisesRegex(ValueError, "^Every sample requires at least one patch$"):
            model(global_features, patch_features, torch.zeros(2, 4, dtype=torch.bool))

    def test_all_inputs_are_shape_validated_for_every_variant(self) -> None:
        import torch
        from cya_detector.models.texture import build_texture_head

        global_features, patch_features, patch_mask = self.inputs()
        bad_inputs = (
            (torch.randn(2, 8), patch_features, patch_mask, "Global features must have shape [batch, layers, global_dimension]"),
            (global_features, torch.randn(2, 6), patch_mask, "Patch features must have shape [batch, patch_count, patch_dimension]"),
            (global_features, patch_features, torch.ones(2, 4), "Patch mask must have shape [batch, patch_count] and bool dtype"),
        )
        for variant in ("global_only", "local_only", "global_local"):
            model = build_texture_head(
                variant=variant,
                layer_count=4,
                global_dimension=8,
                patch_dimension=6,
                fusion_dimension=5,
            )
            for bad_global, bad_patch, bad_mask, message in bad_inputs:
                with self.subTest(variant=variant, message=message):
                    with self.assertRaisesRegex(ValueError, f"^{re.escape(message)}$"):
                        model(bad_global, bad_patch, bad_mask)

    def test_unknown_variant_fails_closed(self) -> None:
        from cya_detector.models.texture import build_texture_head

        with self.assertRaisesRegex(ValueError, "^Unknown texture variant: unknown$"):
            build_texture_head(
                variant="unknown",
                layer_count=4,
                global_dimension=8,
                patch_dimension=6,
                fusion_dimension=5,
            )

    def test_head_has_its_own_trainable_parameters_and_does_not_unfreeze_encoder(self) -> None:
        import torch
        from cya_detector.models.texture import build_texture_head

        encoder = torch.nn.Linear(8, 8)
        encoder.requires_grad_(False)
        model = build_texture_head(
            variant="global_local",
            layer_count=4,
            global_dimension=8,
            patch_dimension=6,
            fusion_dimension=5,
        )
        self.assertGreater(sum(parameter.numel() for parameter in model.parameters()), 0)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))

    def test_attention_and_global_importance_weights_are_available_for_reporting(self) -> None:
        import torch
        from cya_detector.models.texture import build_texture_head

        global_features, patch_features, patch_mask = self.inputs()
        model = build_texture_head(
            variant="global_local",
            layer_count=4,
            global_dimension=8,
            patch_dimension=6,
            fusion_dimension=5,
        )
        weights = model.attention_weights(patch_features, patch_mask)
        self.assertEqual(tuple(weights.shape), (2, 4))
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(2)))
        importance = model.global_importance_weights()
        self.assertEqual(len(importance), 4)
        self.assertAlmostEqual(sum(importance), 1.0)


if __name__ == "__main__":
    unittest.main()
