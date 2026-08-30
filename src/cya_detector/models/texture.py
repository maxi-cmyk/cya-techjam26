"""Learnable texture-model heads over global and patch-level features."""

from __future__ import annotations

from typing import Any

from cya_detector.models.clip_baseline import require_ml_dependencies


TEXTURE_VARIANTS = ("global_only", "local_only", "global_local")


def masked_patch_weights(scores: Any, mask: Any) -> Any:
    """Return a normalized patch distribution with absent patches fixed at zero."""

    torch, _, _ = require_ml_dependencies()
    if scores.ndim != 2 or mask.ndim != 2 or scores.shape != mask.shape:
        raise ValueError("Scores and patch mask must have matching shape [batch, patch_count]")
    if mask.dtype != torch.bool:
        raise ValueError("Patch mask must have bool dtype")
    if not mask.any(dim=1).all():
        raise ValueError("Every sample requires at least one patch")

    masked_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked_scores, dim=1) * mask.to(dtype=scores.dtype)
    return weights / weights.sum(dim=1, keepdim=True)


def build_texture_head(
    *,
    variant: str,
    layer_count: int,
    global_dimension: int,
    patch_dimension: int,
    fusion_dimension: int,
) -> Any:
    """Build a texture classifier for one configured feature-fusion variant."""

    torch, _, _ = require_ml_dependencies()
    if variant not in TEXTURE_VARIANTS:
        raise ValueError(f"Unknown texture variant: {variant}")
    if min(layer_count, global_dimension, patch_dimension, fusion_dimension) <= 0:
        raise ValueError("Texture dimensions must be positive")

    class TextureHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer_logits = torch.nn.Parameter(torch.zeros(layer_count))
            if variant != "global_only":
                self.patch_score_projection = torch.nn.Linear(patch_dimension, fusion_dimension)
                self.patch_score_classifier = torch.nn.Linear(fusion_dimension, 1)
            if variant == "global_only":
                self.classifier = torch.nn.Linear(global_dimension, 1)
            elif variant == "local_only":
                self.classifier = torch.nn.Linear(patch_dimension, 1)
            else:
                self.global_projection = torch.nn.Linear(global_dimension, fusion_dimension)
                self.local_projection = torch.nn.Linear(patch_dimension, fusion_dimension)
                self.classifier = torch.nn.Linear(2 * fusion_dimension, 1)

        @staticmethod
        def _validate_inputs(global_features: Any, patch_features: Any, patch_mask: Any) -> None:
            if global_features.ndim != 3 or global_features.shape[1:] != (layer_count, global_dimension):
                raise ValueError("Global features must have shape [batch, layers, global_dimension]")
            if patch_features.ndim != 3 or patch_features.shape[2] != patch_dimension:
                raise ValueError("Patch features must have shape [batch, patch_count, patch_dimension]")
            if (
                patch_mask.ndim != 2
                or patch_mask.dtype != torch.bool
                or patch_mask.shape != patch_features.shape[:2]
                or patch_features.shape[0] != global_features.shape[0]
            ):
                raise ValueError("Patch mask must have shape [batch, patch_count] and bool dtype")
            if not patch_mask.any(dim=1).all():
                raise ValueError("Every sample requires at least one patch")

        def global_importance_weights(self) -> list[float]:
            return torch.softmax(self.layer_logits.detach().cpu(), dim=0).tolist()

        def _global_features(self, global_features: Any) -> Any:
            normalized = torch.nn.functional.layer_norm(global_features, (global_dimension,))
            weights = torch.softmax(self.layer_logits, dim=0)
            return (normalized * weights.view(1, layer_count, 1)).sum(dim=1)

        def attention_weights(self, patch_features: Any, patch_mask: Any) -> Any:
            if patch_features.ndim != 3 or patch_features.shape[2] != patch_dimension:
                raise ValueError("Patch features must have shape [batch, patch_count, patch_dimension]")
            if patch_mask.ndim != 2 or patch_mask.dtype != torch.bool or patch_mask.shape != patch_features.shape[:2]:
                raise ValueError("Patch mask must have shape [batch, patch_count] and bool dtype")
            if variant == "global_only":
                scores = torch.zeros(patch_mask.shape, device=patch_features.device, dtype=patch_features.dtype)
            else:
                scores = self.patch_score_classifier(
                    torch.tanh(self.patch_score_projection(patch_features))
                ).squeeze(-1)
            return masked_patch_weights(scores, patch_mask)

        def _local_features(self, patch_features: Any, patch_mask: Any) -> Any:
            weights = self.attention_weights(patch_features, patch_mask)
            return (patch_features * weights.unsqueeze(-1)).sum(dim=1)

        def forward(self, global_features: Any, patch_features: Any, patch_mask: Any) -> Any:
            self._validate_inputs(global_features, patch_features, patch_mask)
            global_vector = self._global_features(global_features)
            if variant == "global_only":
                return self.classifier(global_vector)
            local_vector = self._local_features(patch_features, patch_mask)
            if variant == "local_only":
                return self.classifier(local_vector)
            global_fused = torch.nn.functional.gelu(self.global_projection(global_vector))
            local_fused = torch.nn.functional.gelu(self.local_projection(local_vector))
            return self.classifier(torch.cat((global_fused, local_fused), dim=1))

    return TextureHead()
