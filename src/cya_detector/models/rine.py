"""RINE-style frozen-CLIP intermediate-layer fusion."""

from __future__ import annotations

from typing import Any

from cya_detector.models.clip_baseline import require_ml_dependencies


def validate_rine_layers(layers: list[int] | tuple[int, ...], *, layer_count: int) -> tuple[int, ...]:
    selected = tuple(int(layer) for layer in layers)
    if not selected:
        raise ValueError("At least one RINE layer is required")
    if tuple(sorted(set(selected))) != selected:
        raise ValueError("RINE layers must be unique and strictly increasing")
    if selected[0] < 1 or selected[-1] > layer_count:
        raise ValueError(f"RINE layers must be between 1 and {layer_count}")
    return selected


def build_rine_head(*, layer_count: int, hidden_dimension: int) -> Any:
    """Build a learnable layer-importance estimator and one binary head."""

    torch, _, _ = require_ml_dependencies()
    if layer_count <= 0 or hidden_dimension <= 0:
        raise ValueError("RINE dimensions must be positive")

    class RineLayerFusion(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer_logits = torch.nn.Parameter(torch.zeros(layer_count))
            self.classifier = torch.nn.Linear(hidden_dimension, 1)

        def forward(self, layer_features: Any) -> Any:
            if layer_features.ndim != 3:
                raise ValueError("RINE input must have shape [batch, layers, hidden]")
            if layer_features.shape[1:] != (layer_count, hidden_dimension):
                raise ValueError(
                    "Unexpected RINE feature shape: "
                    f"{tuple(layer_features.shape[1:])}, expected "
                    f"{(layer_count, hidden_dimension)}"
                )
            normalized = torch.nn.functional.layer_norm(
                layer_features, (hidden_dimension,)
            )
            weights = torch.softmax(self.layer_logits, dim=0)
            fused = (normalized * weights.view(1, layer_count, 1)).sum(dim=1)
            return self.classifier(fused)

        def importance_weights(self) -> list[float]:
            return torch.softmax(self.layer_logits.detach().cpu(), dim=0).tolist()

    return RineLayerFusion()
