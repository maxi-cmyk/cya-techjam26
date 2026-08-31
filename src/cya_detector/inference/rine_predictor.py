"""The real controlled-RINE seed-42 predictor adapter (Task 10B).

Frozen CLIP-ViT-L/14-336 intermediate-layer CLS representations, fused by the
retained controlled-RINE head (layer_logits softmax weighting + one linear
classifier). This is the model selected in Task 9/10B: seed 42, the highest
locked score of the three retained seeds (99.85%).

No temperature calibration is applied (T=1, i.e. raw sigmoid probabilities).
Fitting one on the checkpoint's clean selection_val logits produced a
degenerate result (the optimizer drove toward the search bound) because that
165-row set has zero errors, so NLL minimization has no overconfidence to
penalize and would only crush already-informative probabilities toward 0/1.
See docs/planning/nextSteps.md for the full writeup of that finding.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image

from cya_detector.models.clip_baseline import load_frozen_clip, require_ml_dependencies
from cya_detector.models.rine import build_rine_head

MODEL_IDENTIFIER = "openai/clip-vit-large-patch14-336"
RESOLVED_REVISION = "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
RINE_LAYERS = (6, 12, 18, 24)
CHECKPOINT_SEED = 42


class RinePredictorError(RuntimeError):
    """Raised when the checkpoint or resolved model do not match the pinned contract."""


class RinePredictor:
    """Loads once; ``__call__`` scores one image at a time (the Predictor protocol)."""

    def __init__(self, *, checkpoint_path: Path, device: str = "cpu") -> None:
        torch, _, _ = require_ml_dependencies()
        self._torch = torch
        self._device = device

        loaded = load_frozen_clip(MODEL_IDENTIFIER, revision=RESOLVED_REVISION, device=device)
        if loaded.resolved_revision != RESOLVED_REVISION:
            raise RinePredictorError(
                f"Resolved CLIP revision {loaded.resolved_revision!r} does not match "
                f"the pinned revision {RESOLVED_REVISION!r} this checkpoint was trained against"
            )
        self._loaded_clip = loaded

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("stage") != "controlled_rine_robustness"
            or checkpoint.get("seed") != CHECKPOINT_SEED
            or checkpoint.get("matching_policy") != "fixed_q96"
            or tuple(checkpoint.get("layers", ())) != RINE_LAYERS
        ):
            raise RinePredictorError(f"Unexpected checkpoint identity: {checkpoint_path}")
        state = checkpoint.get("model_state_dict")
        if not isinstance(state, dict):
            raise RinePredictorError(f"Checkpoint has no model_state_dict: {checkpoint_path}")
        layer_count = int(state["layer_logits"].shape[0])
        hidden_dimension = int(state["classifier.weight"].shape[1])
        if layer_count != len(RINE_LAYERS):
            raise RinePredictorError("Checkpoint layer count does not match the pinned RINE layers")

        head = build_rine_head(layer_count=layer_count, hidden_dimension=hidden_dimension)
        head.load_state_dict(state, strict=True)
        head.eval()
        for parameter in head.parameters():
            parameter.requires_grad_(False)
        head.to(device)
        self._head = head

    def __call__(self, image: Image.Image) -> float:
        torch = self._torch
        pixel_values = self._loaded_clip.processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].to(self._device)
        with torch.inference_mode():
            outputs = self._loaded_clip.model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True,
            )
            features = torch.stack(
                [outputs.hidden_states[layer][:, 0, :] for layer in RINE_LAYERS], dim=1,
            ).float()
            logit = self._head(features).squeeze()
            probability = torch.sigmoid(logit)
        value = float(probability.item())
        if not math.isfinite(value):
            raise RinePredictorError("RINE predictor produced a non-finite probability")
        return value
