from __future__ import annotations

import math
import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from cya_detector.inference.rine_predictor import (
    CHECKPOINT_SEED,
    RESOLVED_REVISION,
    RINE_LAYERS,
    RinePredictor,
    RinePredictorError,
)
from cya_detector.models.clip_baseline import LoadedClip
from cya_detector.models.rine import build_rine_head


class _FixtureProcessor:
    def __call__(self, *, images, return_tensors: str):
        return {"pixel_values": torch.zeros((1, 3, 8, 8))}


class _FixtureModel:
    """Deterministic stand-in for the frozen CLIP vision tower."""

    def __init__(self, *, hidden_size: int = 3) -> None:
        self.config = SimpleNamespace(num_hidden_layers=24, hidden_size=hidden_size)
        self.hidden_size = hidden_size
        self.calls = 0

    def __call__(self, *, pixel_values, output_hidden_states, return_dict):
        self.calls += 1
        batch = pixel_values.shape[0]
        states = tuple(
            torch.full((batch, 1, self.hidden_size), float(layer))
            for layer in range(25)
        )
        return SimpleNamespace(hidden_states=states)


def _fixture_loaded_clip(*, resolved_revision: str = RESOLVED_REVISION, hidden_size: int = 3) -> LoadedClip:
    return LoadedClip(
        model=_FixtureModel(hidden_size=hidden_size),
        processor=_FixtureProcessor(),
        identifier="fixture/clip",
        requested_revision="main",
        resolved_revision=resolved_revision,
        embedding_dimension=hidden_size,
    )


class RinePredictorTests(unittest.TestCase):
    """The break caught here is a predictor that silently loads a mismatched
    checkpoint/revision, or returns a non-finite/invalid probability."""

    def setUp(self) -> None:
        self.root = Path(".tmp") / f"rine-predictor-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def _checkpoint(self, *, hidden_dimension: int = 3, **overrides) -> Path:
        head = build_rine_head(layer_count=len(RINE_LAYERS), hidden_dimension=hidden_dimension)
        payload = {
            "model_state_dict": head.state_dict(),
            "stage": "controlled_rine_robustness",
            "seed": CHECKPOINT_SEED,
            "matching_policy": "fixed_q96",
            "layers": list(RINE_LAYERS),
        }
        payload.update(overrides)
        path = self.root / "best_50_50.pt"
        torch.save(payload, path)
        return path

    def test_rejects_wrong_stage(self) -> None:
        checkpoint = self._checkpoint(stage="something_else")
        with patch(
            "cya_detector.inference.rine_predictor.load_frozen_clip",
            return_value=_fixture_loaded_clip(),
        ), self.assertRaises(RinePredictorError):
            RinePredictor(checkpoint_path=checkpoint)

    def test_rejects_wrong_seed(self) -> None:
        checkpoint = self._checkpoint(seed=43)
        with patch(
            "cya_detector.inference.rine_predictor.load_frozen_clip",
            return_value=_fixture_loaded_clip(),
        ), self.assertRaises(RinePredictorError):
            RinePredictor(checkpoint_path=checkpoint)

    def test_rejects_wrong_matching_policy(self) -> None:
        checkpoint = self._checkpoint(matching_policy="uniform_q95_q100")
        with patch(
            "cya_detector.inference.rine_predictor.load_frozen_clip",
            return_value=_fixture_loaded_clip(),
        ), self.assertRaises(RinePredictorError):
            RinePredictor(checkpoint_path=checkpoint)

    def test_rejects_wrong_layers(self) -> None:
        checkpoint = self._checkpoint(layers=[1, 2, 3, 4])
        with patch(
            "cya_detector.inference.rine_predictor.load_frozen_clip",
            return_value=_fixture_loaded_clip(),
        ), self.assertRaises(RinePredictorError):
            RinePredictor(checkpoint_path=checkpoint)

    def test_rejects_missing_model_state_dict(self) -> None:
        checkpoint = self.root / "best_50_50.pt"
        torch.save(
            {
                "stage": "controlled_rine_robustness", "seed": CHECKPOINT_SEED,
                "matching_policy": "fixed_q96", "layers": list(RINE_LAYERS),
            },
            checkpoint,
        )
        with patch(
            "cya_detector.inference.rine_predictor.load_frozen_clip",
            return_value=_fixture_loaded_clip(),
        ), self.assertRaises(RinePredictorError):
            RinePredictor(checkpoint_path=checkpoint)

    def test_rejects_a_resolved_revision_mismatch(self) -> None:
        checkpoint = self._checkpoint()
        with patch(
            "cya_detector.inference.rine_predictor.load_frozen_clip",
            return_value=_fixture_loaded_clip(resolved_revision="a-different-commit"),
        ), self.assertRaises(RinePredictorError):
            RinePredictor(checkpoint_path=checkpoint)

    def test_call_matches_a_manual_forward_pass(self) -> None:
        checkpoint = self._checkpoint(hidden_dimension=3)
        with patch(
            "cya_detector.inference.rine_predictor.load_frozen_clip",
            return_value=_fixture_loaded_clip(hidden_size=3),
        ):
            predictor = RinePredictor(checkpoint_path=checkpoint)
            probability = predictor(Image.new("RGB", (8, 8)))

        # Reproduce the expected forward pass independently: the fixture CLIP
        # returns hidden_states[layer] filled with float(layer), so the RINE
        # head's input for layers (6, 12, 18, 24) is deterministic regardless
        # of image content.
        features = torch.stack(
            [torch.full((1, 3), float(layer)) for layer in RINE_LAYERS], dim=1
        )
        head = build_rine_head(layer_count=len(RINE_LAYERS), hidden_dimension=3)
        head.load_state_dict(torch.load(checkpoint, weights_only=False)["model_state_dict"])
        head.eval()
        with torch.inference_mode():
            expected = torch.sigmoid(head(features).squeeze()).item()

        self.assertAlmostEqual(probability, expected, places=6)
        self.assertTrue(0.0 <= probability <= 1.0)
        self.assertTrue(math.isfinite(probability))

    def test_call_raises_on_non_finite_output(self) -> None:
        checkpoint = self._checkpoint(hidden_dimension=3)
        with patch(
            "cya_detector.inference.rine_predictor.load_frozen_clip",
            return_value=_fixture_loaded_clip(hidden_size=3),
        ):
            predictor = RinePredictor(checkpoint_path=checkpoint)
            with torch.no_grad():
                predictor._head.classifier.bias.fill_(float("nan"))
            with self.assertRaises(RinePredictorError):
                predictor(Image.new("RGB", (8, 8)))


if __name__ == "__main__":
    unittest.main()
