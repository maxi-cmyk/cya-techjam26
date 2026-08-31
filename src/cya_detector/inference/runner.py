"""Orchestrates discovery, validation, C2PA Stage 0, and prediction.

The predictor and the C2PA check are both passed in (dependency injection),
not looked up from a registry or global state.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from cya_detector.inference.contracts import (
    PredictionRecord,
    Predictor,
    RunResult,
    RunSummary,
    ValidationError,
)
from cya_detector.inference.inputs import DiscoveryError, discover_images, load_and_validate_image


class InferenceRunFailure(RuntimeError):
    """Raised for a fatal run failure. Nothing is published when this is raised."""


_C2PA_CLAIM_PREDICTION = 1.0


def run_inference(
    image_dir: Path,
    *,
    predict_probability: Predictor,
    has_verified_ai_generation_claim: Callable[[Path], bool],
    print_progress: Callable[[str], None] | None = None,
) -> RunResult:
    """Run the full synchronous, per-image pipeline over ``image_dir``.

    Raises ``InferenceRunFailure`` for any fatal condition: empty/collided
    discovery, a predictor exception or invalid predictor output, or any
    uncatalogued exception during image loading or the C2PA check.
    """

    progress = print_progress or (lambda _line: None)
    image_dir = Path(image_dir)

    try:
        relative_paths = discover_images(image_dir)
    except DiscoveryError as exc:
        raise InferenceRunFailure(str(exc)) from exc

    predictions: list[PredictionRecord] = []
    errors: list[ValidationError] = []
    for relative_path in relative_paths:
        absolute_path = image_dir / relative_path
        loaded = load_and_validate_image(absolute_path, relative_image_path=relative_path)
        if isinstance(loaded, ValidationError):
            errors.append(loaded)
            progress(f"invalid   {relative_path}  {loaded.code}")
            continue

        image: Image.Image = loaded
        try:
            if has_verified_ai_generation_claim(absolute_path):
                prediction = _C2PA_CLAIM_PREDICTION
            else:
                prediction = predict_probability(image)
        except Exception as exc:  # predictor exceptions are fatal, not per-image errors
            raise InferenceRunFailure(
                f"Predictor failed on {relative_path}: {exc}"
            ) from exc
        finally:
            image.close()

        if (
            isinstance(prediction, bool)
            or not isinstance(prediction, (int, float))
            or not math.isfinite(prediction)
            or not 0.0 <= prediction <= 1.0
        ):
            raise InferenceRunFailure(
                f"Predictor returned an invalid probability for {relative_path}: {prediction!r}"
            )

        predictions.append(PredictionRecord(image_path=relative_path, pred=float(prediction)))
        progress(f"predicted {relative_path}")

    summary = RunSummary(
        discovered=len(relative_paths), predicted=len(predictions), invalid=len(errors)
    )
    return RunResult(
        predictions=tuple(predictions), errors=tuple(errors), summary=summary
    )
