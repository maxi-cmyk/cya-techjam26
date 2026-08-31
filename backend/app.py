"""FastAPI backend for the frontend's Predict page.

Thin wrapper only: all discovery/validation/C2PA/prediction logic lives in
src/cya_detector/inference/. This module just accepts uploaded files, stages
them to a temp directory, and calls the same run_inference() the CLI uses,
so the API and the CLI can never silently diverge in behavior.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from cya_detector.inference.c2pa import has_verified_ai_generation_claim
from cya_detector.inference.contracts import Predictor
from cya_detector.inference.runner import InferenceRunFailure, run_inference

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = (
    REPO_ROOT / "artifacts/robustness/train-controlled-rine/seed_42/best_50_50.pt"
)
THRESHOLD = 0.5

app = FastAPI(title="cya-detector inference API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_predictor: Predictor | None = None


def _build_default_predictor() -> Predictor:
    if not DEFAULT_CHECKPOINT_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Checkpoint not found at {DEFAULT_CHECKPOINT_PATH}",
        )
    from cya_detector.inference.rine_predictor import RinePredictor

    return RinePredictor(checkpoint_path=DEFAULT_CHECKPOINT_PATH, device="cpu")


def get_predictor() -> Predictor:
    """Loads the real predictor once and reuses it. Override in tests via
    ``app.dependency_overrides[get_predictor]``."""

    global _predictor
    if _predictor is None:
        _predictor = _build_default_predictor()
    return _predictor


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    files: list[UploadFile] = File(...),
    predictor: Predictor = Depends(get_predictor),
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as staging:
        staging_path = Path(staging)
        original_names: dict[str, str] = {}
        for index, upload in enumerate(files):
            safe_name = Path(upload.filename or f"image_{index}").name or f"image_{index}"
            staged_path = staging_path / f"{index}_{safe_name}"
            contents = await upload.read()
            staged_path.write_bytes(contents)
            original_names[staged_path.name] = upload.filename or safe_name

        try:
            result = run_inference(
                staging_path,
                predict_probability=predictor,
                has_verified_ai_generation_claim=has_verified_ai_generation_claim,
            )
        except InferenceRunFailure as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    predictions = [
        {
            "filename": original_names.get(record.image_path, record.image_path),
            "label": "ai_generated" if record.pred >= THRESHOLD else "authentic",
            "confidence": record.pred if record.pred >= THRESHOLD else 1.0 - record.pred,
        }
        for record in result.predictions
    ]
    errors = [
        {
            "filename": original_names.get(error.image_path, error.image_path),
            "code": error.code,
            "message": error.message,
        }
        for error in result.errors
    ]
    return {"predictions": predictions, "errors": errors}
