"""The sealed final_test evaluation.

This is the only module in the repository permitted to read final_test rows.
Every other script in this codebase deliberately refuses them. Run this
exactly once, after the model, calibration, and threshold are fully frozen
— it is not resumable, not overwrite-able, and requires an explicit
affirmative flag from the caller in addition to the split filter itself.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from cya_detector.data.manifest import read_manifest, sha256_file, write_json
from cya_detector.evaluation.metrics import binary_metrics
from cya_detector.predictions import PredictionRecord

_FIXED_Q96_SUFFIX = "__matched_clean__fixed_q96"
_SUPPORTED_LABELS = frozenset({"authentic", "ai_generated"})


class FinalTestError(RuntimeError):
    """Raised when the final_test evaluation boundary is violated."""


def _require_final_test_rows(manifest_path: Path) -> list[dict[str, str]]:
    try:
        rows = read_manifest(manifest_path)
    except (OSError, ValueError) as exc:
        raise FinalTestError(f"Could not read the final_test manifest: {manifest_path}") from exc

    final_test_rows = [row for row in rows if row.get("split") == "final_test"]
    if not final_test_rows:
        raise FinalTestError("Manifest contains no final_test rows")
    for row in final_test_rows:
        sample_id = row.get("sample_id", "")
        if not sample_id.endswith(_FIXED_Q96_SUFFIX):
            raise FinalTestError(
                f"final_test row is not a fixed-Q96 matched-clean sample: {sample_id!r}"
            )
        if row.get("image_view") != "matched_clean" or row.get("transform") != "clean":
            raise FinalTestError(f"final_test row is not a direct matched-clean view: {sample_id!r}")
        if row.get("label") not in _SUPPORTED_LABELS:
            raise FinalTestError(f"final_test row has an unsupported label: {sample_id!r}")
        if not row.get("sha256"):
            raise FinalTestError(f"final_test row lacks a recorded SHA-256: {sample_id!r}")
    return final_test_rows


def _probability_to_logit(probability: float) -> float:
    if probability <= 0.0:
        return -math.inf
    if probability >= 1.0:
        return math.inf
    return math.log(probability / (1.0 - probability))


def evaluate_final_test(
    *,
    manifest_path: Path,
    predict_probability: Callable[[Image.Image], float],
    threshold: float,
    output_root: Path,
    checkpoint_identity: dict[str, Any],
    confirm_final_test_read: bool,
) -> dict[str, Any]:
    """Score every final_test row exactly once and publish a hashed report.

    Refuses outright if ``output_root`` already contains a completed result —
    this is intentionally not resumable or overwrite-able. Deleting the prior
    output is a deliberate manual action outside this function's control.
    """

    if not confirm_final_test_read:
        raise FinalTestError(
            "Refusing to read final_test without confirm_final_test_read=True"
        )
    if not 0.0 < threshold < 1.0:
        raise FinalTestError("threshold must be strictly between 0 and 1")

    manifest_path = Path(manifest_path)
    output_root = Path(output_root)
    report_path = output_root / "final_test_report.json"
    if report_path.is_file():
        raise FinalTestError(
            f"A final_test result already exists at {report_path}; refusing to "
            "overwrite. final_test may be evaluated only once."
        )

    rows = _require_final_test_rows(manifest_path)

    predictions: list[PredictionRecord] = []
    for row in rows:
        image_path = Path(row["image_path"])
        if not image_path.is_file():
            raise FinalTestError(f"final_test image is missing: {row['sample_id']}")
        if sha256_file(image_path) != row["sha256"]:
            raise FinalTestError(f"final_test image hash mismatch: {row['sample_id']}")

        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        probability = predict_probability(image)
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0.0 <= probability <= 1.0
        ):
            raise FinalTestError(
                f"Predictor returned an invalid probability for {row['sample_id']}: {probability!r}"
            )

        predictions.append(
            PredictionRecord(
                sample_id=row["sample_id"],
                source_id=row.get("source_id", ""),
                parent_id=row.get("parent_id", ""),
                split="final_test",
                label=row["label"],
                logit=_probability_to_logit(float(probability)),
                probability=float(probability),
                checkpoint=str(checkpoint_identity.get("checkpoint_sha256", "unknown")),
                seed=int(checkpoint_identity.get("seed", 0)),
                matching_policy="fixed_q96",
            )
        )

    metrics = binary_metrics(predictions, threshold=threshold)

    report = {
        "final_test_read": True,
        "sample_count": len(predictions),
        "manifest_sha256": sha256_file(manifest_path),
        "checkpoint_identity": checkpoint_identity,
        "threshold": threshold,
        "metrics": metrics,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    return report
