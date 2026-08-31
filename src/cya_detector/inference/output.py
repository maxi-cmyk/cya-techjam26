"""Serializes the two governed output artifacts, atomically, on non-fatal
completion. A fatal run never calls this — nothing here runs unless the run
completed (exit 0 or 3)."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from cya_detector.inference.contracts import RunResult

PREDICTIONS_FILENAME = "predictions.json"
REPORT_FILENAME = "report.json"
SCHEMA_VERSION = 1


def build_report(result: RunResult) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "summary": {
            "discovered": result.summary.discovered,
            "predicted": result.summary.predicted,
            "invalid": result.summary.invalid,
        },
        "errors": [
            {"image_path": error.image_path, "code": error.code, "message": error.message}
            for error in result.errors
        ],
    }


def build_predictions(result: RunResult) -> list[dict[str, Any]]:
    return [
        {"image_path": record.image_path, "pred": record.pred} for record in result.predictions
    ]


def _write_temp_then_rename(path: Path, payload: Any) -> None:
    token = uuid.uuid4().hex
    temporary = path.with_name(f".{path.name}.{token}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish(result: RunResult, output_dir: Path) -> int:
    """Write report.json, then predictions.json, both atomically, and return
    the exit code. report.json is renamed into place first so that a caller
    who sees predictions.json existing can trust the whole run completed."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_temp_then_rename(output_dir / REPORT_FILENAME, build_report(result))
    _write_temp_then_rename(output_dir / PREDICTIONS_FILENAME, build_predictions(result))
    return result.exit_code
