"""Stable prediction records shared by training and evaluation."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


PREDICTION_FIELDS = [
    "sample_id",
    "source_id",
    "parent_id",
    "split",
    "label",
    "logit",
    "probability",
    "checkpoint",
    "seed",
    "matching_policy",
    "transform",
    "transform_parameter",
    "dataset_name",
    "generator_name",
    "generator_checkpoint",
    "capture_source",
]

LABEL_TO_TARGET = {"authentic": 0, "ai_generated": 1}
REQUIRED_PREDICTION_FIELDS = {"sample_id", "split", "label", "logit", "probability"}


@dataclass(frozen=True)
class PredictionRecord:
    sample_id: str
    parent_id: str
    split: str
    label: str
    logit: float
    probability: float
    checkpoint: str
    seed: int
    matching_policy: str
    transform: str = "clean"
    transform_parameter: str = ""
    dataset_name: str = "unknown"
    generator_name: str = "unknown"
    generator_checkpoint: str = "unknown"
    capture_source: str = "unknown"
    source_id: str = ""

    def __post_init__(self) -> None:
        if self.label not in LABEL_TO_TARGET:
            raise ValueError(f"Unsupported binary label: {self.label!r}")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("Probability must be between 0 and 1")
        if not self.sample_id:
            raise ValueError("sample_id is required")

    @property
    def target(self) -> int:
        return LABEL_TO_TARGET[self.label]

    @property
    def evaluation_cell(self) -> str:
        transform = self.transform or "clean"
        if transform == "clean":
            return "clean"
        parameter = self.transform_parameter or "default"
        try:
            encoded = json.loads(parameter)
        except (json.JSONDecodeError, TypeError):
            encoded = None
        if isinstance(encoded, dict):
            cell_id = encoded.get("cell_id")
            if isinstance(cell_id, str) and cell_id.strip():
                return cell_id.strip()
        return f"{transform}:{parameter}"


def _clean_metadata(value: str) -> str:
    value = value.strip()
    return value if value else "unknown"


def prediction_from_row(row: dict[str, str]) -> PredictionRecord:
    """Parse one prediction CSV row with conservative metadata defaults."""

    return PredictionRecord(
        sample_id=row["sample_id"],
        source_id=row.get("source_id", ""),
        parent_id=row.get("parent_id", ""),
        split=row["split"],
        label=row["label"],
        logit=float(row["logit"]),
        probability=float(row["probability"]),
        checkpoint=row.get("checkpoint", ""),
        seed=int(row.get("seed", 0)),
        matching_policy=row.get("matching_policy", "unknown"),
        transform=row.get("transform", "clean") or "clean",
        transform_parameter=row.get("transform_parameter", ""),
        dataset_name=_clean_metadata(row.get("dataset_name", "")),
        generator_name=_clean_metadata(row.get("generator_name", "")),
        generator_checkpoint=_clean_metadata(row.get("generator_checkpoint", "")),
        capture_source=_clean_metadata(row.get("capture_source", "")),
    )


def read_predictions(path: Path) -> list[PredictionRecord]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(REQUIRED_PREDICTION_FIELDS - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Prediction CSV is missing fields: {', '.join(missing)}")
        return [prediction_from_row(row) for row in reader]


def write_predictions(path: Path, records: Iterable[PredictionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
