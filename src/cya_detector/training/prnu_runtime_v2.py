"""Extraction, readiness, and PRNU-only robustness diagnostics for PRNU v2."""

from __future__ import annotations

import csv
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from cya_detector.data.dataset import ManifestExample
from cya_detector.data.manifest import read_manifest
from cya_detector.evaluation.metrics import evaluate_predictions
from cya_detector.features.common import AuxiliaryFeatureResult, auxiliary_cache_key
from cya_detector.features.prnu_runtime_v2 import (
    PRNU_RUNTIME_V2_FEATURES,
    extract_prnu_runtime_v2,
)
from cya_detector.predictions import PredictionRecord, write_predictions
from cya_detector.training.robustness import controlled_epoch_rows
from cya_detector.transforms.benchmark import TransformCell


PRNU_RUNTIME_BASE_FIELDS = (
    "sample_id",
    "source_id",
    "parent_id",
    "image_path",
    "sha256",
    "label",
    "split",
    "matching_policy",
    "transform",
    "transform_parameter",
    "dataset_name",
    "generator_name",
    "generator_checkpoint",
    "capture_source",
    "image_view",
    "feature_valid",
    "feature_error",
)
PRNU_RUNTIME_MASK_FIELDS = (
    "prnu_v2_runtime_eligible",
    "prnu_v2_runtime_valid",
    "prnu_v2_runtime_confidence",
)


@dataclass(frozen=True)
class PrnuRuntimeRow:
    example: ManifestExample
    values: np.ndarray


def _cache_write(path: Path, result: AuxiliaryFeatureResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(
            {
                "names": result.names,
                "values": result.values.tolist(),
                "families": result.families,
                "valid": result.valid,
                "confidence": result.confidence,
                "metadata": result.metadata,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _cache_read(path: Path) -> AuxiliaryFeatureResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return AuxiliaryFeatureResult(
        names=tuple(payload["names"]),
        values=np.asarray(payload["values"], dtype=np.float64),
        families=tuple(payload["families"]),
        valid=payload["valid"],
        confidence=payload["confidence"],
        metadata=payload["metadata"],
    )


def audit_prnu_runtime_rows(
    rows: Sequence[dict[str, str]],
    *,
    crop_size: int,
    minimum_eligibility_rate: float,
    maximum_label_gap: float,
) -> dict[str, Any]:
    """Audit received-image crop support without consulting labels to set eligibility."""

    if not 0.0 <= minimum_eligibility_rate <= 1.0:
        raise ValueError("Minimum PRNU-v2 eligibility rate must be in [0, 1]")
    if not 0.0 <= maximum_label_gap <= 1.0:
        raise ValueError("Maximum PRNU-v2 label gap must be in [0, 1]")
    if any(row.get("split") == "final_test" for row in rows):
        raise ValueError("PRNU-v2 runtime audit refuses final_test rows")

    counts: Counter[tuple[str, str, str]] = Counter()
    failures: list[dict[str, str]] = []
    eligibility: dict[str, bool] = {}
    for row in rows:
        sample_id = row.get("sample_id", "")
        path = Path(row.get("image_path", ""))
        try:
            with Image.open(path) as image:
                eligible = min(image.size) >= crop_size
        except (OSError, ValueError) as exc:
            eligible = False
            failures.append({"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})
        eligibility[sample_id] = eligible
        group = (row.get("split", ""), row.get("label", ""), "eligible" if eligible else "ineligible")
        counts[group] += 1

    groups: list[dict[str, Any]] = []
    ready = not failures
    for split in ("seed_train", "selection_val"):
        rates: dict[str, float] = {}
        for label in ("authentic", "ai_generated"):
            eligible_count = counts[(split, label, "eligible")]
            row_count = eligible_count + counts[(split, label, "ineligible")]
            rate = eligible_count / row_count if row_count else 0.0
            rates[label] = rate
            groups.append(
                {
                    "split": split,
                    "label": label,
                    "row_count": row_count,
                    "eligible_count": eligible_count,
                    "eligibility_rate": rate,
                }
            )
            ready &= row_count > 0 and rate >= minimum_eligibility_rate
        ready &= abs(rates["authentic"] - rates["ai_generated"]) <= maximum_label_gap

    return {
        "crop_size": crop_size,
        "resize_applied": False,
        "exif_transpose_applied": False,
        "eligibility_rule": "received_encoded_image_min_dimension_gte_crop_size",
        "minimum_eligibility_rate": minimum_eligibility_rate,
        "maximum_label_gap": maximum_label_gap,
        "groups": groups,
        "read_failures": failures,
        "ready_for_binary_ablation": bool(ready),
        "final_test_read": False,
        "eligibility_by_sample_id": eligibility,
    }


def extract_prnu_runtime_manifest(
    *,
    manifest_path: Path,
    output_path: Path,
    report_path: Path,
    cache_root: Path,
    matching_policy: str,
    configuration: dict[str, Any],
    workers: int,
    require_ready: bool = True,
) -> dict[str, Any]:
    """Extract PRNU-v2 runtime vectors for clean and independent-transform rows."""

    rows = [
        row
        for row in read_manifest(manifest_path)
        if row.get("split") in {"seed_train", "selection_val", "final_test"}
    ]
    if not rows or workers <= 0:
        raise ValueError("PRNU-v2 runtime extraction requires rows and positive workers")
    readiness = audit_prnu_runtime_rows(
        rows,
        crop_size=int(configuration["crop_size"]),
        minimum_eligibility_rate=float(configuration["minimum_eligibility_rate"]),
        maximum_label_gap=float(configuration["maximum_label_gap"]),
    )
    eligibility = readiness.pop("eligibility_by_sample_id")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if require_ready and not readiness["ready_for_binary_ablation"]:
        report_path.write_text(json.dumps(readiness, indent=2, sort_keys=True) + "\n")
        raise ValueError("PRNU-v2 runtime data-readiness gate failed; see extraction report")

    cache_configuration = {**configuration, "reference_comparison_used": False}
    keys = {
        row["sample_id"]: auxiliary_cache_key(
            image_sha256=row["sha256"],
            extractor_version=configuration["extractor_version"],
            configuration=cache_configuration,
        )
        for row in rows
    }
    locks = {key: threading.Lock() for key in set(keys.values())}

    def process(row: dict[str, str]) -> tuple[dict[str, str], AuxiliaryFeatureResult | None, str]:
        key = keys[row["sample_id"]]
        path = cache_root / key[:2] / f"{key}.json"
        try:
            with locks[key]:
                if path.is_file():
                    result = _cache_read(path)
                else:
                    result = extract_prnu_runtime_v2(
                        Path(row["image_path"]),
                        crop_size=int(configuration["crop_size"]),
                        wavelet=str(configuration["wavelet"]),
                        wavelet_levels=int(configuration["wavelet_levels"]),
                        edge_keep_quantile=float(configuration["edge_keep_quantile"]),
                        block_size=int(configuration["block_size"]),
                    )
                    _cache_write(path, result)
            return row, result, key
        except Exception as exc:
            return row, None, f"{key}|{type(exc).__name__}: {exc}"

    processed: list[tuple[dict[str, str], AuxiliaryFeatureResult | None, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process, row) for row in rows]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="PRNU v2 runtime", unit="image"
        ):
            processed.append(future.result())
    processed.sort(key=lambda item: item[0]["sample_id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.csv")
    fieldnames = [*PRNU_RUNTIME_BASE_FIELDS, *PRNU_RUNTIME_V2_FEATURES, *PRNU_RUNTIME_MASK_FIELDS]
    valid_count = 0
    extraction_failures: list[dict[str, str]] = []
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for source, result, key_or_error in processed:
            key, _, error = key_or_error.partition("|")
            record = {field: source.get(field, "") for field in PRNU_RUNTIME_BASE_FIELDS}
            record.update(
                {
                    "matching_policy": matching_policy,
                    "feature_valid": str(result is not None).lower(),
                    "feature_error": error,
                }
            )
            if result is None:
                values = np.zeros(len(PRNU_RUNTIME_V2_FEATURES), dtype=np.float64)
                runtime_valid = False
                confidence = 0.0
                extraction_failures.append({"sample_id": source["sample_id"], "error": error})
            else:
                values = result.values
                runtime_valid = bool(result.valid["prnu_v2_runtime"])
                confidence = float(result.confidence["prnu_v2_runtime"])
                valid_count += int(runtime_valid)
            record.update(dict(zip(PRNU_RUNTIME_V2_FEATURES, values.tolist(), strict=True)))
            record.update(
                {
                    "prnu_v2_runtime_eligible": float(eligibility[source["sample_id"]]),
                    "prnu_v2_runtime_valid": float(runtime_valid),
                    "prnu_v2_runtime_confidence": confidence,
                }
            )
            writer.writerow(record)
    temporary.replace(output_path)

    report = {
        **readiness,
        "manifest": str(manifest_path.resolve()),
        "output": str(output_path.resolve()),
        "matching_policy": matching_policy,
        "configuration": configuration,
        "row_count": len(processed),
        "runtime_valid_count": valid_count,
        "runtime_invalid_count": len(processed) - valid_count,
        "extraction_failures": extraction_failures,
        "feature_names": [*PRNU_RUNTIME_V2_FEATURES, *PRNU_RUNTIME_MASK_FIELDS],
        "reference_comparison_used": False,
        "device_identity_used": False,
        "camera_authentication_claim": False,
        "final_test_read": False,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if extraction_failures:
        raise ValueError(f"PRNU-v2 runtime extraction failed for {len(extraction_failures)} rows")
    return report


def load_prnu_runtime_rows(
    *,
    table_path: Path,
    examples: Sequence[ManifestExample],
) -> list[PrnuRuntimeRow]:
    with table_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        table_rows = list(reader)
        fields = set(reader.fieldnames or [])
    feature_names = [*PRNU_RUNTIME_V2_FEATURES, *PRNU_RUNTIME_MASK_FIELDS]
    missing_fields = sorted(set(feature_names) - fields)
    if missing_fields:
        raise ValueError(f"PRNU-v2 runtime table is missing: {', '.join(missing_fields)}")
    if any(row.get("split") == "final_test" for row in table_rows):
        raise ValueError("PRNU-v2 runtime training refuses final_test rows")
    by_id = {row["sample_id"]: row for row in table_rows}
    if len(by_id) != len(table_rows):
        raise ValueError("PRNU-v2 runtime table has duplicate sample_id values")
    output: list[PrnuRuntimeRow] = []
    for example in examples:
        row = by_id.get(example.sample_id)
        if row is None:
            raise ValueError(f"Missing PRNU-v2 runtime row for {example.sample_id}")
        if row.get("feature_valid") != "true" or row.get("label") != example.label:
            raise ValueError(f"Invalid or mismatched PRNU-v2 runtime row: {example.sample_id}")
        output.append(
            PrnuRuntimeRow(
                example=example,
                values=np.asarray([float(row[name]) for name in feature_names], dtype=np.float32),
            )
        )
    return output


def train_prnu_runtime_baseline(
    *,
    train_parent_rows: Sequence[PrnuRuntimeRow],
    train_bank_rows: Sequence[PrnuRuntimeRow],
    selection_rows: Sequence[PrnuRuntimeRow],
    cells: Sequence[TransformCell],
    output_directory: Path,
    matching_policy: str,
    seed: int,
    threshold: float,
    sampling_epochs: int = 4,
    epoch_size: int | None = None,
) -> dict[str, Any]:
    """Fit a PRNU-only diagnostic under the controlled clean/transform sampler."""

    if sampling_epochs <= 0 or not train_parent_rows or not selection_rows:
        raise ValueError("PRNU-v2 baseline requires rows and positive sampling epochs")
    resolved_epoch_size = epoch_size or len(train_parent_rows)
    sampled = [
        row
        for epoch in range(sampling_epochs)
        for row in controlled_epoch_rows(
            train_parent_rows,
            train_bank_rows,
            cells,
            epoch_size=resolved_epoch_size,
            project_seed=seed,
            epoch=epoch,
        )
    ]
    x_train = np.stack([row.values for row in sampled])
    y_train = np.asarray([row.example.target for row in sampled], dtype=np.int64)
    x_selection = np.stack([row.values for row in selection_rows])
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(random_state=seed, max_iter=2000, class_weight="balanced")
    model.fit(scaler.transform(x_train), y_train)
    logits = model.decision_function(scaler.transform(x_selection))
    probabilities = model.predict_proba(scaler.transform(x_selection))[:, 1]
    predictions = [
        PredictionRecord(
            sample_id=row.example.sample_id,
            source_id=row.example.source_id,
            parent_id=row.example.parent_id,
            split=row.example.split,
            label=row.example.label,
            logit=float(logit),
            probability=float(probability),
            checkpoint="prnu_v2_runtime_logistic",
            seed=seed,
            matching_policy=matching_policy,
            transform=row.example.transform,
            transform_parameter=row.example.transform_parameter,
            **row.example.metadata,
        )
        for row, logit, probability in zip(selection_rows, logits, probabilities, strict=True)
    ]
    metrics = evaluate_predictions(predictions, threshold=threshold)
    output_directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_directory / "model.joblib")
    joblib.dump(scaler, output_directory / "scaler.joblib")
    write_predictions(output_directory / "best_50_50_predictions.csv", predictions)
    summary = {
        "stage": "prnu_v2_runtime_binary_diagnostic",
        "seed": seed,
        "matching_policy": matching_policy,
        "sampling_epochs": sampling_epochs,
        "epoch_size": resolved_epoch_size,
        "training_row_count": len(sampled),
        "selection_metrics": metrics,
        "reference_comparison_used": False,
        "device_identity_used": False,
        "physical_claim": "reference-free binary diagnostic only",
        "final_test_read": False,
    }
    (output_directory / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
