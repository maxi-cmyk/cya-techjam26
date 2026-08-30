"""Manifest extraction and frequency-only Stage 1 baselines."""

from __future__ import annotations

import csv
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from cya_detector.data.manifest import read_manifest
from cya_detector.evaluation.metrics import evaluate_predictions
from cya_detector.features.frequency import (
    FrequencyFeatureResult,
    extract_frequency_features,
    frequency_cache_key,
)
from cya_detector.predictions import PredictionRecord, write_predictions


BASE_FIELDS = [
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
    "width",
    "height",
    "file_size",
    "normalization_quality",
    "cache_key",
    "feature_valid",
    "feature_error",
    "analysis_width",
    "analysis_height",
]


def _cache_path(cache_root: Path, key: str) -> Path:
    return cache_root / key[:2] / f"{key}.json"


def _write_cached_result(path: Path, result: FrequencyFeatureResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(
            {
                "names": result.names,
                "values": result.values.tolist(),
                "families": result.families,
                "metadata": result.metadata,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_cached_result(path: Path) -> FrequencyFeatureResult:
    value = json.loads(path.read_text(encoding="utf-8"))
    return FrequencyFeatureResult(
        names=tuple(value["names"]),
        values=np.asarray(value["values"], dtype=np.float64),
        families=tuple(value["families"]),
        metadata=value["metadata"],
    )


def extract_frequency_manifest(
    *,
    manifest_path: Path,
    output_path: Path,
    report_path: Path,
    cache_root: Path,
    matching_policy: str,
    configuration: dict[str, Any],
    workers: int = 4,
) -> dict[str, Any]:
    """Extract train/selection features without reading self-train or final-test rows."""

    extractor_version = configuration["extractor_version"]
    rows = [
        row
        for row in read_manifest(manifest_path)
        if row.get("split") in {"seed_train", "selection_val"}
    ]
    if not rows:
        raise ValueError("No seed_train or selection_val rows found")
    if workers <= 0:
        raise ValueError("workers must be positive")
    keys = {
        row["sample_id"]: frequency_cache_key(
            image_sha256=row["sha256"],
            extractor_version=extractor_version,
            configuration=configuration,
        )
        for row in rows
    }
    cache_locks = {key: threading.Lock() for key in set(keys.values())}

    def process(row: dict[str, str]) -> tuple[dict[str, str], FrequencyFeatureResult | None, str]:
        key = keys[row["sample_id"]]
        path = _cache_path(cache_root, key)
        try:
            with cache_locks[key]:
                result = (
                    _read_cached_result(path)
                    if path.is_file()
                    else extract_frequency_features(
                        Path(row["image_path"]),
                        radial_bins=configuration["radial_bins"],
                        angular_bins=configuration["angular_bins"],
                        dct_bins=configuration["dct_bins"],
                        phase_bins=configuration["phase_bins"],
                        max_analysis_size=configuration["max_analysis_size"],
                    )
                )
                if not path.is_file():
                    _write_cached_result(path, result)
            return row, result, key
        except Exception as exc:
            return row, None, f"{key}|{type(exc).__name__}: {exc}"

    processed: list[tuple[dict[str, str], FrequencyFeatureResult | None, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process, row) for row in rows]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Frequency", unit="image"
        ):
            processed.append(future.result())
    processed.sort(key=lambda item: item[0]["sample_id"])

    reference = next((result for _, result, _ in processed if result is not None), None)
    if reference is None:
        raise ValueError("Frequency extraction failed for every image")
    feature_names = list(reference.names)
    feature_families = dict(zip(reference.names, reference.families, strict=True))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.csv")
    valid_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*BASE_FIELDS, *feature_names])
        writer.writeheader()
        for source, result, key_or_error in processed:
            valid = result is not None
            if valid and result.names != reference.names:
                raise ValueError("Frequency feature schema changed within one extraction run")
            key, _, error = key_or_error.partition("|")
            record = {field: source.get(field, "") for field in BASE_FIELDS}
            record.update(
                {
                    "matching_policy": matching_policy,
                    "cache_key": key,
                    "feature_valid": str(valid).lower(),
                    "feature_error": error,
                    "analysis_width": result.metadata["analysis_width"] if result else "",
                    "analysis_height": result.metadata["analysis_height"] if result else "",
                }
            )
            if result:
                record.update(result.as_dict())
                valid_counts[f"{source['split']}:{source['label']}"] += 1
            else:
                error_counts[error.split(":", 1)[0] or "unknown"] += 1
            writer.writerow(record)
    temporary.replace(output_path)
    report = {
        "manifest": str(manifest_path.resolve()),
        "output": str(output_path.resolve()),
        "matching_policy": matching_policy,
        "extractor_version": extractor_version,
        "configuration": configuration,
        "row_count": len(processed),
        "valid_count": sum(valid_counts.values()),
        "invalid_count": len(processed) - sum(valid_counts.values()),
        "valid_counts": dict(sorted(valid_counts.items())),
        "error_counts": dict(sorted(error_counts.items())),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "feature_families": feature_families,
        "final_test_read": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _load_feature_table(path: Path) -> tuple[list[dict[str, str]], list[str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        feature_names = [name for name in reader.fieldnames or [] if name not in BASE_FIELDS]
    if not rows or not feature_names:
        raise ValueError("Frequency feature table is empty")
    if any(row["feature_valid"] != "true" for row in rows):
        raise ValueError("Invalid frequency rows must be resolved before training")
    family_lookup: dict[str, str] = {}
    for name in feature_names:
        if name.startswith("phase_"):
            family_lookup[name] = "phase"
        elif name.startswith(("residual_", "neighbor_", "pixel_")):
            family_lookup[name] = "residual"
        else:
            family_lookup[name] = "magnitude"
    return rows, feature_names, family_lookup


def _nuisance_correlations(
    rows: list[dict[str, str]], feature_names: list[str]
) -> list[dict[str, float | str]]:
    features = np.asarray([[float(row[name]) for name in feature_names] for row in rows])
    nuisances = {
        "log_file_size": np.log1p([float(row.get("file_size") or 0) for row in rows]),
        "log_width": np.log1p([float(row.get("width") or 0) for row in rows]),
        "log_height": np.log1p([float(row.get("height") or 0) for row in rows]),
        "normalization_quality": np.asarray(
            [float(row.get("normalization_quality") or 0) for row in rows]
        ),
    }
    correlations: list[dict[str, float | str]] = []
    for nuisance_name, nuisance in nuisances.items():
        if np.std(nuisance) < 1e-12:
            continue
        for index, feature_name in enumerate(feature_names):
            feature = features[:, index]
            correlation = (
                float(np.corrcoef(feature, nuisance)[0, 1])
                if np.std(feature) >= 1e-12
                else 0.0
            )
            correlations.append(
                {
                    "feature": feature_name,
                    "nuisance": nuisance_name,
                    "correlation": correlation,
                    "absolute_correlation": abs(correlation),
                }
            )
    return sorted(correlations, key=lambda row: -float(row["absolute_correlation"]))[:20]


def train_frequency_baseline(
    *,
    feature_table: Path,
    output_directory: Path,
    variant: str,
    seed: int,
    threshold: float,
    early_exit_enabled: bool,
) -> dict[str, Any]:
    """Fit scaling on seed_train only and evaluate a frequency-only linear baseline."""

    if early_exit_enabled:
        raise ValueError("The Stage 1 early exit is forbidden during baseline training")
    rows, all_features, families = _load_feature_table(feature_table)
    if variant == "magnitude":
        feature_names = [name for name in all_features if families[name] != "phase"]
    elif variant == "magnitude_phase":
        feature_names = all_features
    else:
        raise ValueError(f"Unsupported frequency variant: {variant}")
    train_rows = [row for row in rows if row["split"] == "seed_train"]
    selection_rows = [row for row in rows if row["split"] == "selection_val"]
    if not train_rows or not selection_rows:
        raise ValueError("Frequency training requires seed_train and selection_val")
    x_train = np.asarray([[float(row[name]) for name in feature_names] for row in train_rows])
    y_train = np.asarray([int(row["label"] == "ai_generated") for row in train_rows])
    x_selection = np.asarray(
        [[float(row[name]) for name in feature_names] for row in selection_rows]
    )
    scaler = StandardScaler().fit(x_train)
    classifier = LogisticRegression(max_iter=2000, random_state=seed).fit(
        scaler.transform(x_train), y_train
    )
    probabilities = classifier.predict_proba(scaler.transform(x_selection))[:, 1]
    logits = classifier.decision_function(scaler.transform(x_selection))
    predictions = [
        PredictionRecord(
            sample_id=row["sample_id"],
            source_id=row["source_id"],
            parent_id=row["parent_id"],
            split=row["split"],
            label=row["label"],
            logit=float(logit),
            probability=float(probability),
            checkpoint=f"frequency_{variant}",
            seed=seed,
            matching_policy=row["matching_policy"],
            transform=row.get("transform") or "clean",
            transform_parameter=row.get("transform_parameter", ""),
            dataset_name=row.get("dataset_name") or "unknown",
            generator_name=row.get("generator_name") or "unknown",
            generator_checkpoint=row.get("generator_checkpoint") or "unknown",
            capture_source=row.get("capture_source") or "unknown",
        )
        for row, logit, probability in zip(
            selection_rows, logits, probabilities, strict=True
        )
    ]
    metrics = evaluate_predictions(predictions, threshold=threshold)
    output_directory.mkdir(parents=True, exist_ok=True)
    write_predictions(output_directory / "selection_predictions.csv", predictions)
    joblib.dump(
        {
            "variant": variant,
            "feature_names": feature_names,
            "scaler": scaler,
            "classifier": classifier,
        },
        output_directory / "model.joblib",
    )
    scaler_report = {
        "fit_split": "seed_train",
        "feature_names": feature_names,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    (output_directory / "scaler.json").write_text(
        json.dumps(scaler_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "variant": variant,
        "seed": seed,
        "feature_count": len(feature_names),
        "train_count": len(train_rows),
        "selection_count": len(selection_rows),
        "metrics": metrics,
        "nuisance_overlap_top20": _nuisance_correlations(train_rows, feature_names),
        "stage1_early_exit_enabled": False,
        "final_test_read": False,
        "coefficient_l2_norm": float(np.linalg.norm(classifier.coef_)),
    }
    (output_directory / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
