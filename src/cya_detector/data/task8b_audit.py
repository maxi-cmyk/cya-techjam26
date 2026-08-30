"""Pre-training readiness and nuisance audit for the Task 8B dataset."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cya_detector.data.manifest import sha256_file, write_json


NUMERIC_NUISANCE_FEATURES = ["width", "height", "file_size", "pixel_count", "aspect_ratio"]
CATEGORICAL_NUISANCE_FEATURES = ["format", "original_format"]
EXPECTED_SPLITS = ("seed_train", "selection_val", "heldout_test")


class Task8BReadinessError(ValueError):
    """Raised when a Task 8B manifest cannot be audited safely."""


def _boolean_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        raise Task8BReadinessError(f"Task 8B manifest is missing {name}")
    return frame[name].astype(str).str.lower() == "true"


def _nuisance_pipeline(categorical_features: list[str]) -> Pipeline:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [
                        ("numeric", numeric, NUMERIC_NUISANCE_FEATURES),
                        ("categorical", categorical, categorical_features),
                    ]
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            ),
        ]
    )


def _prepare_nuisance_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for name in ("width", "height", "file_size"):
        prepared[name] = pd.to_numeric(prepared[name], errors="coerce")
    prepared["pixel_count"] = prepared["width"] * prepared["height"]
    prepared["aspect_ratio"] = prepared["width"] / prepared["height"].replace(0, np.nan)
    for name in CATEGORICAL_NUISANCE_FEATURES:
        prepared[name] = prepared[name].fillna("").astype(str)
    return prepared


def _evaluate_nuisance(frame: pd.DataFrame) -> dict[str, Any]:
    prepared = _prepare_nuisance_frame(frame)
    training = prepared[prepared["split"] == "seed_train"]
    if training["label"].nunique() != 2:
        return {"status": "not_evaluated", "reason": "seed_train lacks both labels"}
    # Original container history is useful when auditing immutable sources, but
    # it is not observable by a model reading a normalized derivative. Keep the
    # matched-view gate restricted to properties of the bytes it actually sees.
    is_source_original = prepared["image_view"].eq("source_original").all()
    categorical = CATEGORICAL_NUISANCE_FEATURES if is_source_original else ["format"]
    features = NUMERIC_NUISANCE_FEATURES + categorical
    model = _nuisance_pipeline(categorical)
    model.fit(training[features], (training["label"] == "ai_generated").astype(int))
    evaluations: dict[str, dict[str, float | int]] = {}
    for split in ("selection_val", "heldout_test"):
        evaluation = prepared[prepared["split"] == split]
        if evaluation["label"].nunique() != 2:
            return {"status": "not_evaluated", "reason": f"{split} lacks both labels"}
        labels = (evaluation["label"] == "ai_generated").astype(int)
        probabilities = model.predict_proba(evaluation[features])[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        evaluations[split] = {
            "count": int(len(evaluation)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "roc_auc": float(roc_auc_score(labels, probabilities)),
        }
    return {
        "status": "evaluated",
        "features": features,
        "training_count": int(len(training)),
        "splits": evaluations,
        "maximum_balanced_accuracy": max(
            value["balanced_accuracy"] for value in evaluations.values()
        ),
        "interpretation": "Lower label separability is better; this is a shortcut audit, not detector evidence.",
    }


def audit_task8b_readiness(
    *,
    manifest_path: Path,
    output_path: Path,
    readiness: dict[str, Any],
    minimum_images_per_device: int,
) -> dict[str, Any]:
    """Audit source validity, split leakage, physical coverage, and nuisance shortcuts."""

    frame = pd.read_csv(manifest_path, keep_default_na=False)
    required = {
        "source_id",
        "label",
        "split",
        "dataset_name",
        "license_status",
        "license_verified",
        "eligible_for_split",
        "device_id",
        "camera_model",
        "generator_name",
        "split_group_id",
        "duplicate_group_id",
        "physical_source_status",
        "lens_model",
        "focal_length",
        "content_category",
        "width",
        "height",
        "file_size",
        "format",
        "original_format",
    }
    missing = required - set(frame.columns)
    if missing:
        raise Task8BReadinessError(
            f"Task 8B manifest is missing columns: {', '.join(sorted(missing))}"
        )
    binary = frame[frame["label"].isin(["authentic", "ai_generated"])].copy()
    eligible = binary[_boolean_column(binary, "eligible_for_split")].copy()
    if eligible.empty:
        raise Task8BReadinessError("Task 8B manifest has no eligible rows")

    label_counts = Counter(eligible["label"])
    device_counts = Counter(
        eligible.loc[eligible["dataset_name"] == "premier", "device_id"]
    )
    generators = set(
        eligible.loc[eligible["dataset_name"] == "genimage", "generator_name"]
    ) - {"", "unknown"}
    smallest_label = min(label_counts.values(), default=0)
    largest_label = max(label_counts.values(), default=0)
    label_ratio = largest_label / smallest_label if smallest_label else float("inf")

    split_label_counts = Counter(zip(eligible["split"], eligible["label"], strict=False))
    complete_splits = all(
        split_label_counts[(split, label)] > 0
        for split in EXPECTED_SPLITS
        for label in ("authentic", "ai_generated")
    )
    split_groups: dict[str, set[str]] = defaultdict(set)
    for row in eligible.to_dict("records"):
        split_groups[str(row["split_group_id"])].add(str(row["split"]))
    split_group_overlap_count = sum(len(splits) > 1 for splits in split_groups.values())

    duplicate_groups: dict[str, set[str]] = defaultdict(set)
    for row in eligible.to_dict("records"):
        duplicate_group = str(row["duplicate_group_id"])
        if duplicate_group:
            duplicate_groups[duplicate_group].add(str(row["split"]))
    duplicate_split_overlap_count = sum(
        len(splits) > 1 for splits in duplicate_groups.values()
    )

    allowed_licenses = {
        "premier": "cc-by-sa-4.0",
        "genimage": "cc-by-nc-sa-4.0",
    }
    license_rows_valid = all(
        row["license_status"] == allowed_licenses.get(row["dataset_name"])
        for row in eligible.to_dict("records")
    )
    physical_states_valid = all(
        row["physical_source_status"]
        in {"native_camera", "minimally_processed_camera", "native_generator_export"}
        for row in eligible.to_dict("records")
    )
    undersized_devices = sorted(
        device for device, count in device_counts.items() if count < minimum_images_per_device
    )

    source_checks = {
        "both_labels_present": set(label_counts) == {"authentic", "ai_generated"},
        "minimum_rows_per_label": smallest_label >= readiness["minimum_rows_per_label"],
        "label_count_ratio": label_ratio <= readiness["max_label_count_ratio"],
        "minimum_authentic_devices": len(device_counts)
        >= readiness["minimum_authentic_devices"],
        "minimum_generator_families": len(generators)
        >= readiness["minimum_generator_families"],
        "minimum_images_per_device": not undersized_devices,
        "all_splits_have_both_labels": complete_splits,
        "no_split_group_overlap": split_group_overlap_count == 0,
        "no_duplicate_split_overlap": duplicate_split_overlap_count == 0,
        "no_competition_final_test_rows": not (frame["split"] == "final_test").any(),
        "licenses_verified": bool(_boolean_column(eligible, "license_verified").all()),
        "license_dataset_mapping_valid": license_rows_valid,
        "physical_source_states_valid": physical_states_valid,
    }
    source_ready = all(bool(value) for value in source_checks.values())

    training_devices = Counter(
        eligible.loc[
            (eligible["dataset_name"] == "premier")
            & (eligible["split"] == "seed_train"),
            "device_id",
        ]
    )
    usable_training_devices = {
        device
        for device, count in training_devices.items()
        if count >= minimum_images_per_device
    }
    prnu_reference_ready = (
        source_ready
        and len(usable_training_devices) >= readiness["minimum_prnu_training_devices"]
    )

    authentic = eligible[eligible["dataset_name"] == "premier"]
    lens_known = ~authentic["lens_model"].astype(str).str.lower().isin(["", "unknown"])
    focal_known = ~authentic["focal_length"].astype(str).str.lower().isin(["", "unknown"])
    ca_metadata_fraction = float((lens_known & focal_known).mean()) if len(authentic) else 0.0
    edge_rich = authentic["content_category"].astype(str).str.lower().isin(
        ["edge_rich", "calibration", "calibration_target"]
    )
    ca_edge_rich_fraction = float(edge_rich.mean()) if len(authentic) else 0.0
    ca_ready = (
        source_ready
        and ca_metadata_fraction >= readiness["minimum_ca_metadata_fraction"]
        and ca_edge_rich_fraction >= readiness["minimum_ca_edge_rich_fraction"]
    )

    nuisance = _evaluate_nuisance(eligible)
    nuisance_pass = (
        nuisance.get("status") == "evaluated"
        and nuisance["maximum_balanced_accuracy"]
        <= readiness["max_nuisance_balanced_accuracy"]
    )
    training_ready = source_ready and nuisance_pass

    report = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "thresholds": readiness,
        "row_count": int(len(frame)),
        "eligible_count": int(len(eligible)),
        "label_counts": dict(sorted(label_counts.items())),
        "label_count_ratio": label_ratio,
        "device_count": len(device_counts),
        "generator_families": sorted(generators),
        "split_label_counts": {
            f"{split}:{label}": int(count)
            for (split, label), count in sorted(split_label_counts.items())
        },
        "undersized_devices": undersized_devices,
        "split_group_overlap_count": split_group_overlap_count,
        "duplicate_split_overlap_count": duplicate_split_overlap_count,
        "source_checks": source_checks,
        "source_ready": source_ready,
        "prnu_reference": {
            "training_device_count": len(usable_training_devices),
            "ready": prnu_reference_ready,
            "claim": "controlled reference comparison only; not camera authentication",
        },
        "chromatic_aberration": {
            "lens_and_focal_metadata_fraction": ca_metadata_fraction,
            "edge_rich_fraction": ca_edge_rich_fraction,
            "ready": ca_ready,
            "remaining_gate": "corrected/uncorrected calibration coverage and estimator validation",
        },
        "nuisance": nuisance,
        "nuisance_pass": nuisance_pass,
        "training_ready": training_ready,
        "final_test_read": False,
    }
    write_json(output_path, report)
    return report
