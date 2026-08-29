"""Measure whether non-semantic file properties predict the binary label."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cya_detector.data.manifest import sha256_file, write_json


NUMERIC_FEATURES = ["width", "height", "file_size", "normalization_quality"]
CATEGORICAL_FEATURES = [
    "format",
    "original_format",
    "normalization_codec",
    "chroma_subsampling",
]


class NuisanceAuditError(ValueError):
    """Raised when a nuisance-only audit cannot be evaluated safely."""


def audit_nuisance_manifest(
    *, manifest_path: Path, output_path: Path, seed: int = 42
) -> dict[str, Any]:
    frame = pd.read_csv(manifest_path)
    frame = frame[frame["label"].isin(["authentic", "ai_generated"])].copy()
    if "eligible_for_split" in frame:
        frame = frame[frame["eligible_for_split"].astype(str).str.lower() == "true"]
    if frame["label"].nunique() != 2:
        raise NuisanceAuditError("Nuisance audit requires both binary labels")

    for feature in NUMERIC_FEATURES:
        frame[feature] = pd.to_numeric(frame.get(feature), errors="coerce")
    frame["pixel_count"] = frame["width"] * frame["height"]
    frame["aspect_ratio"] = frame["width"] / frame["height"].replace(0, np.nan)
    numeric_features = NUMERIC_FEATURES + ["pixel_count", "aspect_ratio"]
    for feature in CATEGORICAL_FEATURES:
        if feature not in frame:
            frame[feature] = ""

    features = frame[numeric_features + CATEGORICAL_FEATURES]
    labels = (frame["label"] == "ai_generated").astype(int)
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=0.3,
        random_state=seed,
        stratify=labels,
    )

    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
            ),
        ]
    )
    model.fit(train_x, train_y)
    probabilities = model.predict_proba(test_x)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    label_stats: dict[str, dict[str, Any]] = defaultdict(dict)
    for label, group in frame.groupby("label"):
        label_stats[label] = {
            "count": int(len(group)),
            "file_size_mean": float(group["file_size"].mean()),
            "file_size_median": float(group["file_size"].median()),
            "width_mean": float(group["width"].mean()),
            "height_mean": float(group["height"].mean()),
            "format_counts": {
                str(key): int(value) for key, value in group["format"].value_counts().items()
            },
        }

    report = {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "seed": seed,
        "sample_count": int(len(frame)),
        "features": numeric_features + CATEGORICAL_FEATURES,
        "test_accuracy": float(accuracy_score(test_y, predictions)),
        "test_balanced_accuracy": float(balanced_accuracy_score(test_y, predictions)),
        "test_roc_auc": float(roc_auc_score(test_y, probabilities)),
        "label_stats": dict(label_stats),
        "interpretation": "Lower separability is better; selection must also consider held-out model results.",
    }
    write_json(output_path, report)
    return report
