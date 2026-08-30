"""Extraction and clean feature-only baselines for Stage C auxiliary families."""

from __future__ import annotations

import csv
import hashlib
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
from cya_detector.features.color import extract_color_features
from cya_detector.features.common import AuxiliaryFeatureResult, auxiliary_cache_key
from cya_detector.features.optics import extract_optics_features
from cya_detector.features.prnu import extract_prnu_features
from cya_detector.predictions import PredictionRecord, write_predictions


BASE_FIELDS = [
    "sample_id", "source_id", "parent_id", "image_path", "sha256", "label", "split",
    "matching_policy", "transform", "transform_parameter", "dataset_name",
    "generator_name", "generator_checkpoint", "capture_source", "image_view",
    "license_status", "license_verified", "source_subset", "processing_state",
    "physical_source_status", "device_id", "camera_make", "camera_model",
    "lens_model", "focal_length", "split_group_id",
    "width", "height", "file_size", "normalization_quality", "cache_key",
    "feature_valid", "feature_error", "rgb_valid", "rgb_confidence", "lab_valid",
    "lab_confidence", "prnu_valid", "prnu_confidence", "prnu_eligible",
    "ca_valid", "ca_confidence", "ca_eligible", "radial_distortion_valid",
    "radial_distortion_eligible", "analysis_width", "analysis_height",
]

VARIANT_FAMILIES = {
    "rgb": ("rgb",),
    "lab": ("lab",),
    "rgb_lab": ("rgb", "lab"),
    "prnu": ("prnu",),
    "ca": ("ca",),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def physical_feature_eligible(row: dict[str, str], *, min_dimension: int) -> bool:
    """Apply label-independent native/export provenance and resolution rules."""

    width = int(float(row.get("original_width") or row.get("width") or 0))
    height = int(float(row.get("original_height") or row.get("height") or 0))
    if row.get("image_view") != "source_original" or min(width, height) < min_dimension:
        return False
    task8b_dataset = row.get("dataset_name") in {"premier", "genimage"}
    provenance = row.get("physical_source_status") or row.get("processing_state") or ""
    allowed = {
        "native_camera",
        "minimally_processed_camera",
        "native_generator_export",
    }
    if task8b_dataset:
        return row.get("license_verified") == "true" and provenance in allowed
    return not provenance or provenance in allowed


def _merge_results(results: list[AuxiliaryFeatureResult]) -> AuxiliaryFeatureResult:
    names = tuple(name for result in results for name in result.names)
    values = np.concatenate([result.values for result in results])
    families = tuple(family for result in results for family in result.families)
    valid = {key: value for result in results for key, value in result.valid.items()}
    confidence = {
        key: value for result in results for key, value in result.confidence.items()
    }
    metadata = {key: value for result in results for key, value in result.metadata.items()}
    return AuxiliaryFeatureResult(names, values, families, valid, confidence, metadata)


def _write_cache(path: Path, result: AuxiliaryFeatureResult) -> None:
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


def _read_cache(path: Path) -> AuxiliaryFeatureResult:
    value = json.loads(path.read_text(encoding="utf-8"))
    return AuxiliaryFeatureResult(
        names=tuple(value["names"]), values=np.asarray(value["values"], dtype=np.float64),
        families=tuple(value["families"]), valid=value["valid"],
        confidence=value["confidence"], metadata=value["metadata"],
    )


def extract_auxiliary_manifest(
    *, manifest_path: Path, output_path: Path, report_path: Path, cache_root: Path,
    matching_policy: str, configuration: dict[str, Any], workers: int,
) -> dict[str, Any]:
    rows = [
        row for row in read_manifest(manifest_path)
        if row.get("split") in {"seed_train", "selection_val"}
    ]
    if not rows or workers <= 0:
        raise ValueError("Auxiliary extraction requires rows and positive workers")
    keys = {
        row["sample_id"]: auxiliary_cache_key(
            image_sha256=row["sha256"], extractor_version=configuration["extractor_version"],
            configuration=configuration,
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
                    result = _read_cache(path)
                else:
                    image_path = Path(row["image_path"])
                    result = _merge_results(
                        [
                            extract_color_features(
                                image_path, window_size=configuration["color_window_size"],
                                low_variance_epsilon=configuration["low_variance_epsilon"],
                                max_analysis_size=configuration["max_analysis_size"],
                            ),
                            extract_prnu_features(
                                image_path, block_size=configuration["prnu_block_size"],
                                denoise_sigma=configuration["prnu_denoise_sigma"],
                                min_dimension=configuration["prnu_min_dimension"],
                                max_analysis_size=configuration["max_analysis_size"],
                            ),
                            extract_optics_features(
                                image_path, scale_limit=configuration["ca_scale_limit"],
                                scale_steps=configuration["ca_scale_steps"],
                                min_dimension=configuration["optics_min_dimension"],
                                min_edge_fraction=configuration["ca_min_edge_fraction"],
                                max_analysis_size=configuration["optics_max_analysis_size"],
                            ),
                        ]
                    )
                    _write_cache(path, result)
            return row, result, key
        except Exception as exc:
            return row, None, f"{key}|{type(exc).__name__}: {exc}"

    processed: list[tuple[dict[str, str], AuxiliaryFeatureResult | None, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process, row) for row in rows]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Auxiliary", unit="image"):
            processed.append(future.result())
    processed.sort(key=lambda item: item[0]["sample_id"])
    reference = next((result for _, result, _ in processed if result is not None), None)
    if reference is None:
        raise ValueError("Auxiliary extraction failed for every image")
    feature_names = list(reference.names)
    feature_families = dict(zip(reference.names, reference.families, strict=True))
    counters: Counter[str] = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*BASE_FIELDS, *feature_names])
        writer.writeheader()
        for source, result, key_or_error in processed:
            key, _, error = key_or_error.partition("|")
            record = {field: source.get(field, "") for field in BASE_FIELDS}
            prnu_eligible = physical_feature_eligible(
                source, min_dimension=configuration["prnu_min_dimension"]
            )
            ca_eligible = physical_feature_eligible(
                source, min_dimension=configuration["optics_min_dimension"]
            )
            record.update(
                {
                    "matching_policy": matching_policy, "cache_key": key,
                    "feature_valid": str(result is not None).lower(), "feature_error": error,
                    "prnu_eligible": str(prnu_eligible).lower(),
                    "ca_eligible": str(ca_eligible).lower(),
                    "radial_distortion_eligible": "false",
                }
            )
            if result is not None:
                record.update(result.as_dict())
                for family in ("rgb", "lab", "prnu", "ca", "radial_distortion"):
                    valid = bool(result.valid.get(family, False))
                    confidence = float(result.confidence.get(family, 0.0))
                    record[f"{family}_valid"] = str(valid).lower()
                    if f"{family}_confidence" in BASE_FIELDS:
                        record[f"{family}_confidence"] = confidence
                    counters[f"{source['split']}:{source['label']}:{family}:valid"] += int(valid)
                    counters[f"{source['split']}:{source['label']}:{family}:confidence_sum"] += confidence
                record["analysis_width"] = result.metadata.get("analysis_width", "")
                record["analysis_height"] = result.metadata.get("analysis_height", "")
            counters[f"{source['split']}:{source['label']}:rows"] += 1
            counters[f"{source['split']}:{source['label']}:prnu:eligible"] += int(prnu_eligible)
            counters[f"{source['split']}:{source['label']}:ca:eligible"] += int(ca_eligible)
            writer.writerow(record)
    temporary.replace(output_path)
    family_audit = []
    for split in ("seed_train", "selection_val"):
        for label in ("authentic", "ai_generated"):
            row_count = counters[f"{split}:{label}:rows"]
            for family in ("rgb", "lab", "prnu", "ca"):
                valid_count = counters[f"{split}:{label}:{family}:valid"]
                eligible_count = (
                    counters[f"{split}:{label}:{family}:eligible"]
                    if family in {"prnu", "ca"}
                    else row_count
                )
                family_audit.append(
                    {
                        "split": split, "label": label, "family": family,
                        "row_count": row_count, "eligible_count": eligible_count,
                        "eligibility_rate": eligible_count / max(row_count, 1),
                        "valid_count": valid_count,
                        "validity_rate": valid_count / max(row_count, 1),
                        "confidence_mean": counters[
                            f"{split}:{label}:{family}:confidence_sum"
                        ] / max(row_count, 1),
                    }
                )
    report = {
        "manifest": str(manifest_path.resolve()), "output": str(output_path.resolve()),
        "matching_policy": matching_policy, "extractor_version": configuration["extractor_version"],
        "configuration": configuration, "row_count": len(processed),
        "valid_count": sum(result is not None for _, result, _ in processed),
        "invalid_count": sum(result is None for _, result, _ in processed),
        "coverage_counts": dict(sorted(counters.items())), "feature_count": len(feature_names),
        "family_audit": family_audit,
        "feature_names": feature_names, "feature_families": feature_families,
        "radial_distortion_status": "deferred_until_eligible_line_support_is_available",
        "physical_claim_warning": "PRNU is a single-image coherence proxy; no camera or lens authenticity claim is made.",
        "final_test_read": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _load_table(path: Path) -> tuple[list[dict[str, str]], list[str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        names = [name for name in reader.fieldnames or [] if name not in BASE_FIELDS]
    if not rows or not names or any(row["feature_valid"] != "true" for row in rows):
        raise ValueError("Auxiliary feature table is empty or contains invalid rows")
    families: dict[str, str] = {}
    for name in names:
        families[name] = "rgb" if name.startswith("rgb_") else "lab" if name.startswith("lab_") else "prnu" if name.startswith("prnu_") else "ca"
    return rows, names, families


def train_auxiliary_baseline(
    *, feature_table: Path, output_directory: Path, variant: str, seed: int,
    threshold: float, max_eligibility_gap: float,
) -> dict[str, Any]:
    if variant not in VARIANT_FAMILIES:
        raise ValueError(f"Unsupported auxiliary variant: {variant}")
    rows, all_names, family_lookup = _load_table(feature_table)
    selected_families = VARIANT_FAMILIES[variant]
    feature_names = [name for name in all_names if family_lookup[name] in selected_families]
    train_rows = [row for row in rows if row["split"] == "seed_train"]
    selection_rows = [row for row in rows if row["split"] == "selection_val"]
    physical_family = variant if variant in {"prnu", "ca"} else None
    eligibility_by_label: dict[str, float] = {}
    validity_by_label: dict[str, float] = {}
    if physical_family:
        for label in ("authentic", "ai_generated"):
            label_rows = [row for row in train_rows if row["label"] == label]
            eligibility_by_label[label] = float(np.mean([row[f"{physical_family}_eligible"] == "true" for row in label_rows])) if label_rows else 0.0
            eligible_rows = [
                row for row in label_rows if row[f"{physical_family}_eligible"] == "true"
            ]
            validity_by_label[label] = (
                float(
                    np.mean(
                        [row[f"{physical_family}_valid"] == "true" for row in eligible_rows]
                    )
                )
                if eligible_rows
                else 0.0
            )
        if max(eligibility_by_label.values(), default=0.0) == 0.0:
            raise ValueError(f"No physically eligible seed_train rows for {physical_family}")
        if abs(eligibility_by_label["authentic"] - eligibility_by_label["ai_generated"]) > max_eligibility_gap:
            raise ValueError(f"{physical_family} eligibility gap exceeds {max_eligibility_gap}")
        if abs(validity_by_label["authentic"] - validity_by_label["ai_generated"]) > max_eligibility_gap:
            raise ValueError(f"{physical_family} validity gap exceeds {max_eligibility_gap}")

    def matrix(source_rows: list[dict[str, str]]) -> np.ndarray:
        return np.asarray([[float(row[name]) for name in feature_names] for row in source_rows])

    x_train = matrix(train_rows)
    x_selection = matrix(selection_rows)
    model_feature_names = list(feature_names)
    if physical_family:
        train_supported = np.asarray(
            [
                row[f"{physical_family}_eligible"] == "true"
                and row[f"{physical_family}_valid"] == "true"
                for row in train_rows
            ]
        )
        selection_supported = np.asarray(
            [
                row[f"{physical_family}_eligible"] == "true"
                and row[f"{physical_family}_valid"] == "true"
                for row in selection_rows
            ]
        )
        if not np.any(train_supported):
            raise ValueError(f"No valid and eligible seed_train rows for {physical_family}")
        scaler = StandardScaler().fit(x_train[train_supported])
        normalized_train = np.zeros_like(x_train)
        normalized_selection = np.zeros_like(x_selection)
        normalized_train[train_supported] = scaler.transform(x_train[train_supported])
        normalized_selection[selection_supported] = scaler.transform(
            x_selection[selection_supported]
        )

        def masks(source_rows: list[dict[str, str]]) -> np.ndarray:
            return np.asarray(
                [
                    [
                        float(row[f"{physical_family}_eligible"] == "true"),
                        float(row[f"{physical_family}_valid"] == "true"),
                        float(row[f"{physical_family}_confidence"] or 0.0),
                    ]
                    for row in source_rows
                ],
                dtype=np.float64,
            )

        transformed_train = np.concatenate([normalized_train, masks(train_rows)], axis=1)
        transformed_selection = np.concatenate(
            [normalized_selection, masks(selection_rows)], axis=1
        )
        model_feature_names.extend(
            [
                f"{physical_family}_eligibility_mask",
                f"{physical_family}_validity_mask",
                f"{physical_family}_confidence_mask",
            ]
        )
    else:
        scaler = StandardScaler().fit(x_train)
        transformed_train = scaler.transform(x_train)
        transformed_selection = scaler.transform(x_selection)
    y_train = np.asarray([row["label"] == "ai_generated" for row in train_rows], dtype=np.int64)
    classifier = LogisticRegression(max_iter=2000, random_state=seed).fit(
        transformed_train, y_train
    )
    probabilities = classifier.predict_proba(transformed_selection)[:, 1]
    logits = classifier.decision_function(transformed_selection)
    predictions = [
        PredictionRecord(
            sample_id=row["sample_id"], source_id=row["source_id"], parent_id=row["parent_id"],
            split=row["split"], label=row["label"], logit=float(logit), probability=float(probability),
            checkpoint=f"auxiliary_{variant}", seed=seed, matching_policy=row["matching_policy"],
            transform=row.get("transform") or "clean", transform_parameter=row.get("transform_parameter", ""),
            dataset_name=row.get("dataset_name") or "unknown", generator_name=row.get("generator_name") or "unknown",
            generator_checkpoint=row.get("generator_checkpoint") or "unknown", capture_source=row.get("capture_source") or "unknown",
        )
        for row, logit, probability in zip(selection_rows, logits, probabilities, strict=True)
    ]
    output_directory.mkdir(parents=True, exist_ok=True)
    write_predictions(output_directory / "selection_predictions.csv", predictions)
    joblib.dump(
        {
            "variant": variant, "feature_names": model_feature_names,
            "raw_feature_names": feature_names, "scaler": scaler, "classifier": classifier,
        },
        output_directory / "model.joblib",
    )
    report = {
        "variant": variant, "seed": seed, "feature_count": len(model_feature_names),
        "feature_table_sha256": _sha256_file(feature_table),
        "train_count": len(train_rows), "selection_count": len(selection_rows),
        "metrics": evaluate_predictions(predictions, threshold=threshold),
        "eligibility_by_label": eligibility_by_label,
        "validity_by_label": validity_by_label,
        "final_test_read": False,
        "physical_claim": False,
    }
    (output_directory / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
