"""Training-only multi-image PRNU reference fingerprints for Task 8B."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage
from sklearn.metrics import roc_auc_score

from cya_detector.data.manifest import read_manifest, write_json
from cya_detector.features.prnu import extract_prnu_features


def _reference_residual(image_path: Path, size: int, denoise_sigma: float) -> np.ndarray:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        scale = size / min(width, height)
        resized = image.resize(
            (max(size, round(width * scale)), max(size, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
        left = (resized.width - size) // 2
        top = (resized.height - size) // 2
        rgb = np.asarray(resized.crop((left, top, left + size, top + size)), dtype=np.float32)
    luminance = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]) / 255.0
    residual = luminance - ndimage.gaussian_filter(
        luminance,
        sigma=denoise_sigma,
        mode="reflect",
    )
    return residual / np.maximum(luminance, 0.1)


def build_training_prnu_references(
    *,
    manifest_path: Path,
    output_root: Path,
    report_path: Path,
    minimum_images_per_device: int,
    reference_size: int = 256,
    denoise_sigma: float = 1.0,
) -> dict[str, Any]:
    """Average seed-train residuals by device without reading held-out images."""

    if minimum_images_per_device < 2 or reference_size < 64 or denoise_sigma <= 0:
        raise ValueError("Invalid PRNU reference configuration")
    rows = read_manifest(manifest_path)
    training_rows = [
        row
        for row in rows
        if row.get("split") == "seed_train"
        and row.get("label") == "authentic"
        and row.get("dataset_name") == "premier"
        and row.get("eligible_for_split") == "true"
        and row.get("license_verified") == "true"
        and row.get("physical_source_status")
        in {"native_camera", "minimally_processed_camera"}
    ]
    by_device: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in training_rows:
        if not row.get("device_id"):
            raise ValueError("Every PRNU reference row requires device_id")
        by_device[row["device_id"]].append(row)

    output_root.mkdir(parents=True, exist_ok=True)
    devices: list[dict[str, Any]] = []
    skipped: list[str] = []
    for device_id, device_rows in sorted(by_device.items()):
        if len(device_rows) < minimum_images_per_device:
            skipped.append(device_id)
            continue
        ordered = sorted(device_rows, key=lambda row: row["source_id"])
        fingerprint = np.mean(
            [
                _reference_residual(
                    Path(row["image_path"]),
                    reference_size,
                    denoise_sigma,
                )
                for row in ordered
            ],
            axis=0,
        ).astype(np.float32)
        reference_id = hashlib.sha256(device_id.encode()).hexdigest()[:20]
        output_path = output_root / f"device_{reference_id}.npz"
        temporary = output_path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                fingerprint=fingerprint,
                device_id=device_id,
                source_ids=np.asarray([row["source_id"] for row in ordered]),
                source_sha256=np.asarray([row["sha256"] for row in ordered]),
            )
        temporary.replace(output_path)
        devices.append(
            {
                "device_id": device_id,
                "reference_id": reference_id,
                "image_count": len(ordered),
                "path": str(output_path.resolve()),
                "split": "seed_train",
            }
        )

    referenced_paths = {Path(device["path"]).resolve() for device in devices}
    stale_removed: list[str] = []
    for path in output_root.glob("device_*.npz"):
        if path.resolve() not in referenced_paths:
            path.unlink()
            stale_removed.append(str(path.resolve()))

    report = {
        "manifest": str(manifest_path.resolve()),
        "reference_size": reference_size,
        "denoise_sigma": denoise_sigma,
        "minimum_images_per_device": minimum_images_per_device,
        "reference_count": len(devices),
        "references": devices,
        "skipped_devices": skipped,
        "stale_reference_files_removed": stale_removed,
        "source_split": "seed_train",
        "selection_or_heldout_rows_read": False,
        "physical_claim": "reference fingerprints support controlled comparison only",
    }
    write_json(report_path, report)
    return report


def _normalized_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left.astype(np.float64) - float(np.mean(left))
    right_centered = right.astype(np.float64) - float(np.mean(right))
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    return float(np.sum(left_centered * right_centered) / denominator) if denominator else 0.0


def validate_prnu_device_signal(
    *,
    manifest_path: Path,
    report_path: Path,
    minimum_images_per_device: int,
    reference_images_per_device: int,
    reference_size: int = 256,
    denoise_sigma: float = 1.0,
    seed: int = 42,
    minimum_auc: float = 0.60,
) -> dict[str, Any]:
    """Validate PRNU with device identity only, never authentic/AI labels."""

    if reference_images_per_device < 2:
        raise ValueError("PRNU validation requires at least two reference images per device")
    rows = read_manifest(manifest_path)
    eligible = [
        row
        for row in rows
        if row.get("split") == "seed_train"
        and row.get("dataset_name") == "premier"
        and row.get("eligible_for_split") == "true"
        and row.get("license_verified") == "true"
        and row.get("physical_source_status")
        in {"native_camera", "minimally_processed_camera"}
    ]
    by_device: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in eligible:
        by_device[row["device_id"]].append(row)

    references: dict[str, np.ndarray] = {}
    reference_rows_by_device: dict[str, list[dict[str, str]]] = {}
    queries: dict[str, list[dict[str, str]]] = {}
    device_details: list[dict[str, Any]] = []
    for device_id, device_rows in sorted(by_device.items()):
        ordered = sorted(
            device_rows,
            key=lambda row: hashlib.sha256(
                f"{seed}:{device_id}:{row['source_id']}".encode()
            ).hexdigest(),
        )
        if len(ordered) < max(minimum_images_per_device, reference_images_per_device + 1):
            continue
        reference_rows = ordered[:reference_images_per_device]
        query_rows = ordered[reference_images_per_device:]
        references[device_id] = np.mean(
            [
                _reference_residual(Path(row["image_path"]), reference_size, denoise_sigma)
                for row in reference_rows
            ],
            axis=0,
        ).astype(np.float32)
        reference_rows_by_device[device_id] = reference_rows
        queries[device_id] = query_rows
        device_details.append(
            {
                "device_id": device_id,
                "reference_count": len(reference_rows),
                "query_count": len(query_rows),
            }
        )

    if len(references) < 2:
        raise ValueError("PRNU validation requires at least two usable seed-train devices")
    labels: list[int] = []
    scores: list[float] = []
    top1_correct = 0
    query_count = 0
    same_scores: list[float] = []
    different_scores: list[float] = []
    for device_id, query_rows in sorted(queries.items()):
        for row in query_rows:
            residual = _reference_residual(
                Path(row["image_path"]), reference_size, denoise_sigma
            )
            comparisons = {
                reference_device: _normalized_correlation(residual, reference)
                for reference_device, reference in references.items()
            }
            query_count += 1
            top1_correct += int(max(comparisons, key=comparisons.get) == device_id)
            for reference_device, score in comparisons.items():
                same = int(reference_device == device_id)
                labels.append(same)
                scores.append(score)
                (same_scores if same else different_scores).append(score)

    auc = float(roc_auc_score(labels, scores))
    top1_accuracy = top1_correct / query_count
    random_top1 = 1.0 / len(references)
    signal_validated = (
        auc >= minimum_auc
        and float(np.mean(same_scores)) > float(np.mean(different_scores))
        and top1_accuracy > random_top1
    )

    proxy_reference_vectors: dict[str, np.ndarray] = {}
    all_proxy_references: list[np.ndarray] = []
    for device_id, reference_rows in sorted(reference_rows_by_device.items()):
        vectors = [
            extract_prnu_features(
                Path(row["image_path"]),
                block_size=64,
                denoise_sigma=denoise_sigma,
                min_dimension=128,
                max_analysis_size=reference_size,
            ).values
            for row in reference_rows
        ]
        proxy_reference_vectors[device_id] = np.mean(vectors, axis=0)
        all_proxy_references.extend(vectors)
    proxy_scale = np.std(np.asarray(all_proxy_references), axis=0)
    proxy_scale[proxy_scale < 1e-12] = 1.0
    proxy_labels: list[int] = []
    proxy_scores: list[float] = []
    proxy_top1_correct = 0
    for device_id, query_rows in sorted(queries.items()):
        for row in query_rows:
            vector = extract_prnu_features(
                Path(row["image_path"]),
                block_size=64,
                denoise_sigma=denoise_sigma,
                min_dimension=128,
                max_analysis_size=reference_size,
            ).values
            comparisons = {
                reference_device: -float(
                    np.linalg.norm((vector - reference_vector) / proxy_scale)
                )
                for reference_device, reference_vector in proxy_reference_vectors.items()
            }
            proxy_top1_correct += int(max(comparisons, key=comparisons.get) == device_id)
            for reference_device, score in comparisons.items():
                proxy_labels.append(int(reference_device == device_id))
                proxy_scores.append(score)
    proxy_auc = float(roc_auc_score(proxy_labels, proxy_scores))
    proxy_top1_accuracy = proxy_top1_correct / query_count
    report = {
        "manifest": str(manifest_path.resolve()),
        "protocol": "seed_train_disjoint_reference_query_device_identity_v1",
        "seed": seed,
        "reference_size": reference_size,
        "denoise_sigma": denoise_sigma,
        "minimum_auc": minimum_auc,
        "device_count": len(references),
        "query_count": query_count,
        "pair_count": len(scores),
        "same_device_pair_count": len(same_scores),
        "different_device_pair_count": len(different_scores),
        "same_device_correlation_mean": float(np.mean(same_scores)),
        "different_device_correlation_mean": float(np.mean(different_scores)),
        "roc_auc": auc,
        "top1_device_accuracy": top1_accuracy,
        "random_top1_accuracy": random_top1,
        "signal_validated": signal_validated,
        "single_image_proxy": {
            "feature_count": len(next(iter(proxy_reference_vectors.values()))),
            "roc_auc": proxy_auc,
            "top1_device_accuracy": proxy_top1_accuracy,
            "signal_validated": (
                proxy_auc >= minimum_auc and proxy_top1_accuracy > random_top1
            ),
            "interpretation": "Device separability of the existing coherence proxy; not camera authentication.",
        },
        "devices": device_details,
        "binary_authenticity_labels_used": False,
        "selection_or_heldout_rows_read": False,
        "physical_claim": "controlled device-signal validation only; not camera authentication",
    }
    write_json(report_path, report)
    return report
