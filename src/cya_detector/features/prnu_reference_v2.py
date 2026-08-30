"""Native-coordinate PRNU v2 validation for the Task 8B evidence track.

This module deliberately uses device identity only.  It does not read binary
authentic/AI labels to decide whether the estimator carries a repeatable sensor
signal, and it makes no camera-authentication claim.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.restoration import denoise_wavelet
from sklearn.metrics import roc_auc_score

from cya_detector.data.manifest import read_manifest, write_json


def _native_luminance_crop(image_path: Path, crop_size: int) -> np.ndarray:
    """Read a fixed encoded-pixel crop without resize or EXIF transposition."""

    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        if min(rgb_image.size) < crop_size:
            raise ValueError(
                f"Image is smaller than the native crop ({crop_size}px): {image_path}"
            )
        left = (rgb_image.width - crop_size) // 2
        top = (rgb_image.height - crop_size) // 2
        rgb = np.asarray(
            rgb_image.crop((left, top, left + crop_size, top + crop_size)),
            dtype=np.float32,
        )
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]) / 255.0


def _zero_mean_rows_columns(array: np.ndarray) -> np.ndarray:
    centered = array - np.mean(array, axis=1, keepdims=True)
    centered -= np.mean(centered, axis=0, keepdims=True)
    return centered - float(np.mean(centered))


def _spectral_wiener_cleanup(array: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Suppress narrow/uneven spectral energy before correlation."""

    spectrum = np.fft.fft2(array)
    power = np.abs(spectrum) ** 2
    local_power = ndimage.gaussian_filter(power, sigma=sigma, mode="wrap")
    positive = local_power[local_power > 0]
    floor = float(np.median(positive)) if positive.size else 1.0
    gain = np.minimum(1.0, np.sqrt(floor / np.maximum(local_power, floor * 1e-6)))
    cleaned = np.fft.ifft2(spectrum * gain).real.astype(np.float32)
    standard_deviation = float(np.std(cleaned))
    return cleaned / standard_deviation if standard_deviation > 1e-12 else cleaned


def _wavelet_residual(
    image_path: Path,
    *,
    crop_size: int,
    wavelet: str,
    wavelet_levels: int,
    edge_keep_quantile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    luminance = _native_luminance_crop(image_path, crop_size)
    denoised = denoise_wavelet(
        luminance,
        wavelet=wavelet,
        wavelet_levels=wavelet_levels,
        method="BayesShrink",
        mode="soft",
        rescale_sigma=True,
        channel_axis=None,
    ).astype(np.float32)
    residual = _spectral_wiener_cleanup(_zero_mean_rows_columns(luminance - denoised))

    gradient = np.hypot(
        ndimage.sobel(luminance, axis=0, mode="reflect"),
        ndimage.sobel(luminance, axis=1, mode="reflect"),
    )
    intensity_mask = (luminance >= 0.05) & (luminance <= 0.95)
    if np.any(intensity_mask):
        edge_limit = float(np.quantile(gradient[intensity_mask], edge_keep_quantile))
        mask = intensity_mask & (gradient <= edge_limit)
    else:
        mask = intensity_mask
    return luminance, residual, mask


def _estimate_fingerprint(
    rows: list[dict[str, str]],
    *,
    crop_size: int,
    wavelet: str,
    wavelet_levels: int,
    edge_keep_quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    numerator = np.zeros((crop_size, crop_size), dtype=np.float64)
    denominator = np.zeros_like(numerator)
    support = np.zeros_like(numerator)
    for row in rows:
        intensity, residual, mask = _wavelet_residual(
            Path(row["image_path"]),
            crop_size=crop_size,
            wavelet=wavelet,
            wavelet_levels=wavelet_levels,
            edge_keep_quantile=edge_keep_quantile,
        )
        weight = mask.astype(np.float32)
        numerator += residual * intensity * weight
        denominator += intensity**2 * weight
        support += weight
    fingerprint = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )
    fingerprint = _spectral_wiener_cleanup(_zero_mean_rows_columns(fingerprint))
    minimum_support = max(2, int(np.ceil(len(rows) * 0.5)))
    support_mask = support >= minimum_support
    fingerprint[~support_mask] = 0.0
    return fingerprint.astype(np.float32), support_mask


def _pce_score(
    residual: np.ndarray,
    expected: np.ndarray,
    mask: np.ndarray,
    *,
    maximum_shift: int,
    exclusion_radius: int = 5,
) -> float:
    left = np.where(mask, residual, 0.0).astype(np.float64)
    right = np.where(mask, expected, 0.0).astype(np.float64)
    left -= float(np.sum(left) / max(int(np.sum(mask)), 1)) * mask
    right -= float(np.sum(right) / max(int(np.sum(mask)), 1)) * mask
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    correlation = np.fft.ifft2(np.fft.fft2(left) * np.conj(np.fft.fft2(right))).real
    correlation /= denominator

    shifts = range(-maximum_shift, maximum_shift + 1)
    candidates = [
        (float(correlation[row % correlation.shape[0], column % correlation.shape[1]]), row, column)
        for row in shifts
        for column in shifts
    ]
    peak, peak_row, peak_column = max(candidates, key=lambda item: item[0])
    energy_mask = np.ones(correlation.shape, dtype=bool)
    for row_offset in range(-exclusion_radius, exclusion_radius + 1):
        for column_offset in range(-exclusion_radius, exclusion_radius + 1):
            if row_offset**2 + column_offset**2 <= exclusion_radius**2:
                energy_mask[
                    (peak_row + row_offset) % correlation.shape[0],
                    (peak_column + column_offset) % correlation.shape[1],
                ] = False
    background_energy = float(np.mean(correlation[energy_mask] ** 2))
    return float(np.sign(peak) * peak**2 / background_energy) if background_energy else 0.0


def validate_prnu_device_signal_v2(
    *,
    manifest_path: Path,
    artifact_root: Path,
    reference_images_per_device: int = 25,
    crop_size: int = 512,
    wavelet: str = "db2",
    wavelet_levels: int = 4,
    edge_keep_quantile: float = 0.75,
    maximum_shift: int = 8,
    seed: int = 42,
    minimum_auc: float = 0.60,
) -> dict[str, Any]:
    """Run the predeclared v2 device-separation experiment and persist evidence."""

    if reference_images_per_device < 2 or crop_size < 128:
        raise ValueError("PRNU v2 needs at least two references and a crop of at least 128px")
    if not 0.0 < edge_keep_quantile <= 1.0 or maximum_shift < 0:
        raise ValueError("Invalid PRNU v2 masking or alignment configuration")

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
        if row.get("device_id"):
            by_device[row["device_id"]].append(row)

    reference_rows: dict[str, list[dict[str, str]]] = {}
    query_rows: dict[str, list[dict[str, str]]] = {}
    skipped: dict[str, str] = {}
    for device_id, device_rows in sorted(by_device.items()):
        ordered = sorted(
            device_rows,
            key=lambda row: hashlib.sha256(
                f"{seed}:{device_id}:{row['source_id']}".encode()
            ).hexdigest(),
        )
        usable: list[dict[str, str]] = []
        for row in ordered:
            try:
                with Image.open(row["image_path"]) as image:
                    large_enough = min(image.size) >= crop_size
            except (OSError, ValueError):
                large_enough = False
            if large_enough:
                usable.append(row)
        if len(usable) < reference_images_per_device + 1:
            skipped[device_id] = (
                f"{len(usable)} native images meet the {crop_size}px crop; "
                f"need {reference_images_per_device + 1}"
            )
            continue
        reference_rows[device_id] = usable[:reference_images_per_device]
        query_rows[device_id] = usable[reference_images_per_device:]

    if len(reference_rows) < 2:
        raise ValueError("PRNU v2 requires at least two devices with reference and query images")

    fingerprint_root = artifact_root / "fingerprints"
    fingerprint_root.mkdir(parents=True, exist_ok=True)
    references: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    device_details: list[dict[str, Any]] = []
    written_fingerprints: set[Path] = set()
    for device_id, device_reference_rows in sorted(reference_rows.items()):
        fingerprint, support_mask = _estimate_fingerprint(
            device_reference_rows,
            crop_size=crop_size,
            wavelet=wavelet,
            wavelet_levels=wavelet_levels,
            edge_keep_quantile=edge_keep_quantile,
        )
        references[device_id] = (fingerprint, support_mask)
        reference_id = hashlib.sha256(device_id.encode()).hexdigest()[:20]
        output_path = fingerprint_root / f"device_{reference_id}.npz"
        temporary_path = output_path.with_suffix(".tmp")
        with temporary_path.open("wb") as stream:
            np.savez_compressed(
                stream,
                fingerprint=fingerprint,
                support_mask=support_mask,
                device_id=device_id,
                source_ids=np.asarray([row["source_id"] for row in device_reference_rows]),
                source_sha256=np.asarray([row["sha256"] for row in device_reference_rows]),
            )
        temporary_path.replace(output_path)
        written_fingerprints.add(output_path.resolve())
        device_details.append(
            {
                "device_id": device_id,
                "reference_count": len(device_reference_rows),
                "query_count": len(query_rows[device_id]),
                "fingerprint_path": str(output_path.resolve()),
                "support_fraction": float(np.mean(support_mask)),
            }
        )
    stale_fingerprints_removed: list[str] = []
    for fingerprint_path in fingerprint_root.glob("device_*.npz"):
        if fingerprint_path.resolve() not in written_fingerprints:
            fingerprint_path.unlink()
            stale_fingerprints_removed.append(str(fingerprint_path.resolve()))

    labels: list[int] = []
    scores: list[float] = []
    same_scores: list[float] = []
    different_scores: list[float] = []
    top1_correct = 0
    query_count = 0
    for device_id, rows_for_query in sorted(query_rows.items()):
        for row in rows_for_query:
            intensity, residual, query_mask = _wavelet_residual(
                Path(row["image_path"]),
                crop_size=crop_size,
                wavelet=wavelet,
                wavelet_levels=wavelet_levels,
                edge_keep_quantile=edge_keep_quantile,
            )
            comparisons: dict[str, float] = {}
            for reference_device, (fingerprint, support_mask) in references.items():
                comparison_mask = query_mask & support_mask
                comparisons[reference_device] = _pce_score(
                    residual,
                    fingerprint * intensity,
                    comparison_mask,
                    maximum_shift=maximum_shift,
                )
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
    same_mean = float(np.mean(same_scores))
    different_mean = float(np.mean(different_scores))
    signal_validated = (
        auc >= minimum_auc and same_mean > different_mean and top1_accuracy > random_top1
    )
    report = {
        "manifest": str(manifest_path.resolve()),
        "artifact_root": str(artifact_root.resolve()),
        "protocol": "seed_train_disjoint_reference_query_device_identity_prnu_v2",
        "estimator": "native_crop_wavelet_multiplicative_spectral_wiener_pce",
        "seed": seed,
        "crop_size": crop_size,
        "resize_applied": False,
        "exif_transpose_applied": False,
        "wavelet": wavelet,
        "wavelet_levels": wavelet_levels,
        "edge_keep_quantile": edge_keep_quantile,
        "maximum_registration_shift_pixels": maximum_shift,
        "reference_images_per_device": reference_images_per_device,
        "minimum_auc": minimum_auc,
        "device_count": len(references),
        "query_count": query_count,
        "pair_count": len(scores),
        "same_device_pair_count": len(same_scores),
        "different_device_pair_count": len(different_scores),
        "same_device_pce_mean": same_mean,
        "different_device_pce_mean": different_mean,
        "roc_auc": auc,
        "top1_device_accuracy": top1_accuracy,
        "random_top1_accuracy": random_top1,
        "signal_validated": signal_validated,
        "devices": device_details,
        "skipped_devices": skipped,
        "stale_fingerprint_files_removed": stale_fingerprints_removed,
        "binary_authenticity_labels_used": False,
        "selection_or_heldout_rows_read": False,
        "fusion_training_run": False,
        "physical_claim": "controlled repeatable device-signal validation only",
        "camera_authentication_claim": False,
        "decision": (
            "Independent device-signal gate passed; binary usefulness still requires a separate locked ablation."
            if signal_validated
            else "Independent device-signal gate failed; do not expose PRNU v2 to binary fusion."
        ),
    }
    write_json(artifact_root / "audits" / "prnu_v2_signal_validation.json", report)
    return report
