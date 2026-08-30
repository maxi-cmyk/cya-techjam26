"""Frozen global RINE and local texture-patch feature extraction for Task 9."""

from __future__ import annotations

import json
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cya_detector.data.dataset import ManifestExample
from cya_detector.features.texture import prepare_texture_patch_views, texture_patch_cache_key
from cya_detector.models.clip_baseline import LoadedClip, require_ml_dependencies
from cya_detector.training.clip_stage_a import cache_location
from cya_detector.training.rine_stage_b import extract_rine_features


_ALLOWED_SPLITS = {"seed_train", "selection_val"}


@dataclass(frozen=True)
class CachedTextureFeatures:
    example: ManifestExample
    global_cache_path: Path
    patch_cache_path: Path


def _contract(
    *,
    loaded_clip: LoadedClip,
    preprocessing_version: str,
    texture_extractor_version: str,
    patch_size: int,
    patch_count: int,
    boxes: tuple[tuple[int, int, int, int], ...],
) -> dict[str, Any]:
    return {
        "model_identifier": loaded_clip.identifier,
        "resolved_revision": loaded_clip.resolved_revision,
        "preprocessing_version": preprocessing_version,
        "texture_extractor_version": texture_extractor_version,
        "patch_size": patch_size,
        "patch_count": patch_count,
        "patch_boxes": [list(box) for box in boxes],
        "projection_dimension": loaded_clip.embedding_dimension,
    }


def _valid_patch_cache(path: Path, *, contract: dict[str, Any]) -> bool:
    torch, _, _ = require_ml_dependencies()
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        features, mask = payload["patch_features"], payload["patch_mask"]
        return (
            payload.get("cache_contract") == contract
            and payload.get("patch_boxes") == contract["patch_boxes"]
            and tuple(features.shape) == (contract["patch_count"], contract["projection_dimension"])
            and features.dtype.is_floating_point
            and bool(torch.isfinite(features).all())
            and tuple(mask.shape) == (contract["patch_count"],)
            and mask.dtype == torch.bool
            and mask.tolist() == [True] * len(contract["patch_boxes"])
            + [False] * (contract["patch_count"] - len(contract["patch_boxes"]))
        )
    except (KeyError, RuntimeError, TypeError, ValueError):
        return False


def _global_contract(
    *, example: ManifestExample, loaded_clip: LoadedClip, matching_policy: str,
    preprocessing_version: str, representation_version: str, layers: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "image_sha256": example.sha256, "image_view": example.image_view,
        "transform": example.transform, "transform_parameter": example.transform_parameter,
        "matching_policy": matching_policy, "model_identifier": loaded_clip.identifier,
        "resolved_revision": loaded_clip.resolved_revision,
        "preprocessing_version": preprocessing_version,
        "representation_version": representation_version, "layers": list(layers),
        "hidden_dimension": int(loaded_clip.model.config.hidden_size),
    }


def _global_metadata_path(path: Path) -> Path:
    return path.with_suffix(".meta.json")


def _valid_global_cache(path: Path, *, contract: dict[str, Any]) -> bool:
    torch, _, _ = require_ml_dependencies()
    try:
        metadata = json.loads(_global_metadata_path(path).read_text(encoding="utf-8"))
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        return metadata == contract and _valid_global_tensor(tensor, contract=contract)
    except (OSError, json.JSONDecodeError, RuntimeError, TypeError, ValueError):
        return False


def _valid_global_tensor(tensor: Any, *, contract: dict[str, Any]) -> bool:
    torch, _, _ = require_ml_dependencies()
    return (
        tuple(tensor.shape) == (len(contract["layers"]), contract["hidden_dimension"])
        and tensor.dtype.is_floating_point
        and bool(torch.isfinite(tensor).all())
    )


def _write_global_metadata(path: Path, contract: dict[str, Any]) -> None:
    metadata_path = _global_metadata_path(path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = metadata_path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(contract, sort_keys=True), encoding="utf-8")
    temporary.replace(metadata_path)


def _patch_pixels(patch: np.ndarray, processor: Any) -> Any:
    from PIL import Image

    image = Image.fromarray(np.asarray(patch, dtype=np.uint8)[..., :3], mode="RGB")
    return processor(images=image, return_tensors="pt")["pixel_values"][0]


def _write_patch(path: Path, payload: dict[str, Any]) -> None:
    torch, _, _ = require_ml_dependencies()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.pt")
    torch.save(payload, temporary)
    temporary.replace(path)


def extract_texture_features(
    *,
    loaded_clip: LoadedClip,
    examples: list[ManifestExample],
    global_cache_root: Path,
    patch_cache_root: Path,
    matching_policy: str,
    preprocessing_version: str,
    rine_representation_version: str,
    texture_extractor_version: str,
    layers: tuple[int, ...],
    patch_size: int,
    patch_count: int,
    batch_size: int,
    device: str,
) -> tuple[list[CachedTextureFeatures], dict[str, Any]]:
    """Extract immutable global and fixed four-position local feature caches."""

    torch, _, _ = require_ml_dependencies()
    if patch_count != 4:
        raise ValueError("Texture patch cache contract requires exactly four positions")
    if patch_size <= 0 or batch_size <= 0:
        raise ValueError("patch_size and batch_size must be positive")
    if any(example.split not in _ALLOWED_SPLITS for example in examples):
        raise ValueError("Texture extraction accepts only seed_train and selection_val examples")
    if any(example.image_view != "matched_clean" or example.transform != "clean" for example in examples):
        raise ValueError("Texture extraction accepts only matched-clean input views")
    if any(not example.sha256 for example in examples):
        raise ValueError("Every texture example must have an image SHA-256")

    started = time.perf_counter()
    existing_global_paths = {path.resolve() for path in global_cache_root.rglob("*.pt")}
    globals_, global_report = extract_rine_features(
        loaded_clip=loaded_clip, examples=examples, cache_root=global_cache_root,
        matching_policy=matching_policy, preprocessing_version=preprocessing_version,
        representation_version=rine_representation_version, layers=layers,
        batch_size=batch_size, device=device,
    )
    global_path_by_sample = {row.example.sample_id: row.cache_path for row in globals_}
    global_contracts = {
        example.sample_id: _global_contract(
            example=example, loaded_clip=loaded_clip, matching_policy=matching_policy,
            preprocessing_version=preprocessing_version,
            representation_version=rine_representation_version, layers=layers,
        )
        for example in examples
    }
    for row in globals_:
        if row.cache_path.resolve() not in existing_global_paths:
            contract = global_contracts[row.example.sample_id]
            tensor = torch.load(row.cache_path, map_location="cpu", weights_only=True)
            if not _valid_global_tensor(tensor, contract=contract):
                row.cache_path.unlink(missing_ok=True)
                raise ValueError("Frozen RINE encoder returned an invalid global feature tensor")
            _write_global_metadata(row.cache_path, contract)
    invalid_globals = [
        row for row in globals_
        if not _valid_global_cache(row.cache_path, contract=global_contracts[row.example.sample_id])
    ]
    if invalid_globals:
        for row in invalid_globals:
            row.cache_path.unlink(missing_ok=True)
            _global_metadata_path(row.cache_path).unlink(missing_ok=True)
        refreshed, _ = extract_rine_features(
            loaded_clip=loaded_clip, examples=[row.example for row in invalid_globals],
            cache_root=global_cache_root, matching_policy=matching_policy,
            preprocessing_version=preprocessing_version,
            representation_version=rine_representation_version, layers=layers,
            batch_size=batch_size, device=device,
        )
        for row in refreshed:
            contract = global_contracts[row.example.sample_id]
            tensor = torch.load(row.cache_path, map_location="cpu", weights_only=True)
            if not _valid_global_tensor(tensor, contract=contract):
                row.cache_path.unlink(missing_ok=True)
                raise ValueError("Frozen RINE encoder returned an invalid global feature tensor")
            _write_global_metadata(row.cache_path, contract)
        global_path_by_sample = {row.example.sample_id: row.cache_path for row in refreshed} | global_path_by_sample
    patch_rows: list[tuple[ManifestExample, Path, Any, dict[str, Any]]] = []
    missing: list[tuple[ManifestExample, Path, Any, dict[str, Any]]] = []
    for example in examples:
        from PIL import Image, ImageOps

        with Image.open(example.image_path) as image:
            source = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        views = prepare_texture_patch_views(source, patch_size=patch_size, patch_count=patch_count)
        contract = _contract(
            loaded_clip=loaded_clip, preprocessing_version=preprocessing_version,
            texture_extractor_version=texture_extractor_version, patch_size=patch_size,
            patch_count=patch_count, boxes=views.patch_boxes,
        )
        key = texture_patch_cache_key(
            image_sha256=example.sha256, patch_boxes=views.patch_boxes,
            model_identifier=loaded_clip.identifier, resolved_revision=loaded_clip.resolved_revision,
            preprocessing_version=preprocessing_version, extractor_version=texture_extractor_version,
        )
        path = cache_location(patch_cache_root, key)
        row = (example, path, views, contract)
        patch_rows.append(row)
        if not _valid_patch_cache(path, contract=contract):
            missing.append(row)

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    flattened: list[tuple[int, Any]] = []
    for index, (_, _, views, _) in enumerate(missing):
        flattened.extend((index, _patch_pixels(patch, loaded_clip.processor)) for patch in views.patches)
    embeddings: list[list[Any]] = [[] for _ in missing]
    with torch.inference_mode():
        for start in range(0, len(flattened), batch_size):
            batch = flattened[start : start + batch_size]
            pixels = torch.stack([pixel for _, pixel in batch]).to(device, non_blocking=True)
            precision = torch.autocast(device_type="cuda", dtype=torch.float16) if device.startswith("cuda") else nullcontext()
            with precision:
                output = loaded_clip.model(pixel_values=pixels).image_embeds
            output = output.detach().float().cpu()
            if not bool(torch.isfinite(output).all()):
                raise ValueError("Frozen texture encoder returned non-finite patch features")
            for (index, _), embedding in zip(batch, output, strict=True):
                embeddings[index].append(embedding)
    for (example, path, views, contract), values in zip(missing, embeddings, strict=True):
        features = torch.zeros((patch_count, loaded_clip.embedding_dimension), dtype=torch.float32)
        if values:
            features[: len(values)] = torch.stack(values)
        mask = torch.tensor(views.availability_mask, dtype=torch.bool)
        _write_patch(path, {
            "patch_features": features, "patch_mask": mask,
            "patch_boxes": contract["patch_boxes"], "cache_contract": contract,
        })

    rows = [CachedTextureFeatures(example, global_path_by_sample[example.sample_id], path) for example, path, _, _ in patch_rows]
    elapsed = time.perf_counter() - started
    total_bytes = sum(row.global_cache_path.stat().st_size + row.patch_cache_path.stat().st_size for row in rows)
    invalid_global_ids = {row.example.sample_id for row in invalid_globals}
    missing_patch_ids = {row[0].sample_id for row in missing}
    report = {
        "example_count": len(examples), "cache_hit_count": sum(
            example.sample_id not in invalid_global_ids and example.sample_id not in missing_patch_ids
            for example in examples
        ),
        "extracted_count": len(missing), "elapsed_seconds": elapsed,
        "extracted_images_per_second": len(missing) / elapsed if missing and elapsed else None,
        "cache_total_bytes": total_bytes, "cache_bytes_per_image": total_bytes / len(rows) if rows else 0,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") and torch.cuda.is_available() else 0,
        "model_identifier": loaded_clip.identifier, "model_revision": loaded_clip.resolved_revision,
        "patch_size": patch_size, "patch_count": patch_count,
        "projection_dimension": loaded_clip.embedding_dimension, "preprocessing_version": preprocessing_version,
        "rine_representation_version": rine_representation_version,
        "texture_extractor_version": texture_extractor_version, "layers": list(layers),
        "global_report": global_report,
    }
    return rows, report
