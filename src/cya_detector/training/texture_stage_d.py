"""Frozen global RINE and local texture-patch feature extraction for Task 9."""

from __future__ import annotations

import json
import hashlib
import math
import os
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cya_detector.data.dataset import ManifestExample
from cya_detector.evaluation.metrics import evaluate_predictions
from cya_detector.features.texture import prepare_texture_patch_views, texture_patch_cache_key
from cya_detector.models.clip_baseline import LoadedClip, require_ml_dependencies
from cya_detector.models.texture import build_texture_head
from cya_detector.predictions import PredictionRecord, write_predictions
from cya_detector.training.clip_stage_a import cache_location
from cya_detector.training.rine_stage_b import extract_rine_features


_ALLOWED_SPLITS = {"seed_train", "selection_val"}
APPROVED_MATCHING_POLICY = "fixed_q96"
LOCKED_TEXTURE_VARIANTS = ("global_only", "local_only", "global_local")
LOCKED_TEXTURE_SEEDS = (42, 43, 44)
REQUIRED_RUN_ARTIFACTS = (
    "checkpoints/best_clean.pt", "checkpoints/latest.pt", "predictions/selection_val.csv",
    "reports/metrics.json", "reports/training_history.json", "metadata/run_metadata.json",
)


@dataclass(frozen=True)
class CachedTextureFeatures:
    example: ManifestExample
    global_cache_path: Path
    patch_cache_path: Path
    matching_policy: str = ""


def _contract(
    *,
    loaded_clip: LoadedClip,
    preprocessing_version: str,
    texture_extractor_version: str,
    patch_size: int,
    patch_count: int,
    boxes: tuple[tuple[int, int, int, int], ...],
    matching_policy: str,
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
        "matching_policy": matching_policy,
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
            patch_count=patch_count, boxes=views.patch_boxes, matching_policy=matching_policy,
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

    rows = [
        CachedTextureFeatures(example, global_path_by_sample[example.sample_id], path, matching_policy)
        for example, path, _, _ in patch_rows
    ]
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


def _atomic_json(path: Path, value: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_predictions(path: Path, records: list[PredictionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_predictions(temporary, records)
    temporary.replace(path)


def _atomic_checkpoint(path: Path, *, model: Any, state: dict[str, Any]) -> None:
    torch, _, _ = require_ml_dependencies()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model_state_dict": model.state_dict(), **state}, temporary)
    temporary.replace(path)


def write_cached_texture_features_payload(
    path: Path, *, rows: list[CachedTextureFeatures], task4_extraction_report: dict[str, Any],
) -> None:
    """Serialize the supported handoff from Task 4 extraction to Task 5 training."""

    if any(row.matching_policy != APPROVED_MATCHING_POLICY for row in rows):
        raise ValueError("Task 4 cache payload requires fixed_q96 cached provenance")
    payload = {
        "schema_version": 1,
        "matching_policy": APPROVED_MATCHING_POLICY,
        "rows": [
            {
                "example": {
                    **row.example.__dict__,
                    "image_path": str(row.example.image_path),
                },
                "global_cache_path": str(row.global_cache_path),
                "patch_cache_path": str(row.patch_cache_path),
                "matching_policy": row.matching_policy,
            }
            for row in rows
        ],
        "task4_extraction_report": task4_extraction_report,
    }
    _atomic_json(Path(path), payload)


def read_cached_texture_features_payload(path: Path) -> tuple[list[CachedTextureFeatures], dict[str, Any]]:
    """Load the versioned Task 4 cache handoff accepted by texture training."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rows_payload = payload["rows"]
        report = payload["task4_extraction_report"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Task 4 cache payload must contain rows and task4_extraction_report") from exc
    if (
        not isinstance(payload, dict) or payload.get("schema_version") != 1
        or payload.get("matching_policy") != APPROVED_MATCHING_POLICY
        or not isinstance(rows_payload, list) or not isinstance(report, dict)
    ):
        raise ValueError("Task 4 cache payload requires schema version 1 and fixed_q96 provenance")
    try:
        rows = [
            CachedTextureFeatures(
                example=ManifestExample(**{**row["example"], "image_path": Path(row["example"]["image_path"])}),
                global_cache_path=Path(row["global_cache_path"]),
                patch_cache_path=Path(row["patch_cache_path"]),
                matching_policy=row["matching_policy"],
            )
            for row in rows_payload
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError("Task 4 cache payload contains an invalid cached row") from exc
    if any(row.matching_policy != APPROVED_MATCHING_POLICY for row in rows):
        raise ValueError("Task 4 cache payload requires fixed_q96 cached provenance")
    return rows, report


def _require_finite(value: Any, *, name: str) -> None:
    """Reject non-finite numerical data before it can become a published artifact."""

    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite(item, name=f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite(item, name=f"{name}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Texture training refused non-finite {name}")


def _replace_published_file(source: Path, destination: Path) -> None:
    """Atomically replace one completed artifact where the filesystem supports it."""

    source.replace(destination)


def _artifact_sha256(root: Path) -> dict[str, str]:
    """Hash every required run artifact except the metadata commit record itself.

    Hashes exactly the fixed REQUIRED_RUN_ARTIFACTS set (not a directory glob) so a stray
    file that ever lands in staging cannot get hashed and then fail the gate's exact-set
    comparison against a run that is otherwise completely and correctly published.
    """

    return {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in REQUIRED_RUN_ARTIFACTS
        if relative != "metadata/run_metadata.json"
    }


def _publish_staged_run(staging: Path, run_root: Path) -> None:
    """Commit one staged run with metadata-last content hashes as the consumer boundary."""

    if not run_root.exists():
        staging.replace(run_root)
        return
    staged_files = sorted(path for path in staging.rglob("*") if path.is_file())
    metadata = staging / "metadata" / "run_metadata.json"
    for source in [path for path in staged_files if path != metadata] + [metadata]:
        destination = run_root / source.relative_to(staging)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _replace_published_file(source, destination)
    import shutil
    shutil.rmtree(staging)


def _load_cached_texture_rows(rows: list[CachedTextureFeatures]) -> tuple[Any, Any, Any, Any, list[CachedTextureFeatures], list[CachedTextureFeatures]]:
    """Load each frozen cache exactly once after validating its clean-only contract."""

    torch, _, _ = require_ml_dependencies()
    if not rows:
        raise ValueError("Texture training requires cached rows")
    if any(row.example.split not in _ALLOWED_SPLITS for row in rows):
        raise ValueError("Texture training accepts only seed_train and selection_val rows")
    if any(row.example.image_view != "matched_clean" or row.example.transform != "clean" for row in rows):
        raise ValueError("Texture training accepts only matched-clean clean rows")
    if any(row.matching_policy != APPROVED_MATCHING_POLICY for row in rows):
        raise ValueError("Texture training requires fixed_q96 cached provenance")
    train_rows = [row for row in rows if row.example.split == "seed_train"]
    selection_rows = [row for row in rows if row.example.split == "selection_val"]
    if not train_rows or not selection_rows:
        raise ValueError("Texture training requires seed_train and selection_val rows")
    if {row.example.target for row in train_rows} != {0, 1}:
        raise ValueError("Texture training data must contain both classes")

    loaded: list[tuple[CachedTextureFeatures, Any, Any, Any]] = []
    for row in rows:
        try:
            global_contract = json.loads(_global_metadata_path(row.global_cache_path).read_text(encoding="utf-8"))
            global_features = torch.load(row.global_cache_path, map_location="cpu", weights_only=True)
            patch_payload = torch.load(row.patch_cache_path, map_location="cpu", weights_only=True)
            patch_features = patch_payload["patch_features"]
            patch_mask = patch_payload["patch_mask"]
            patch_contract = patch_payload["cache_contract"]
        except (OSError, KeyError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot load frozen texture caches for {row.example.sample_id}") from exc
        if (
            not isinstance(global_contract, dict)
            or not isinstance(patch_contract, dict)
            or global_contract.get("matching_policy") != APPROVED_MATCHING_POLICY
            or patch_contract.get("matching_policy") != APPROVED_MATCHING_POLICY
        ):
            raise ValueError("Frozen texture cache contracts require fixed_q96 provenance")
        if (
            global_features.ndim != 2
            or patch_features.ndim != 2
            or patch_mask.ndim != 1
            or patch_features.shape[0] != patch_mask.shape[0]
            or patch_mask.dtype != torch.bool
            or not global_features.dtype.is_floating_point
            or not patch_features.dtype.is_floating_point
            or not bool(torch.isfinite(global_features).all())
            or not bool(torch.isfinite(patch_features).all())
            or not bool(patch_mask.any())
        ):
            raise ValueError("Frozen texture caches must be shape-valid with non-finite values refused")
        loaded.append((row, global_features.float(), patch_features.float(), patch_mask))

    global_shape = tuple(loaded[0][1].shape)
    patch_shape = tuple(loaded[0][2].shape)
    if any(tuple(global_features.shape) != global_shape or tuple(patch_features.shape) != patch_shape or tuple(mask.shape) != (patch_shape[0],) for _, global_features, patch_features, mask in loaded):
        raise ValueError("Frozen texture cache shapes must agree across rows")

    def stacked(split: str) -> tuple[Any, Any, Any, Any, list[CachedTextureFeatures]]:
        selected = [value for value in loaded if value[0].example.split == split]
        return (
            torch.stack([value[1] for value in selected]),
            torch.stack([value[2] for value in selected]),
            torch.stack([value[3] for value in selected]),
            torch.tensor([value[0].example.target for value in selected], dtype=torch.float32),
            [value[0] for value in selected],
        )

    train_global, train_patch, train_mask, train_targets, ordered_train = stacked("seed_train")
    selection_global, selection_patch, selection_mask, selection_targets, ordered_selection = stacked("selection_val")
    return (
        train_global, train_patch, train_mask, train_targets,
        selection_global, selection_patch, selection_mask, selection_targets,
        ordered_train, ordered_selection,
    )


def train_texture_head(
    *,
    rows: list[CachedTextureFeatures],
    variant: str,
    seed: int,
    output_root: Path,
    overwrite: bool,
    run_configuration: dict[str, Any],
    **optimization: Any,
) -> dict[str, Any]:
    """Train one lightweight head from frozen clean-only global and patch caches."""

    torch, _, _ = require_ml_dependencies()
    if "texture" not in run_configuration:
        raise ValueError("Texture training requires run_configuration['texture']")
    texture_config = run_configuration["texture"]
    if "fusion_dimension" not in texture_config:
        raise ValueError("Texture training requires configured texture.fusion_dimension")
    if variant not in LOCKED_TEXTURE_VARIANTS:
        raise ValueError("Texture training requires a configured texture variant")
    if seed not in LOCKED_TEXTURE_SEEDS:
        raise ValueError("Texture training requires a configured texture seed")
    required = (
        "device", "learning_rate", "weight_decay", "warmup_fraction", "max_epochs",
        "early_stopping_patience", "physical_batch_size", "effective_batch_size", "threshold",
    )
    missing = [name for name in required if name not in optimization]
    if missing:
        raise ValueError(f"Missing texture optimization setting(s): {', '.join(missing)}")
    device = str(optimization["device"])
    physical_batch_size = int(optimization["physical_batch_size"])
    effective_batch_size = int(optimization["effective_batch_size"])
    max_epochs = int(optimization["max_epochs"])
    if physical_batch_size <= 0 or effective_batch_size <= 0 or max_epochs <= 0:
        raise ValueError("Texture batch sizes and max_epochs must be positive")

    (
        train_global, train_patch, train_mask, train_targets,
        selection_global, selection_patch, selection_mask, _, _, selection_rows,
    ) = _load_cached_texture_rows(rows)
    run_root = Path(output_root) / variant / f"seed_{seed}"
    completed_marker = run_root / "metadata" / "run_metadata.json"
    if run_root.exists() and completed_marker.is_file() and not overwrite:
        raise FileExistsError(f"Refusing completed texture run overwrite: {run_root}")
    if run_root.exists() and not completed_marker.is_file() and not overwrite:
        raise FileExistsError(f"Refusing incomplete texture run overwrite: {run_root}")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # cuBLAS refuses deterministic GEMMs (every Linear head uses one) unless this is set;
        # it must be present before the first cuBLAS call, which the next line can trigger.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=False)
    layer_count, global_dimension = train_global.shape[1:]
    _, patch_dimension = train_patch.shape[1:]
    fusion_dimension = int(texture_config["fusion_dimension"])
    model = build_texture_head(
        variant=variant, layer_count=layer_count, global_dimension=global_dimension,
        patch_dimension=patch_dimension, fusion_dimension=fusion_dimension,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    accumulation_steps = max(1, math.ceil(effective_batch_size / physical_batch_size))
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_global, train_patch, train_mask, train_targets),
        batch_size=physical_batch_size, shuffle=True, generator=generator,
    )
    total_steps = max(1, math.ceil(len(loader) / accumulation_steps) * max_epochs)
    warmup_steps = int(total_steps * float(optimization["warmup_fraction"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (step + 1) / warmup_steps if warmup_steps and step < warmup_steps else 1.0,
    )

    staging = run_root.parent / f".{run_root.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    best_accuracy = -math.inf
    best_predictions: list[PredictionRecord] = []
    best_metrics: dict[str, Any] | None = None
    best_inference: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    patience = 0
    try:
        for epoch in range(1, max_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            for step, (global_features, patch_features, patch_mask, targets) in enumerate(loader, start=1):
                logits = model(global_features.to(device), patch_features.to(device), patch_mask.to(device)).squeeze(1)
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("Texture training refused non-finite training logits")
                loss = criterion(logits, targets.to(device))
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Texture training refused a non-finite loss")
                (loss / accumulation_steps).backward()
                loss_total += float(loss.item())
                if step % accumulation_steps == 0 or step == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(optimization.get("gradient_clip_norm", 1.0)))
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            model.eval()
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            inference_started = time.perf_counter()
            with torch.inference_mode():
                logits = model(selection_global.to(device), selection_patch.to(device), selection_mask.to(device)).squeeze(1).cpu()
                probabilities = torch.sigmoid(logits)
            inference = {
                "sample_count": len(selection_rows),
                "latency_seconds": time.perf_counter() - inference_started,
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") and torch.cuda.is_available() else 0,
            }
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(probabilities).all()):
                raise ValueError("Texture training refused non-finite validation logits or probabilities")
            predictions = [
                PredictionRecord(
                    sample_id=row.example.sample_id, source_id=row.example.source_id,
                    parent_id=row.example.parent_id, split=row.example.split, label=row.example.label,
                    logit=float(logit), probability=float(probability), checkpoint="best_clean",
                    seed=seed, matching_policy=APPROVED_MATCHING_POLICY, transform=row.example.transform,
                    transform_parameter=row.example.transform_parameter, **row.example.metadata,
                )
                for row, logit, probability in zip(selection_rows, logits.tolist(), probabilities.tolist(), strict=True)
            ]
            metrics = evaluate_predictions(predictions, threshold=float(optimization["threshold"]))
            _require_finite(metrics, name="metrics")
            accuracy = metrics["clean"]["accuracy"]
            history.append({"epoch": epoch, "training_loss": loss_total / len(loader), "clean_accuracy": accuracy})
            state = {"stage": "texture_stage_d", "variant": variant, "seed": seed, "epoch": epoch}
            _atomic_checkpoint(staging / "checkpoints" / "latest.pt", model=model, state=state)
            if accuracy > best_accuracy:
                best_accuracy, best_predictions, best_metrics, best_inference = accuracy, predictions, metrics, inference
                patience = 0
                _atomic_checkpoint(staging / "checkpoints" / "best_clean.pt", model=model, state=state)
            else:
                patience += 1
                if patience >= int(optimization["early_stopping_patience"]):
                    break

        if best_metrics is None or best_inference is None:
            raise ValueError("Texture training produced no selection metrics")
        extraction_report = optimization.get("task4_extraction_report")
        if not isinstance(extraction_report, dict):
            raise ValueError("Texture training requires the Task 4 extraction report")
        _require_finite(extraction_report, name="task4_extraction")
        _atomic_predictions(staging / "predictions" / "selection_val.csv", best_predictions)
        _atomic_json(staging / "reports" / "metrics.json", {
            "selection_split": "selection_val", "matching_policy": APPROVED_MATCHING_POLICY,
            "inference": best_inference, "task4_extraction": extraction_report, **best_metrics,
        })
        _atomic_json(staging / "reports" / "training_history.json", {"history": history})
        run_metadata = {
            "status": "completed", "stage": "texture_stage_d", "variant": variant, "seed": seed,
            "matching_policy": APPROVED_MATCHING_POLICY,
            "allowed_splits": sorted(_ALLOWED_SPLITS), "run_configuration": run_configuration,
            "optimization": optimization, "accumulation_steps": accumulation_steps, "warmup_steps": warmup_steps,
            "artifact_sha256": _artifact_sha256(staging),
        }
        _atomic_json(staging / "metadata" / "run_metadata.json", run_metadata)
        _publish_staged_run(staging, run_root)
    except Exception:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"status": "completed", "run_root": str(run_root), "best_clean_accuracy": best_accuracy, "epochs_completed": len(history)}
