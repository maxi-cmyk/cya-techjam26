"""Extraction and training for the frozen-CLIP RINE-style Stage B ablation."""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from cya_detector.data.dataset import ClipImageDataset, ManifestExample
from cya_detector.evaluation.metrics import evaluate_predictions
from cya_detector.models.clip_baseline import (
    LoadedClip,
    embedding_cache_key,
    require_ml_dependencies,
)
from cya_detector.models.rine import build_rine_head, validate_rine_layers
from cya_detector.predictions import PredictionRecord, write_predictions
from cya_detector.training.clip_stage_a import CachedEmbedding, cache_location


def _collate_images(batch: list[dict[str, Any]]) -> dict[str, Any]:
    torch, _, _ = require_ml_dependencies()
    return {
        "pixel_values": torch.stack([row["pixel_values"] for row in batch]),
        "examples": [row["example"] for row in batch],
    }


def extract_rine_features(
    *,
    loaded_clip: LoadedClip,
    examples: list[ManifestExample],
    cache_root: Path,
    matching_policy: str,
    preprocessing_version: str,
    representation_version: str,
    layers: list[int] | tuple[int, ...],
    batch_size: int,
    device: str,
) -> tuple[list[CachedEmbedding], dict[str, Any]]:
    """Cache selected intermediate CLS representations from a frozen vision tower."""

    torch, _, _ = require_ml_dependencies()
    model_layer_count = int(loaded_clip.model.config.num_hidden_layers)
    selected_layers = validate_rine_layers(layers, layer_count=model_layer_count)
    hidden_dimension = int(loaded_clip.model.config.hidden_size)
    if any(not example.sha256 for example in examples):
        raise ValueError("Every RINE example must have an image SHA-256")

    cached: list[CachedEmbedding] = []
    missing: list[ManifestExample] = []
    keys: dict[str, str] = {}
    layer_signature = "-".join(str(layer) for layer in selected_layers)
    for example in examples:
        view = ":".join(
            (
                matching_policy,
                example.image_view,
                example.transform or "clean",
                example.transform_parameter or "default",
                representation_version,
                f"layers-{layer_signature}",
            )
        )
        key = embedding_cache_key(
            image_sha256=example.sha256,
            model_identifier=loaded_clip.identifier,
            resolved_revision=loaded_clip.resolved_revision,
            preprocessing_version=preprocessing_version,
            view_identifier=view,
        )
        keys[example.sample_id] = key
        path = cache_location(cache_root, key)
        cached.append(CachedEmbedding(example=example, cache_key=key, cache_path=path))
        if not path.is_file():
            missing.append(example)

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if missing:
        from tqdm.auto import tqdm

        loader = torch.utils.data.DataLoader(
            ClipImageDataset(missing, loaded_clip.processor),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.startswith("cuda"),
            collate_fn=_collate_images,
        )
        with torch.inference_mode():
            for batch in tqdm(
                loader,
                total=len(loader),
                desc=f"RINE {matching_policy}",
                unit="batch",
                dynamic_ncols=True,
            ):
                pixels = batch["pixel_values"].to(device, non_blocking=True)
                precision_context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if device.startswith("cuda")
                    else nullcontext()
                )
                with precision_context:
                    outputs = loaded_clip.model(
                        pixel_values=pixels,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    features = torch.stack(
                        [outputs.hidden_states[layer][:, 0, :] for layer in selected_layers],
                        dim=1,
                    )
                features = features.detach().float().cpu()
                for example, feature in zip(batch["examples"], features, strict=True):
                    path = cache_location(cache_root, keys[example.sample_id])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(".tmp.pt")
                    torch.save(feature, temporary)
                    temporary.replace(path)

    elapsed = time.perf_counter() - started
    total_bytes = sum(row.cache_path.stat().st_size for row in cached)
    peak_memory = (
        int(torch.cuda.max_memory_allocated())
        if device.startswith("cuda") and torch.cuda.is_available()
        else 0
    )
    report = {
        "example_count": len(examples),
        "cache_hit_count": len(examples) - len(missing),
        "extracted_count": len(missing),
        "elapsed_seconds": elapsed,
        "extracted_images_per_second": len(missing) / elapsed if elapsed and missing else None,
        "cache_total_bytes": total_bytes,
        "cache_bytes_per_image": total_bytes / len(cached),
        "peak_gpu_memory_bytes": peak_memory,
        "model_identifier": loaded_clip.identifier,
        "resolved_revision": loaded_clip.resolved_revision,
        "preprocessing_version": preprocessing_version,
        "representation_version": representation_version,
        "layers": list(selected_layers),
        "hidden_dimension": hidden_dimension,
        "batch_size": batch_size,
    }
    return cached, report


def _load_features(rows: list[CachedEmbedding]) -> tuple[Any, Any]:
    torch, _, _ = require_ml_dependencies()
    features = torch.stack(
        [torch.load(row.cache_path, map_location="cpu", weights_only=True) for row in rows]
    )
    targets = torch.tensor([row.example.target for row in rows], dtype=torch.float32)
    return features, targets


def _predict(
    *,
    model: Any,
    features: Any,
    rows: list[CachedEmbedding],
    checkpoint: str,
    seed: int,
    matching_policy: str,
    device: str,
) -> list[PredictionRecord]:
    torch, _, _ = require_ml_dependencies()
    model.eval()
    with torch.inference_mode():
        logits = model(features.to(device)).squeeze(1).detach().cpu()
        probabilities = torch.sigmoid(logits)
    predictions: list[PredictionRecord] = []
    for row, logit, probability in zip(rows, logits.tolist(), probabilities.tolist(), strict=True):
        example = row.example
        predictions.append(
            PredictionRecord(
                sample_id=example.sample_id,
                source_id=example.source_id,
                parent_id=example.parent_id,
                split=example.split,
                label=example.label,
                logit=logit,
                probability=probability,
                checkpoint=checkpoint,
                seed=seed,
                matching_policy=matching_policy,
                transform=example.transform,
                transform_parameter=example.transform_parameter,
                **example.metadata,
            )
        )
    return predictions


def _save_checkpoint(path: Path, *, model: Any, state: dict[str, Any]) -> None:
    torch, _, _ = require_ml_dependencies()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.pt")
    torch.save({"model_state_dict": model.state_dict(), **state}, temporary)
    temporary.replace(path)


def train_rine_head(
    *,
    train_rows: list[CachedEmbedding],
    selection_rows: list[CachedEmbedding],
    output_directory: Path,
    matching_policy: str,
    layers: list[int] | tuple[int, ...],
    resolved_revision: str,
    manifest_sha256: str,
    seed: int,
    device: str,
    learning_rate: float,
    weight_decay: float,
    warmup_fraction: float,
    max_epochs: int,
    early_stopping_patience: int,
    physical_batch_size: int,
    effective_batch_size: int,
    threshold: float,
    run_configuration: dict[str, Any],
) -> dict[str, Any]:
    """Train only layer importance and the binary classifier."""

    torch, _, _ = require_ml_dependencies()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_features, train_targets = _load_features(train_rows)
    selection_features, _ = _load_features(selection_rows)
    if set(train_targets.tolist()) != {0.0, 1.0}:
        raise ValueError("RINE training data must contain both classes")
    layer_count, hidden_dimension = train_features.shape[1:]
    if layer_count != len(layers):
        raise ValueError("Cached RINE layer count does not match configuration")

    model = build_rine_head(
        layer_count=layer_count, hidden_dimension=hidden_dimension
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    accumulation_steps = max(1, math.ceil(effective_batch_size / physical_batch_size))
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_features, train_targets),
        batch_size=physical_batch_size,
        shuffle=True,
        generator=generator,
    )
    total_steps = max(1, math.ceil(len(loader) / accumulation_steps) * max_epochs)
    warmup_steps = int(total_steps * warmup_fraction)

    def learning_rate_multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
    output_directory.mkdir(parents=True, exist_ok=True)
    base_state = {
        "stage": "rine_stage_b",
        "seed": seed,
        "matching_policy": matching_policy,
        "layers": list(layers),
        "resolved_model_revision": resolved_revision,
        "manifest_sha256": manifest_sha256,
        "threshold": threshold,
        "resolved_run_configuration": run_configuration,
    }
    best_clean = -math.inf
    best_layer_importance: list[float] | None = None
    patience = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for step, (features, targets) in enumerate(loader, start=1):
            logits = model(features.to(device)).squeeze(1)
            loss = criterion(logits, targets.to(device)) / accumulation_steps
            loss.backward()
            total_loss += loss.item() * accumulation_steps
            if step % accumulation_steps == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        predictions = _predict(
            model=model,
            features=selection_features,
            rows=selection_rows,
            checkpoint=f"epoch_{epoch:03d}",
            seed=seed,
            matching_policy=matching_policy,
            device=device,
        )
        report = evaluate_predictions(predictions, threshold=threshold)
        clean_accuracy = report["clean"]["accuracy"] if report["clean"] else None
        history.append(
            {
                "epoch": epoch,
                "training_loss": total_loss / len(loader),
                "clean_accuracy": clean_accuracy,
                "layer_importance": model.importance_weights(),
            }
        )
        state = {
            **base_state,
            "epoch": epoch,
            "selection_metrics": report,
            "layer_importance": model.importance_weights(),
        }
        _save_checkpoint(output_directory / "latest.pt", model=model, state=state)
        write_predictions(output_directory / "latest_predictions.csv", predictions)
        if clean_accuracy is not None and clean_accuracy > best_clean:
            best_clean = clean_accuracy
            best_layer_importance = model.importance_weights()
            patience = 0
            _save_checkpoint(output_directory / "best_clean.pt", model=model, state=state)
            write_predictions(output_directory / "best_clean_predictions.csv", predictions)
        else:
            patience += 1
        if patience >= early_stopping_patience:
            break

    summary = {
        **base_state,
        "epochs_completed": len(history),
        "best_values": {
            "clean": None if best_clean == -math.inf else best_clean,
            "robustness": None,
            "selection_score": None,
        },
        "best_clean_layer_importance": best_layer_importance,
        "final_layer_importance": history[-1]["layer_importance"],
        "physical_batch_size": physical_batch_size,
        "effective_batch_size": effective_batch_size,
        "accumulation_steps": accumulation_steps,
        "warmup_steps": warmup_steps,
        "history": history,
        "robustness_pending_task3": True,
    }
    (output_directory / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
