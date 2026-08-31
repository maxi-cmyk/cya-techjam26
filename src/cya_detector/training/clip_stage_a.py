"""Embedding extraction and linear-probe training for frozen CLIP Stage A."""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cya_detector.data.dataset import ClipImageDataset, ManifestExample
from cya_detector.evaluation.metrics import evaluate_predictions
from cya_detector.models.clip_baseline import (
    LoadedClip,
    assert_only_head_trainable,
    build_binary_head,
    embedding_cache_key,
    require_ml_dependencies,
)
from cya_detector.predictions import PredictionRecord, write_predictions


@dataclass(frozen=True)
class CachedEmbedding:
    example: ManifestExample
    cache_key: str
    cache_path: Path


def _collate_images(batch: list[dict[str, Any]]) -> dict[str, Any]:
    torch, _, _ = require_ml_dependencies()
    return {
        "pixel_values": torch.stack([row["pixel_values"] for row in batch]),
        "examples": [row["example"] for row in batch],
    }


def _view_identifier(example: ManifestExample, matching_policy: str) -> str:
    return ":".join(
        (
            matching_policy,
            example.image_view,
            example.transform or "clean",
            example.transform_parameter or "default",
        )
    )


def cache_location(cache_root: Path, cache_key: str) -> Path:
    return cache_root / cache_key[:2] / f"{cache_key}.pt"


def extract_embeddings(
    *,
    loaded_clip: LoadedClip,
    examples: list[ManifestExample],
    cache_root: Path,
    matching_policy: str,
    preprocessing_version: str,
    batch_size: int,
    device: str,
) -> tuple[list[CachedEmbedding], dict[str, Any]]:
    """Populate immutable per-view embedding cache entries and report throughput."""

    torch, _, _ = require_ml_dependencies()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if any(not example.sha256 for example in examples):
        raise ValueError("Every cacheable example must have an image SHA-256")

    cached: list[CachedEmbedding] = []
    missing: list[ManifestExample] = []
    key_by_sample: dict[str, str] = {}
    for example in examples:
        key = embedding_cache_key(
            image_sha256=example.sha256,
            model_identifier=loaded_clip.identifier,
            resolved_revision=loaded_clip.resolved_revision,
            preprocessing_version=preprocessing_version,
            view_identifier=_view_identifier(example, matching_policy),
        )
        key_by_sample[example.sample_id] = key
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
                desc=f"CLIP {matching_policy}",
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
                    embeddings = loaded_clip.model(pixel_values=pixels).image_embeds
                embeddings = embeddings.detach().float().cpu()
                for example, embedding in zip(batch["examples"], embeddings, strict=True):
                    key = key_by_sample[example.sample_id]
                    path = cache_location(cache_root, key)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(".tmp.pt")
                    torch.save(embedding, temporary)
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
        "requested_revision": loaded_clip.requested_revision,
        "resolved_revision": loaded_clip.resolved_revision,
        "preprocessing_version": preprocessing_version,
        "matching_policy": matching_policy,
        "batch_size": batch_size,
    }
    return cached, report


def load_cached_tensors(rows: list[CachedEmbedding]) -> tuple[Any, Any]:
    torch, _, _ = require_ml_dependencies()
    embeddings = torch.stack(
        [torch.load(row.cache_path, map_location="cpu", weights_only=True) for row in rows]
    )
    targets = torch.tensor([row.example.target for row in rows], dtype=torch.float32)
    return embeddings, targets


def _predictions(
    *,
    head: Any,
    embeddings: Any,
    rows: list[CachedEmbedding],
    checkpoint: str,
    seed: int,
    matching_policy: str,
    device: str,
) -> list[PredictionRecord]:
    torch, _, _ = require_ml_dependencies()
    head.eval()
    with torch.inference_mode():
        logits = head(embeddings.to(device)).squeeze(1).detach().cpu()
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


def _save_checkpoint(path: Path, *, head: Any, state: dict[str, Any]) -> None:
    torch, _, _ = require_ml_dependencies()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.pt")
    torch.save({"head_state_dict": head.state_dict(), **state}, temporary)
    temporary.replace(path)


def predict_linear_probe_checkpoint(
    *,
    checkpoint_path: Path,
    rows: list[CachedEmbedding],
    seed: int,
    matching_policy: str,
    device: str,
) -> tuple[list[PredictionRecord], dict[str, Any]]:
    """Load one Stage A head and score a clean-plus-transform embedding bank."""

    torch, _, _ = require_ml_dependencies()
    if not rows:
        raise ValueError("Stage A checkpoint evaluation requires embedding rows")
    embeddings, _ = load_cached_tensors(rows)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("head_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"Stage A checkpoint has no head_state_dict: {checkpoint_path}")
    hidden_dimension = checkpoint.get("hidden_dimension")
    head = build_binary_head(
        embeddings.shape[1],
        hidden_dimension=hidden_dimension,
    ).to(device)
    head.load_state_dict(state, strict=True)
    predictions = _predictions(
        head=head,
        embeddings=embeddings,
        rows=rows,
        checkpoint=str(checkpoint_path.resolve()),
        seed=seed,
        matching_policy=matching_policy,
        device=device,
    )
    metadata = {
        key: value
        for key, value in checkpoint.items()
        if key != "head_state_dict"
    }
    return predictions, metadata


def train_linear_probe(
    *,
    train_rows: list[CachedEmbedding],
    selection_rows: list[CachedEmbedding],
    output_directory: Path,
    matching_policy: str,
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
    hidden_dimension: int | None = None,
) -> dict[str, Any]:
    """Train only the binary head and select checkpoints on selection_val."""

    torch, _, _ = require_ml_dependencies()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_embeddings, train_targets = load_cached_tensors(train_rows)
    selection_embeddings, _ = load_cached_tensors(selection_rows)
    if set(train_targets.tolist()) != {0.0, 1.0}:
        raise ValueError("Training data must contain both binary classes")

    head = build_binary_head(train_embeddings.shape[1], hidden_dimension=hidden_dimension).to(device)
    class _FrozenEncoderSentinel:
        @staticmethod
        def parameters() -> list[Any]:
            return []

    assert_only_head_trainable(_FrozenEncoderSentinel(), head)
    optimizer = torch.optim.AdamW(
        head.parameters(), learning_rate, weight_decay=weight_decay
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    accumulation_steps = max(1, math.ceil(effective_batch_size / physical_batch_size))
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_embeddings, train_targets),
        batch_size=physical_batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer_steps_per_epoch = math.ceil(len(loader) / accumulation_steps)
    total_optimizer_steps = max(1, optimizer_steps_per_epoch * max_epochs)
    warmup_steps = int(total_optimizer_steps * warmup_fraction)

    def learning_rate_multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)

    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_state = {
        "seed": seed,
        "matching_policy": matching_policy,
        "resolved_model_revision": resolved_revision,
        "manifest_sha256": manifest_sha256,
        "embedding_dimension": int(train_embeddings.shape[1]),
        "head_type": "linear" if hidden_dimension is None else "mlp",
        "hidden_dimension": hidden_dimension,
        "threshold": threshold,
        "resolved_run_configuration": run_configuration,
    }
    history: list[dict[str, Any]] = []
    best_values = {"clean": -math.inf, "robustness": -math.inf, "selection_score": -math.inf}
    patience = 0
    for epoch in range(1, max_epochs + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for step, (embeddings, targets) in enumerate(loader, start=1):
            logits = head(embeddings.to(device)).squeeze(1)
            loss = criterion(logits, targets.to(device)) / accumulation_steps
            loss.backward()
            total_loss += loss.item() * accumulation_steps
            if step % accumulation_steps == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        checkpoint_name = f"epoch_{epoch:03d}"
        predictions = _predictions(
            head=head,
            embeddings=selection_embeddings,
            rows=selection_rows,
            checkpoint=checkpoint_name,
            seed=seed,
            matching_policy=matching_policy,
            device=device,
        )
        report = evaluate_predictions(predictions, threshold=threshold)
        clean_accuracy = report["clean"]["accuracy"] if report["clean"] else None
        robust_accuracy = report["robustness"]["mean_accuracy"]
        score = report["selection_score"]
        history.append(
            {
                "epoch": epoch,
                "training_loss": total_loss / len(loader),
                "clean_accuracy": clean_accuracy,
                "robustness_mean_accuracy": robust_accuracy,
                "selection_score": score,
            }
        )
        state = {**checkpoint_state, "epoch": epoch, "selection_metrics": report}
        _save_checkpoint(output_directory / "latest.pt", head=head, state=state)

        improved = False
        candidates = {
            "clean": clean_accuracy,
            "robustness": robust_accuracy,
            "selection_score": score,
        }
        checkpoint_files = {
            "clean": "best_clean.pt",
            "robustness": "best_robustness.pt",
            "selection_score": "best_50_50.pt",
        }
        prediction_files = {
            "clean": "best_clean_predictions.csv",
            "robustness": "best_robustness_predictions.csv",
            "selection_score": "best_50_50_predictions.csv",
        }
        for name, value in candidates.items():
            if value is not None and value > best_values[name]:
                best_values[name] = value
                _save_checkpoint(output_directory / checkpoint_files[name], head=head, state=state)
                write_predictions(output_directory / prediction_files[name], predictions)
                improved = True
        patience = 0 if improved else patience + 1
        write_predictions(output_directory / "latest_predictions.csv", predictions)
        if patience >= early_stopping_patience:
            break

    summary = {
        **checkpoint_state,
        "epochs_completed": len(history),
        "physical_batch_size": physical_batch_size,
        "effective_batch_size": effective_batch_size,
        "accumulation_steps": accumulation_steps,
        "warmup_fraction": warmup_fraction,
        "warmup_steps": warmup_steps,
        "best_values": {key: (None if value == -math.inf else value) for key, value in best_values.items()},
        "history": history,
        "robustness_checkpoint_available": (output_directory / "best_robustness.pt").is_file(),
        "selection_score_checkpoint_available": (output_directory / "best_50_50.pt").is_file(),
    }
    (output_directory / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
