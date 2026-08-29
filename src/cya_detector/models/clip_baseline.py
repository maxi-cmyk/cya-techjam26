"""Frozen CLIP vision encoder and Stage A binary heads."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def require_ml_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    except ImportError as exc:
        raise RuntimeError(
            "Task 4 requires torch and transformers. In Colab run `make install-colab`."
        ) from exc
    return torch, CLIPImageProcessor, CLIPVisionModelWithProjection


def embedding_cache_key(
    *,
    image_sha256: str,
    model_identifier: str,
    resolved_revision: str,
    preprocessing_version: str,
    view_identifier: str,
) -> str:
    """Build a content-addressed key; no cache is valid across these boundaries."""

    if not all(
        (image_sha256, model_identifier, resolved_revision, preprocessing_version, view_identifier)
    ):
        raise ValueError("All embedding cache-key fields are required")
    payload = {
        "image_sha256": image_sha256,
        "model_identifier": model_identifier,
        "preprocessing_version": preprocessing_version,
        "resolved_revision": resolved_revision,
        "view_identifier": view_identifier,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class LoadedClip:
    model: Any
    processor: Any
    identifier: str
    requested_revision: str
    resolved_revision: str
    embedding_dimension: int


def load_frozen_clip(
    identifier: str,
    *,
    revision: str = "main",
    device: str | None = None,
) -> LoadedClip:
    """Load only CLIP's vision tower and fail if its freeze invariant is broken."""

    torch, processor_type, model_type = require_ml_dependencies()
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model_type.from_pretrained(identifier, revision=revision)
    resolved = getattr(model.config, "_commit_hash", None) or revision
    if revision in {"main", "master"} and resolved == revision:
        raise RuntimeError(
            "The model host did not expose a resolved commit; refusing to create mutable caches"
        )
    processor = processor_type.from_pretrained(identifier, revision=resolved)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(selected_device)

    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("CLIP backbone must remain completely frozen")
    dimension = int(model.config.projection_dim)
    return LoadedClip(
        model=model,
        processor=processor,
        identifier=identifier,
        requested_revision=revision,
        resolved_revision=resolved,
        embedding_dimension=dimension,
    )


def build_binary_head(embedding_dimension: int, *, hidden_dimension: int | None = None) -> Any:
    """Create the linear baseline, or an explicitly requested small MLP ablation."""

    torch, _, _ = require_ml_dependencies()
    if hidden_dimension is None:
        return torch.nn.Linear(embedding_dimension, 1)
    if hidden_dimension <= 0:
        raise ValueError("hidden_dimension must be positive")
    return torch.nn.Sequential(
        torch.nn.Linear(embedding_dimension, hidden_dimension),
        torch.nn.GELU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(hidden_dimension, 1),
    )


def assert_only_head_trainable(encoder: Any, head: Any) -> None:
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("Frozen encoder has a trainable parameter")
    if not any(parameter.requires_grad for parameter in head.parameters()):
        raise RuntimeError("Binary head has no trainable parameters")


def write_cache_metadata(path: Path, entries: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(entries)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
