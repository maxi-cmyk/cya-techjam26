"""Controlled RINE plus frequency/Lab robustness fusion training."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from cya_detector.evaluation.metrics import evaluate_predictions
from cya_detector.models.clip_baseline import require_ml_dependencies
from cya_detector.models.rine import build_rine_auxiliary_fusion, build_rine_head
from cya_detector.predictions import PredictionRecord, write_predictions
from cya_detector.training.auxiliary_stage_c import BASE_FIELDS as AUXILIARY_BASE_FIELDS
from cya_detector.training.clip_stage_a import CachedEmbedding
from cya_detector.training.frequency_stage1 import BASE_FIELDS as FREQUENCY_BASE_FIELDS
from cya_detector.training.robustness import controlled_epoch_rows
from cya_detector.transforms.benchmark import TransformCell


FUSION_VARIANTS = frozenset({"frequency", "lab", "frequency_lab"})


@dataclass(frozen=True)
class TabularFeatureBank:
    names: tuple[str, ...]
    values_by_sample_id: dict[str, np.ndarray]


def _read_table(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        names = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"Feature table is empty: {path}")
    if any(row.get("split") == "final_test" for row in rows):
        raise ValueError(f"Robustness feature table contains final_test rows: {path}")
    if any(row.get("feature_valid") != "true" for row in rows):
        raise ValueError(f"Robustness feature table contains invalid rows: {path}")
    return rows, names


def load_tabular_feature_bank(
    *,
    variant: str,
    frequency_table: Path | None,
    auxiliary_table: Path | None,
) -> TabularFeatureBank:
    """Load only the retained magnitude/residual and/or Lab feature families."""

    if variant not in FUSION_VARIANTS:
        raise ValueError(f"Unsupported robustness fusion variant: {variant}")
    vectors: dict[str, list[float]] = {}
    labels: dict[str, tuple[str, str]] = {}
    feature_names: list[str] = []

    def add_table(path: Path, *, selected_names: list[str], prefix: str) -> set[str]:
        rows, _ = _read_table(path)
        if not selected_names:
            raise ValueError(f"No retained {prefix} features found in {path}")
        feature_names.extend(f"{prefix}:{name}" for name in selected_names)
        observed: set[str] = set()
        for row in rows:
            sample_id = row.get("sample_id", "")
            if not sample_id or sample_id in observed:
                raise ValueError(f"Duplicate or empty sample_id in {path}: {sample_id!r}")
            observed.add(sample_id)
            identity = (row.get("label", ""), row.get("split", ""))
            previous = labels.setdefault(sample_id, identity)
            if previous != identity:
                raise ValueError(f"Feature tables disagree for sample {sample_id!r}")
            vectors.setdefault(sample_id, []).extend(float(row[name]) for name in selected_names)
        return observed

    if variant in {"frequency", "frequency_lab"}:
        if frequency_table is None:
            raise ValueError("Frequency fusion requires --frequency-table")
        frequency_rows, frequency_fields = _read_table(frequency_table)
        del frequency_rows
        selected = [
            name
            for name in frequency_fields
            if name not in FREQUENCY_BASE_FIELDS and not name.startswith("phase_")
        ]
        add_table(frequency_table, selected_names=selected, prefix="frequency")

    if variant in {"lab", "frequency_lab"}:
        if auxiliary_table is None:
            raise ValueError("Lab fusion requires --auxiliary-table")
        auxiliary_rows, auxiliary_fields = _read_table(auxiliary_table)
        del auxiliary_rows
        selected = [
            name
            for name in auxiliary_fields
            if name not in AUXILIARY_BASE_FIELDS and name.startswith("lab_")
        ]
        before = set(vectors)
        observed = add_table(auxiliary_table, selected_names=selected, prefix="lab")
        if before and observed != before:
            raise ValueError("Frequency and Lab feature tables have different sample sets")

    return TabularFeatureBank(
        names=tuple(feature_names),
        values_by_sample_id={
            sample_id: np.asarray(values, dtype=np.float32) for sample_id, values in vectors.items()
        },
    )


def _load_rine_features(rows: list[CachedEmbedding]) -> Any:
    torch, _, _ = require_ml_dependencies()
    return torch.stack(
        [torch.load(row.cache_path, map_location="cpu", weights_only=True) for row in rows]
    )


def _auxiliary_matrix(
    rows: list[CachedEmbedding],
    bank: TabularFeatureBank,
) -> np.ndarray:
    missing = [
        row.example.sample_id
        for row in rows
        if row.example.sample_id not in bank.values_by_sample_id
    ]
    if missing:
        raise ValueError(f"Missing auxiliary features for {len(missing)} RINE rows")
    return np.stack([bank.values_by_sample_id[row.example.sample_id] for row in rows])


def _predict(
    *,
    model: Any,
    rine_features: Any,
    auxiliary_features: Any,
    rows: list[CachedEmbedding],
    checkpoint: str,
    seed: int,
    matching_policy: str,
    device: str,
) -> list[PredictionRecord]:
    torch, _, _ = require_ml_dependencies()
    model.eval()
    with torch.inference_mode():
        logits = (
            model(rine_features.to(device), auxiliary_features.to(device)).squeeze(1).detach().cpu()
        )
        probabilities = torch.sigmoid(logits)
    return [
        PredictionRecord(
            sample_id=row.example.sample_id,
            source_id=row.example.source_id,
            parent_id=row.example.parent_id,
            split=row.example.split,
            label=row.example.label,
            logit=float(logit),
            probability=float(probability),
            checkpoint=checkpoint,
            seed=seed,
            matching_policy=matching_policy,
            transform=row.example.transform,
            transform_parameter=row.example.transform_parameter,
            **row.example.metadata,
        )
        for row, logit, probability in zip(
            rows,
            logits.tolist(),
            probabilities.tolist(),
            strict=True,
        )
    ]


def _save_checkpoint(path: Path, *, model: Any, state: dict[str, Any]) -> None:
    torch, _, _ = require_ml_dependencies()
    temporary = path.with_suffix(".tmp.pt")
    torch.save({"model_state_dict": model.state_dict(), **state}, temporary)
    temporary.replace(path)


def train_controlled_rine_auxiliary_fusion(
    *,
    train_parent_rows: list[CachedEmbedding],
    train_bank_rows: list[CachedEmbedding],
    selection_rows: list[CachedEmbedding],
    feature_bank: TabularFeatureBank,
    parent_checkpoint: Path,
    cells: list[TransformCell] | tuple[TransformCell, ...],
    output_directory: Path,
    variant: str,
    matching_policy: str,
    layers: list[int] | tuple[int, ...],
    seed: int,
    device: str,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    early_stopping_patience: int,
    physical_batch_size: int,
    effective_batch_size: int,
    threshold: float,
    epoch_size: int | None = None,
) -> dict[str, Any]:
    """Fit one auxiliary projection and fusion head over a frozen RINE parent."""

    torch, _, _ = require_ml_dependencies()
    if variant not in FUSION_VARIANTS:
        raise ValueError(f"Unsupported robustness fusion variant: {variant}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    resolved_epoch_size = epoch_size or len(train_parent_rows)
    scaler_rows = list(
        controlled_epoch_rows(
            train_parent_rows,
            train_bank_rows,
            cells,
            epoch_size=resolved_epoch_size,
            project_seed=seed,
            epoch=0,
        )
    )
    scaler = StandardScaler().fit(_auxiliary_matrix(scaler_rows, feature_bank))

    first = _load_rine_features([train_bank_rows[0]])
    layer_count, hidden_dimension = first.shape[1:]
    if layer_count != len(layers):
        raise ValueError("Cached RINE layer count does not match configuration")
    global_model = build_rine_head(
        layer_count=layer_count,
        hidden_dimension=hidden_dimension,
    )
    parent = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
    parent_state = parent.get("model_state_dict")
    if not isinstance(parent_state, dict):
        raise ValueError("Controlled RINE parent checkpoint has no model_state_dict")
    global_model.load_state_dict(parent_state, strict=True)
    model = build_rine_auxiliary_fusion(
        global_model=global_model,
        auxiliary_dimension=len(feature_bank.names),
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    accumulation_steps = max(1, math.ceil(effective_batch_size / physical_batch_size))

    selection_rine = _load_rine_features(selection_rows)
    selection_auxiliary = torch.tensor(
        scaler.transform(_auxiliary_matrix(selection_rows, feature_bank)),
        dtype=torch.float32,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_directory / "scaler.joblib")
    (output_directory / "feature_schema.json").write_text(
        json.dumps(
            {"variant": variant, "feature_names": feature_bank.names},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    best_score = -math.inf
    patience = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        rows = list(
            controlled_epoch_rows(
                train_parent_rows,
                train_bank_rows,
                cells,
                epoch_size=resolved_epoch_size,
                project_seed=seed,
                epoch=epoch - 1,
            )
        )
        rine = _load_rine_features(rows)
        auxiliary = torch.tensor(
            scaler.transform(_auxiliary_matrix(rows, feature_bank)),
            dtype=torch.float32,
        )
        targets = torch.tensor(
            [row.example.target for row in rows],
            dtype=torch.float32,
        )
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(rine, auxiliary, targets),
            batch_size=physical_batch_size,
            shuffle=False,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for step, (rine_batch, auxiliary_batch, targets_batch) in enumerate(loader, start=1):
            logits = model(rine_batch.to(device), auxiliary_batch.to(device)).squeeze(1)
            loss = criterion(logits, targets_batch.to(device)) / accumulation_steps
            loss.backward()
            total_loss += loss.item() * accumulation_steps
            if step % accumulation_steps == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        predictions = _predict(
            model=model,
            rine_features=selection_rine,
            auxiliary_features=selection_auxiliary,
            rows=selection_rows,
            checkpoint=f"epoch_{epoch:03d}",
            seed=seed,
            matching_policy=matching_policy,
            device=device,
        )
        report = evaluate_predictions(predictions, threshold=threshold)
        score = report["selection_score"]
        if score is None:
            raise ValueError("Fusion selection requires clean and robustness cells")
        history.append(
            {
                "epoch": epoch,
                "training_loss": total_loss / len(loader),
                "clean_accuracy": report["clean"]["accuracy"],
                "robustness_mean_accuracy": report["robustness"]["mean_accuracy"],
                "selection_score": score,
            }
        )
        state = {
            "stage": "rine_auxiliary_robustness_fusion",
            "variant": variant,
            "seed": seed,
            "epoch": epoch,
            "matching_policy": matching_policy,
            "parent_checkpoint": str(parent_checkpoint.resolve()),
            "feature_names": list(feature_bank.names),
            "selection_metrics": report,
        }
        _save_checkpoint(output_directory / "latest.pt", model=model, state=state)
        write_predictions(output_directory / "latest_predictions.csv", predictions)
        if score > best_score:
            best_score = score
            patience = 0
            _save_checkpoint(output_directory / "best_50_50.pt", model=model, state=state)
            write_predictions(output_directory / "best_50_50_predictions.csv", predictions)
        else:
            patience += 1
        if patience >= early_stopping_patience:
            break

    summary = {
        "stage": "rine_auxiliary_robustness_fusion",
        "variant": variant,
        "seed": seed,
        "parent_checkpoint": str(parent_checkpoint.resolve()),
        "feature_names": list(feature_bank.names),
        "epochs_completed": len(history),
        "best_selection_score": None if best_score == -math.inf else best_score,
        "history": history,
        "final_test_read": False,
    }
    (output_directory / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
