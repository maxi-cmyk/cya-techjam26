"""Deterministic, lazy scheduling for controlled transform training."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from PIL import Image

from cya_detector.transforms.benchmark import (
    TransformCell,
    apply_benchmark,
    derive_seed,
    validate_parent_record,
)
from cya_detector.transforms.preprocessing import random_crop_input

LABELS = ("authentic", "ai_generated")
CLEAN_CELL_ID = "clean"


@dataclass(frozen=True)
class TrainingView:
    """One parent image and optional benchmark cell selected for training."""

    sample_id: str
    label: str
    image_path: str
    cell_id: str
    seed: int


def _validated_parent_pools(
    records: Sequence[Mapping[str, str]],
) -> dict[str, list[Mapping[str, str]]]:
    pools: dict[str, list[Mapping[str, str]]] = {label: [] for label in LABELS}
    for record in records:
        validate_parent_record(dict(record))
        label = record.get("label")
        if label not in pools:
            raise ValueError(f"Unsupported controlled-training label: {label!r}")
        for field in ("sample_id", "image_path"):
            value = record.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Controlled-training parent requires nonempty {field!r}")
        pools[label].append(record)

    missing = [label for label in LABELS if not pools[label]]
    if missing:
        raise ValueError(f"Controlled training requires parent records for: {', '.join(missing)}")
    return pools


def _validated_cells(cells: Sequence[TransformCell]) -> list[TransformCell]:
    if not cells:
        raise ValueError("Controlled training requires at least one transform cell")

    validated: list[TransformCell] = []
    seen: set[str] = set()
    for cell in cells:
        if not isinstance(cell, TransformCell):
            raise TypeError(f"Invalid transform cell: {cell!r}")
        if not cell.cell_id or cell.cell_id == CLEAN_CELL_ID:
            raise ValueError(f"Invalid transform cell_id: {cell.cell_id!r}")
        if cell.cell_id in seen:
            raise ValueError(f"Duplicate transform cell_id: {cell.cell_id!r}")
        seen.add(cell.cell_id)
        validated.append(cell)
    return validated


def _shuffled(
    values: Sequence[Mapping[str, str]] | Sequence[TransformCell],
    *,
    seed: int,
) -> list[Mapping[str, str]] | list[TransformCell]:
    shuffled = list(values)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def build_controlled_epoch(
    records: Sequence[Mapping[str, str]],
    cells: Sequence[TransformCell],
    *,
    epoch_size: int,
    project_seed: int,
    epoch: int,
) -> tuple[TrainingView, ...]:
    """Build a balanced epoch description without reading any parent image."""

    if isinstance(epoch_size, bool) or not isinstance(epoch_size, int) or epoch_size <= 0:
        raise ValueError(f"epoch_size must be a positive integer, got {epoch_size!r}")

    parent_pools = _validated_parent_pools(records)
    transform_cells = _validated_cells(cells)
    epoch_key = f"controlled-epoch:{epoch}"

    for label in LABELS:
        parent_pools[label] = _shuffled(
            sorted(parent_pools[label], key=lambda row: (row["sample_id"], row["image_path"])),
            seed=derive_seed(project_seed, epoch_key, f"parents:{label}"),
        )
    cell_pools = {
        label: _shuffled(
            sorted(transform_cells, key=lambda cell: cell.cell_id),
            seed=derive_seed(project_seed, epoch_key, f"cells:{label}"),
        )
        for label in LABELS
    }

    parent_positions = {label: 0 for label in LABELS}
    cell_positions = {label: 0 for label in LABELS}
    schedule: list[TrainingView] = []
    for slot in range(epoch_size):
        label = LABELS[slot % len(LABELS)]
        parents = parent_pools[label]
        parent = parents[parent_positions[label] % len(parents)]
        parent_positions[label] += 1

        is_clean = slot % 4 in (0, 3)
        if is_clean:
            cell_id = CLEAN_CELL_ID
        else:
            label_cells = cell_pools[label]
            cell_id = label_cells[cell_positions[label] % len(label_cells)].cell_id
            cell_positions[label] += 1

        sample_id = parent["sample_id"]
        view_seed = derive_seed(
            project_seed,
            sample_id,
            f"{epoch_key}:slot:{slot}:cell:{cell_id}",
        )
        schedule.append(
            TrainingView(
                sample_id=sample_id,
                label=label,
                image_path=parent["image_path"],
                cell_id=cell_id,
                seed=view_seed,
            )
        )
    return tuple(schedule)


def apply_training_view(
    view: TrainingView,
    cells_by_id: Mapping[str, TransformCell],
    *,
    input_size: int,
) -> Image.Image:
    """Read and realize one scheduled view, then take its seeded input crop."""

    if view.label not in LABELS:
        raise ValueError(f"Unsupported controlled-training label: {view.label!r}")

    cell: TransformCell | None = None
    if view.cell_id != CLEAN_CELL_ID:
        cell = cells_by_id.get(view.cell_id)
        if not isinstance(cell, TransformCell) or cell.cell_id != view.cell_id:
            raise ValueError(f"Unknown or invalid transform cell_id: {view.cell_id!r}")

    with Image.open(view.image_path) as parent:
        image = parent.copy()

    if cell is not None:
        image = apply_benchmark(image, cell, view.sample_id, view.seed).image

    crop_seed = derive_seed(
        view.seed,
        view.sample_id,
        f"{view.cell_id}:input_crop",
    )
    return random_crop_input(image, input_size, seed=crop_seed)
