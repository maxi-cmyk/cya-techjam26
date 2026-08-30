"""Contracts and feature-bank sampling for the Task 3 robustness milestone."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from cya_detector.data.dataset import ManifestExample
from cya_detector.transforms.benchmark import TransformCell
from cya_detector.transforms.controlled import build_controlled_epoch


class RobustnessContractError(ValueError):
    """Raised when clean and transformed views cannot form a locked benchmark."""


T = TypeVar("T")


@dataclass(frozen=True)
class RobustnessBank:
    """Validated clean parents and their complete independent transform bank."""

    clean: tuple[ManifestExample, ...]
    variants: tuple[ManifestExample, ...]
    cell_ids: tuple[str, ...]

    @property
    def all_examples(self) -> tuple[ManifestExample, ...]:
        return self.clean + self.variants


def transform_cell_id(example: ManifestExample) -> str:
    """Return the stable Task 3 cell ID encoded in one manifest example."""

    if (example.transform or "clean") == "clean":
        return "clean"
    try:
        value = json.loads(example.transform_parameter)
    except (json.JSONDecodeError, TypeError):
        value = None
    if isinstance(value, dict) and isinstance(value.get("cell_id"), str):
        cell_id = value["cell_id"].strip()
        if cell_id:
            return cell_id
    raise RobustnessContractError(
        f"Robustness row {example.sample_id!r} has no encoded transform cell_id"
    )


def validate_robustness_bank(
    clean_examples: Sequence[ManifestExample],
    variant_examples: Sequence[ManifestExample],
    cells: Sequence[TransformCell],
    *,
    split: str,
) -> RobustnessBank:
    """Validate one split without allowing missing, extra, chained, or crossed views."""

    if split == "final_test":
        raise RobustnessContractError("The robustness milestone cannot read final_test")
    expected_cell_ids = tuple(cell.cell_id for cell in cells)
    if not expected_cell_ids or len(set(expected_cell_ids)) != len(expected_cell_ids):
        raise RobustnessContractError("Configured robustness cells must be nonempty and unique")

    clean_by_id: dict[str, ManifestExample] = {}
    source_ids: set[str] = set()
    for example in clean_examples:
        if example.split != split:
            raise RobustnessContractError(
                f"Clean row {example.sample_id!r} is in {example.split!r}, expected {split!r}"
            )
        if example.image_view != "matched_clean" or transform_cell_id(example) != "clean":
            raise RobustnessContractError(
                f"Clean row {example.sample_id!r} is not a matched-clean parent"
            )
        if example.sample_id in clean_by_id:
            raise RobustnessContractError(f"Duplicate clean sample_id: {example.sample_id!r}")
        source_unit = example.source_id or example.sample_id
        if source_unit in source_ids:
            raise RobustnessContractError(f"Duplicate clean source_id: {source_unit!r}")
        clean_by_id[example.sample_id] = example
        source_ids.add(source_unit)
    if not clean_by_id:
        raise RobustnessContractError(f"No matched-clean parents found for {split!r}")

    variants_by_parent: dict[str, dict[str, ManifestExample]] = {
        sample_id: {} for sample_id in clean_by_id
    }
    for example in variant_examples:
        if example.split != split:
            raise RobustnessContractError(
                f"Variant row {example.sample_id!r} is in {example.split!r}, expected {split!r}"
            )
        if example.image_view != "benchmark":
            raise RobustnessContractError(
                f"Variant row {example.sample_id!r} is not a benchmark view"
            )
        parent = clean_by_id.get(example.parent_id)
        if parent is None:
            raise RobustnessContractError(
                f"Variant row {example.sample_id!r} has unknown parent {example.parent_id!r}"
            )
        if (example.source_id, example.label, example.split) != (
            parent.source_id,
            parent.label,
            parent.split,
        ):
            raise RobustnessContractError(
                f"Variant row {example.sample_id!r} crosses its parent contract"
            )
        cell_id = transform_cell_id(example)
        if cell_id not in expected_cell_ids:
            raise RobustnessContractError(
                f"Variant row {example.sample_id!r} uses undeclared cell {cell_id!r}"
            )
        if cell_id in variants_by_parent[example.parent_id]:
            raise RobustnessContractError(
                f"Duplicate variant for parent/cell: {example.parent_id!r}/{cell_id!r}"
            )
        variants_by_parent[example.parent_id][cell_id] = example

    expected = set(expected_cell_ids)
    for parent_id, observed in variants_by_parent.items():
        missing = sorted(expected - observed.keys())
        if missing:
            raise RobustnessContractError(
                f"Parent {parent_id!r} is missing robustness cells: {', '.join(missing)}"
            )

    ordered_clean = tuple(sorted(clean_by_id.values(), key=lambda row: row.sample_id))
    ordered_variants = tuple(
        variants_by_parent[parent.sample_id][cell_id]
        for parent in ordered_clean
        for cell_id in expected_cell_ids
    )
    return RobustnessBank(ordered_clean, ordered_variants, expected_cell_ids)


def _row_example(row: Any) -> ManifestExample:
    example = getattr(row, "example", None)
    if not isinstance(example, ManifestExample):
        raise RobustnessContractError("Feature-bank rows must expose a ManifestExample")
    return example


def controlled_epoch_rows(
    clean_parent_rows: Sequence[T],
    bank_rows: Sequence[T],
    cells: Sequence[TransformCell],
    *,
    epoch_size: int,
    project_seed: int,
    epoch: int,
) -> tuple[T, ...]:
    """Select cached clean/variant feature rows using the Task 3 epoch schedule."""

    clean_examples = [_row_example(row) for row in clean_parent_rows]
    parent_records: list[Mapping[str, str]] = [
        {
            "sample_id": example.sample_id,
            "label": example.label,
            "image_path": str(example.image_path),
            "image_view": example.image_view,
            "transform": example.transform,
        }
        for example in clean_examples
    ]
    schedule = build_controlled_epoch(
        parent_records,
        cells,
        epoch_size=epoch_size,
        project_seed=project_seed,
        epoch=epoch,
    )

    lookup: dict[tuple[str, str], T] = {}
    for row in bank_rows:
        example = _row_example(row)
        cell_id = transform_cell_id(example)
        parent_id = example.sample_id if cell_id == "clean" else example.parent_id
        key = (parent_id, cell_id)
        if key in lookup:
            raise RobustnessContractError(f"Duplicate cached feature row for {key!r}")
        lookup[key] = row

    selected: list[T] = []
    for view in schedule:
        key = (view.sample_id, view.cell_id)
        row = lookup.get(key)
        if row is None:
            raise RobustnessContractError(f"Missing cached feature row for {key!r}")
        if _row_example(row).label != view.label:
            raise RobustnessContractError(f"Cached feature label mismatch for {key!r}")
        selected.append(row)

    counts = Counter((_row_example(row).label, transform_cell_id(_row_example(row)) == "clean") for row in selected)
    if counts and max(counts.values()) - min(counts.values()) > 1:
        raise RobustnessContractError("Controlled epoch lost label/view balance")
    return tuple(selected)
