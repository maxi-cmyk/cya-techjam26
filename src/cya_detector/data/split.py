"""Deterministic, source-grouped dataset splitting."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cya_detector.data.manifest import read_manifest, sha256_file, write_json, write_manifest


DEFAULT_SPLIT_FRACTIONS = {
    "seed_train": 0.60,
    "self_train_pool": 0.25,
    "selection_val": 0.075,
    "final_test": 0.075,
}


class SplitContractError(ValueError):
    """Raised when grouped splitting would violate the dataset contract."""


def _validate_fractions(fractions: dict[str, float]) -> None:
    if set(fractions) != set(DEFAULT_SPLIT_FRACTIONS):
        raise SplitContractError(f"Expected split names: {sorted(DEFAULT_SPLIT_FRACTIONS)}")
    if any(value < 0 for value in fractions.values()):
        raise SplitContractError("Split fractions cannot be negative")
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise SplitContractError("Split fractions must sum to 1.0")


def _allocate_counts(total: int, fractions: dict[str, float]) -> dict[str, int]:
    raw = {name: total * fraction for name, fraction in fractions.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(fractions, key=lambda name: (raw[name] - counts[name], name), reverse=True)
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def assign_grouped_splits(
    records: list[dict[str, str]],
    *,
    seed: int = 42,
    fractions: dict[str, float] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    fractions = fractions or DEFAULT_SPLIT_FRACTIONS
    _validate_fractions(fractions)

    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        group_key = record.get("duplicate_group_id") or record["source_id"]
        groups[group_key].append(index)

    groups_by_label: dict[str, list[str]] = defaultdict(list)
    excluded_groups = 0
    for group_key, indices in groups.items():
        labels = {records[index]["label"] for index in indices}
        if len(labels) != 1:
            for index in indices:
                records[index]["split"] = "excluded_review"
                records[index]["eligible_for_split"] = "false"
            excluded_groups += 1
            continue

        if not any(records[index].get("eligible_for_split") == "true" for index in indices):
            for index in indices:
                records[index]["split"] = "excluded"
            excluded_groups += 1
            continue
        groups_by_label[next(iter(labels))].append(group_key)

    for label, group_keys in sorted(groups_by_label.items()):
        ordered = sorted(
            group_keys,
            key=lambda key: hashlib.sha256(f"{seed}:{label}:{key}".encode()).hexdigest(),
        )
        counts = _allocate_counts(len(ordered), fractions)
        cursor = 0
        for split_name in fractions:
            selected = ordered[cursor : cursor + counts[split_name]]
            cursor += counts[split_name]
            for group_key in selected:
                for index in groups[group_key]:
                    records[index]["split"] = split_name

    split_label_counts = Counter(
        (record.get("split", ""), record["label"])
        for record in records
        if record.get("eligible_for_split") == "true"
    )
    report = {
        "seed": seed,
        "fractions": fractions,
        "group_count": len(groups),
        "excluded_group_count": excluded_groups,
        "eligible_record_count": sum(
            record.get("eligible_for_split") == "true" for record in records
        ),
        "counts": {
            f"{split_name}:{label}": count
            for (split_name, label), count in sorted(split_label_counts.items())
        },
    }
    return records, report


def split_manifest_file(
    *,
    input_path: Path,
    output_path: Path,
    report_path: Path,
    seed: int = 42,
) -> dict[str, Any]:
    records, report = assign_grouped_splits(read_manifest(input_path), seed=seed)
    write_manifest(output_path, records)
    report["input_manifest_sha256"] = sha256_file(input_path)
    report["output_manifest_sha256"] = sha256_file(output_path)
    write_json(report_path, report)
    return report

