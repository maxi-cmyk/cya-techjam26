#!/usr/bin/env python3
"""Build and split the licensed Task 8B native/synthetic source manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cya_detector.config import load_config  # noqa: E402
from cya_detector.data.manifest import (  # noqa: E402
    DatasetContractError,
    write_json,
    write_manifest,
)
from cya_detector.data.task8b import (  # noqa: E402
    assign_task8b_splits,
    build_task8b_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/colab.json"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument("--split-report", type=Path)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    task8b = config["task8b"]
    paths = config["paths"]
    dataset_root = args.dataset_root or (
        Path(paths["local_data_root"]) / task8b["source_relative_path"]
    )
    inventory = args.inventory or dataset_root / task8b["inventory_filename"]
    artifact_root = Path(paths["local_artifact_root"]) / task8b["artifact_relative_path"]
    manifest = args.manifest or artifact_root / "manifests/source_manifest_split.csv"
    audit_report = args.audit_report or artifact_root / "audits/source_audit.json"
    split_report_path = args.split_report or artifact_root / "audits/split_report.json"
    seed = args.seed if args.seed is not None else int(config["runtime"]["seed"])

    try:
        records, audit = build_task8b_manifest(
            dataset_root=dataset_root,
            inventory_path=inventory,
            allow_noncommercial_genimage=task8b["allow_noncommercial_genimage"],
            supported_extensions=set(task8b["supported_source_extensions"]),
            minimum_images_per_device=task8b["minimum_images_per_device"],
            perceptual_distance=config["dataset"]["perceptual_hash_max_distance"],
        )
        records, split_report = assign_task8b_splits(
            records,
            seed=seed,
            fractions=task8b["split_fractions"],
        )
    except DatasetContractError as exc:
        print(f"TASK 8B DATASET CONTRACT ERROR: {exc}", file=sys.stderr)
        return 2

    write_manifest(manifest, records)
    write_json(audit_report, audit)
    write_json(split_report_path, split_report)
    print(f"Manifest: {manifest}")
    print(f"Audit: {audit_report}")
    print(f"Split report: {split_report_path}")
    print(f"Eligible rows: {audit['eligible_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
