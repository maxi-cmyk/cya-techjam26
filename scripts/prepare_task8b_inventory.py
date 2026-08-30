#!/usr/bin/env python3
"""Discover extracted licensed Task 8B files and create a reviewable inventory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cya_detector.config import load_config  # noqa: E402
from cya_detector.data.task8b_prepare import (  # noqa: E402
    Task8BPreparationError,
    prepare_task8b_inventory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/colab.json"))
    parser.add_argument("--task8b-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--max-per-generator", type=int, default=830)
    parser.add_argument(
        "--generators",
        nargs="+",
        help="Optional allow-list; downloaded generators outside it remain stress-only.",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    task8b = config["task8b"]
    paths = config["paths"]
    task8b_root = args.task8b_root or (
        Path(paths["local_data_root"]) / task8b["source_relative_path"]
    )
    output = args.output or task8b_root / task8b["inventory_filename"]
    artifact_root = Path(paths["local_artifact_root"]) / task8b["artifact_relative_path"]
    report = args.report or artifact_root / "audits/inventory_preparation.json"
    seed = args.seed if args.seed is not None else int(config["runtime"]["seed"])

    try:
        result = prepare_task8b_inventory(
            task8b_root=task8b_root,
            output_path=output,
            report_path=report,
            supported_extensions=set(task8b["supported_source_extensions"]),
            max_per_generator=args.max_per_generator,
            seed=seed,
            allowed_generators=set(args.generators) if args.generators else None,
            overwrite=args.overwrite,
        )
    except Task8BPreparationError as exc:
        print(f"TASK 8B PREPARATION ERROR: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result["selected"], indent=2))
    print(f"Inventory requiring review: {output}")
    print(f"Preparation report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
