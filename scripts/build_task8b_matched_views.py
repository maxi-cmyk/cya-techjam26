#!/usr/bin/env python3
"""Build deterministic 256 px lossless matched crops for Task 8B."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cya_detector.data.manifest import DatasetContractError  # noqa: E402
from cya_detector.data.task8b_matched import build_task8b_matched_views  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--perceptual-distance",
        type=int,
        default=1,
        help="Matched crops use a tighter dHash radius to avoid flat-region chaining.",
    )
    parser.add_argument("--minimum-rgb-std", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        report = build_task8b_matched_views(
            source_manifest=args.source_manifest,
            output_root=args.output_root,
            output_manifest=args.output_manifest,
            report_path=args.report,
            size=args.size,
            seed=args.seed,
            perceptual_distance=args.perceptual_distance,
            minimum_rgb_std=args.minimum_rgb_std,
            overwrite=args.overwrite,
        )
    except DatasetContractError as exc:
        print(f"TASK 8B MATCHED-VIEW ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
