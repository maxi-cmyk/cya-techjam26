#!/usr/bin/env python3
"""Materialize the locked Task 9 texture robustness Stage-1 matrix."""

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
from cya_detector.evaluation.texture_robustness import (  # noqa: E402
    materialize_texture_stage1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/colab.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = materialize_texture_stage1(
        input_manifest=args.input_manifest,
        output_root=args.output_root,
        output_manifest=args.output_manifest,
        report_path=args.report,
        config=load_config(args.config),
        overwrite=args.overwrite,
    )
    print(f"Materialized {report['image_count']} images")
    print(f"Cell counts: {json.dumps(report['cell_counts'], sort_keys=True)}")
    print(f"Manifest: {args.output_manifest.resolve()}")
    print(f"Report: {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
