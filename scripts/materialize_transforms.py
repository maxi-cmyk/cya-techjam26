#!/usr/bin/env python3
"""Materialize configured independent benchmark transforms from clean parents."""

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
from cya_detector.transforms.benchmark import benchmark_cells  # noqa: E402
from cya_detector.transforms.materialize import materialize_benchmarks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--cells",
        help="Comma-separated benchmark cell IDs; defaults to every configured cell.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    declared_cells = benchmark_cells(config)
    cells_by_id = {cell.cell_id: cell for cell in declared_cells}
    requested_ids = (
        [cell_id.strip() for cell_id in args.cells.split(",")]
        if args.cells is not None
        else [cell.cell_id for cell in declared_cells]
    )
    unknown_ids = sorted({cell_id for cell_id in requested_ids if cell_id not in cells_by_id})
    if unknown_ids:
        print(f"Unknown benchmark cell IDs: {', '.join(unknown_ids)}", file=sys.stderr)
        return 2

    report = materialize_benchmarks(
        input_manifest=args.input_manifest,
        output_root=args.output_root,
        output_manifest=args.output_manifest,
        report_path=args.report,
        config=config,
        cells=tuple(cells_by_id[cell_id] for cell_id in requested_ids),
        overwrite=args.overwrite,
    )
    print(f"Materialized {report['image_count']} images")
    print(f"Cell counts: {json.dumps(report['cell_counts'], sort_keys=True)}")
    print(f"Manifest: {args.output_manifest.resolve()}")
    print(f"Report: {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
