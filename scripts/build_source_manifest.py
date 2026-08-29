#!/usr/bin/env python3
"""Audit immutable SID sources and create the source-original manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cya_detector.data.manifest import (  # noqa: E402
    DatasetContractError,
    build_source_manifest,
    write_json,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dataset-name", default="sid_set")
    parser.add_argument("--license-status", default="cc-by-4.0")
    parser.add_argument("--perceptual-distance", type=int, default=4)
    parser.add_argument(
        "--skip-c2pa",
        action="store_true",
        help="Fixture/debug use only. Canonical preprocessing requires a completed C2PA scan.",
    )
    parser.add_argument("--expected-csv-rows", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        records, report = build_source_manifest(
            dataset_root=args.dataset_root,
            dataset_name=args.dataset_name,
            license_status=args.license_status,
            check_c2pa=not args.skip_c2pa,
            perceptual_distance=args.perceptual_distance,
        )
    except DatasetContractError as exc:
        print(f"DATASET CONTRACT ERROR: {exc}", file=sys.stderr)
        return 2

    if args.expected_csv_rows is not None and report["csv_rows"] != args.expected_csv_rows:
        print(
            f"EXPECTED {args.expected_csv_rows} CSV ROWS, FOUND {report['csv_rows']}",
            file=sys.stderr,
        )
        return 3

    write_manifest(args.manifest, records)
    write_json(args.report, report)
    print(f"Manifest: {args.manifest}")
    print(f"Report: {args.report}")
    print(f"Eligible primaries: {report['eligible_primary_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

