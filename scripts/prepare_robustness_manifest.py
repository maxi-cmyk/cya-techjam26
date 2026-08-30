#!/usr/bin/env python3
"""Create the non-final matched-clean parent manifest for robustness work."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cya_detector.data.manifest import (  # noqa: E402
    read_manifest,
    sha256_file,
    write_json,
    write_manifest,
)
from cya_detector.transforms.benchmark import validate_parent_record  # noqa: E402


ALLOWED_SPLITS = frozenset({"seed_train", "selection_val"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def prepare_manifest(
    *,
    input_manifest: Path,
    output_manifest: Path,
    report_path: Path,
) -> dict[str, object]:
    rows = [row for row in read_manifest(input_manifest) if row.get("split") in ALLOWED_SPLITS]
    if not rows:
        raise ValueError("No seed_train or selection_val matched-clean rows found")
    for row in rows:
        validate_parent_record(row)
        if not row.get("source_id") or not row.get("sample_id"):
            raise ValueError("Robustness parents require source_id and sample_id")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Robustness parent sample_id values must be unique")
    source_ids = [row["source_id"] for row in rows]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Each robustness source_id must have exactly one matched-clean parent")

    write_manifest(output_manifest, sorted(rows, key=lambda row: row["sample_id"]))
    counts = Counter((row["split"], row["label"]) for row in rows)
    report: dict[str, object] = {
        "input_manifest": str(input_manifest.resolve()),
        "input_manifest_sha256": sha256_file(input_manifest),
        "output_manifest": str(output_manifest.resolve()),
        "output_manifest_sha256": sha256_file(output_manifest),
        "allowed_splits": sorted(ALLOWED_SPLITS),
        "final_test_rows": 0,
        "row_count": len(rows),
        "counts": {f"{split}:{label}": count for (split, label), count in sorted(counts.items())},
    }
    write_json(report_path, report)
    return report


def main() -> int:
    args = parse_args()
    report = prepare_manifest(
        input_manifest=args.input_manifest,
        output_manifest=args.output_manifest,
        report_path=args.report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
