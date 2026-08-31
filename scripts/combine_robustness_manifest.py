#!/usr/bin/env python3
"""Validate and combine clean/Task 3 development rows for feature extraction."""

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

from cya_detector.config import load_config  # noqa: E402
from cya_detector.data.dataset import load_examples  # noqa: E402
from cya_detector.data.manifest import (  # noqa: E402
    read_manifest,
    sha256_file,
    write_json,
    write_manifest,
)
from cya_detector.training.robustness import validate_robustness_bank  # noqa: E402
from cya_detector.transforms.benchmark import benchmark_cells  # noqa: E402


SPLITS = ("seed_train", "selection_val")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--transform-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/colab.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    cells = benchmark_cells(config)
    ordered_sample_ids: list[str] = []
    for split in SPLITS:
        bank = validate_robustness_bank(
            load_examples(args.clean_manifest, splits={split}),
            load_examples(args.transform_manifest, splits={split}),
            cells,
            split=split,
        )
        ordered_sample_ids.extend(example.sample_id for example in bank.all_examples)

    rows_by_id = {
        row["sample_id"]: row
        for row in (*read_manifest(args.clean_manifest), *read_manifest(args.transform_manifest))
    }
    if len(rows_by_id) != len(ordered_sample_ids):
        raise ValueError("Clean and transform manifests contain duplicate or unexpected rows")
    output_rows = [rows_by_id[sample_id] for sample_id in ordered_sample_ids]
    write_manifest(args.output_manifest, output_rows)
    counts = Counter((row["split"], row["label"], row["image_view"]) for row in output_rows)
    report = {
        "clean_manifest_sha256": sha256_file(args.clean_manifest),
        "transform_manifest_sha256": sha256_file(args.transform_manifest),
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256_file(args.output_manifest),
        "row_count": len(output_rows),
        "final_test_rows": 0,
        "cell_count": len(cells),
        "counts": {
            f"{split}:{label}:{view}": count
            for (split, label, view), count in sorted(counts.items())
        },
    }
    write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
