#!/usr/bin/env python3
"""Fit one temperature on a retained controlled-RINE seed's clean logits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cya_detector.data.manifest import sha256_file, write_json  # noqa: E402
from cya_detector.evaluation.calibration import fit_temperature  # noqa: E402
from cya_detector.predictions import read_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_predictions(args.predictions)
    clean_rows = [
        row for row in rows if row.evaluation_cell == "clean" and row.seed == args.seed
    ]
    result = fit_temperature(clean_rows)
    result["source_predictions_path"] = str(args.predictions)
    result["source_predictions_sha256"] = sha256_file(args.predictions)
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
