#!/usr/bin/env python3
"""Create the locked Task 5 report from prediction records."""

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
from cya_detector.evaluation.reporting import build_report, write_report  # noqa: E402
from cya_detector.predictions import read_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/colab.json"))
    parser.add_argument("--bootstrap-iterations", type=int)
    parser.add_argument("--final-evaluation", action="store_true")
    parser.add_argument("--architecture-frozen", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    evaluation = config["evaluation"]
    records = read_predictions(args.predictions)
    report = build_report(
        records,
        threshold=evaluation["threshold"],
        bootstrap_iterations=(
            args.bootstrap_iterations
            if args.bootstrap_iterations is not None
            else evaluation["bootstrap_iterations"]
        ),
        bootstrap_seed=config["runtime"]["seed"],
        final_evaluation=args.final_evaluation,
        architecture_frozen=args.architecture_frozen,
    )
    write_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Evaluation artifacts: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
