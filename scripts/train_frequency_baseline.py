#!/usr/bin/env python3
"""Train one deterministic frequency-only Stage 1 baseline."""

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
from cya_detector.training.frequency_stage1 import train_frequency_baseline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("magnitude", "magnitude_phase"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=Path("configs/colab.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    report = train_frequency_baseline(
        feature_table=args.features,
        output_directory=args.output,
        variant=args.variant,
        seed=args.seed,
        threshold=config["evaluation"]["threshold"],
        early_exit_enabled=config["features"]["frequency_fast_track"],
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
