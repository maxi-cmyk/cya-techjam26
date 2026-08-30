#!/usr/bin/env python3
"""Apply the deterministic Task 9 clean texture-pilot gate."""

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
from cya_detector.evaluation.texture_gate import compare_texture_pilot  # noqa: E402
from cya_detector.training.texture_stage_d import LOCKED_TEXTURE_SEEDS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/task9"))
    parser.add_argument("--config", type=Path, default=Path("configs/colab.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    report = compare_texture_pilot(
        experiment_root=args.output_root / config["texture"]["experiment_name"],
        seeds=LOCKED_TEXTURE_SEEDS,
        max_per_class_regression=config["evaluation"]["max_per_class_accuracy_regression"],
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
