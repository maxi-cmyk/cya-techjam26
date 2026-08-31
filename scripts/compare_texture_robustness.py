#!/usr/bin/env python3
"""Verify controlled RINE and gate the locked Task 9 Stage-1 texture matrix."""

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
from cya_detector.evaluation.texture_robustness import compare_texture_stage1  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-experiment-root", type=Path, required=True)
    parser.add_argument("--robustness-root", type=Path, required=True)
    parser.add_argument("--controlled-rine-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/colab.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = compare_texture_stage1(
        clean_experiment_root=args.clean_experiment_root,
        robustness_root=args.robustness_root,
        controlled_rine_root=args.controlled_rine_root,
        config=load_config(args.config),
        output_root=args.output_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
