#!/usr/bin/env python3
"""Train one clean feature-only Stage C auxiliary baseline."""

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
from cya_detector.training.auxiliary_stage_c import (  # noqa: E402
    VARIANT_FAMILIES,
    train_auxiliary_baseline,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANT_FAMILIES), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=Path("configs/colab.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    report = train_auxiliary_baseline(
        feature_table=args.features,
        output_directory=args.output,
        variant=args.variant,
        seed=args.seed,
        threshold=config["evaluation"]["threshold"],
        max_eligibility_gap=config["auxiliary"]["max_eligibility_rate_gap"],
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
