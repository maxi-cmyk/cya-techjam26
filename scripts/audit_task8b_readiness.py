#!/usr/bin/env python3
"""Audit Task 8B source, split, PRNU, CA, and nuisance readiness gates."""

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
from cya_detector.data.task8b_audit import (  # noqa: E402
    Task8BReadinessError,
    audit_task8b_readiness,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/colab.json"))
    parser.add_argument("--require-source-ready", action="store_true")
    parser.add_argument("--require-training-ready", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        report = audit_task8b_readiness(
            manifest_path=args.manifest,
            output_path=args.output,
            readiness=config["task8b"]["readiness"],
            minimum_images_per_device=config["task8b"]["minimum_images_per_device"],
        )
    except Task8BReadinessError as exc:
        print(f"TASK 8B READINESS ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_source_ready and not report["source_ready"]:
        return 3
    if args.require_training_ready and not report["training_ready"]:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
