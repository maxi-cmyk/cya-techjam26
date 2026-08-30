#!/usr/bin/env python3
"""Build training-only per-device PRNU references for Task 8B."""

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
from cya_detector.features.prnu_reference import (  # noqa: E402
    build_training_prnu_references,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/colab.json"))
    parser.add_argument("--reference-size", type=int, default=256)
    args = parser.parse_args()
    config = load_config(args.config)
    report = build_training_prnu_references(
        manifest_path=args.manifest,
        output_root=args.output_root,
        report_path=args.report,
        minimum_images_per_device=config["task8b"]["minimum_images_per_device"],
        reference_size=args.reference_size,
        denoise_sigma=config["auxiliary"]["prnu_denoise_sigma"],
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
