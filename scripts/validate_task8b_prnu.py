#!/usr/bin/env python3
"""Run label-free, seed-train-only Task 8B PRNU device validation."""

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
from cya_detector.features.prnu_reference import validate_prnu_device_signal  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/colab.json"))
    parser.add_argument("--reference-images-per-device", type=int, default=10)
    parser.add_argument("--reference-size", type=int, default=256)
    parser.add_argument("--minimum-auc", type=float, default=0.60)
    parser.add_argument("--require-signal", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    report = validate_prnu_device_signal(
        manifest_path=args.manifest,
        report_path=args.output,
        minimum_images_per_device=config["task8b"]["minimum_images_per_device"],
        reference_images_per_device=args.reference_images_per_device,
        reference_size=args.reference_size,
        denoise_sigma=config["auxiliary"]["prnu_denoise_sigma"],
        seed=config["runtime"]["seed"],
        minimum_auc=args.minimum_auc,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["signal_validated"] or not args.require_signal else 3


if __name__ == "__main__":
    raise SystemExit(main())
