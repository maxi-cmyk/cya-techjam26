#!/usr/bin/env python3
"""Run the bounded, label-free Task 8B PRNU v2 experiment."""

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
from cya_detector.features.prnu_reference_v2 import (  # noqa: E402
    validate_prnu_device_signal_v2,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/colab.json"))
    parser.add_argument("--reference-images-per-device", type=int, default=25)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--wavelet", default="db2")
    parser.add_argument("--wavelet-levels", type=int, default=4)
    parser.add_argument("--edge-keep-quantile", type=float, default=0.75)
    parser.add_argument("--maximum-shift", type=int, default=8)
    parser.add_argument("--minimum-auc", type=float, default=0.60)
    parser.add_argument("--require-signal", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    report = validate_prnu_device_signal_v2(
        manifest_path=args.manifest,
        artifact_root=args.artifact_root,
        reference_images_per_device=args.reference_images_per_device,
        crop_size=args.crop_size,
        wavelet=args.wavelet,
        wavelet_levels=args.wavelet_levels,
        edge_keep_quantile=args.edge_keep_quantile,
        maximum_shift=args.maximum_shift,
        seed=config["runtime"]["seed"],
        minimum_auc=args.minimum_auc,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["signal_validated"] or not args.require_signal else 3


if __name__ == "__main__":
    raise SystemExit(main())
