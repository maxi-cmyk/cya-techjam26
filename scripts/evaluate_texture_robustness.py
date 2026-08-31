#!/usr/bin/env python3
"""Evaluate frozen Task 9 clean heads over the locked Stage-1 transform matrix."""

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
from cya_detector.evaluation.texture_robustness import evaluate_texture_stage1  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract transformed features once and emit 81 frozen prediction slices."
    )
    parser.add_argument("--transformed-manifest", type=Path, required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--clean-experiment-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/colab.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device is None:
        try:
            import torch
        except ImportError as exc:
            raise SystemExit("torch is unavailable; run this script in Colab") from exc
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    summary = evaluate_texture_stage1(
        transformed_manifest=args.transformed_manifest,
        materialization_report=args.materialization_report,
        clean_experiment_root=args.clean_experiment_root,
        cache_root=args.cache_root,
        output_root=args.output_root,
        config=load_config(args.config),
        device=device,
        overwrite=args.overwrite,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
