#!/usr/bin/env python3
"""Extract and cache deterministic Stage 1 frequency features."""

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
from cya_detector.training.frequency_stage1 import extract_frequency_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--matching-policy", default="fixed_q96")
    parser.add_argument("--config", type=Path, default=Path("configs/colab.json"))
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    report = extract_frequency_manifest(
        manifest_path=args.manifest,
        output_path=args.output,
        report_path=args.report,
        cache_root=args.cache_root,
        matching_policy=args.matching_policy,
        configuration=config["frequency"],
        workers=args.workers,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
