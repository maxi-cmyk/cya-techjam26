#!/usr/bin/env python3
"""Generate one label-independent canonical matched-clean candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cya_detector.data.matched import (  # noqa: E402
    SUPPORTED_POLICIES,
    MatchedViewError,
    build_matched_clean,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--policy", choices=sorted(SUPPORTED_POLICIES), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-label", type=int)
    parser.add_argument(
        "--allow-unchecked-c2pa",
        action="store_true",
        help="Fixture/debug use only; do not use for the canonical dataset.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_matched_clean(
            source_manifest=args.source_manifest,
            output_root=args.output_root,
            output_manifest=args.output_manifest,
            report_path=args.report,
            policy=args.policy,
            seed=args.seed,
            limit_per_label=args.limit_per_label,
            allow_unchecked_c2pa=args.allow_unchecked_c2pa,
        )
    except MatchedViewError as exc:
        print(f"MATCHED VIEW ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

