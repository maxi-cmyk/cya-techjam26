#!/usr/bin/env python3
"""Extract a bounded deterministic AI-only sample from one GenImage ZIP."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cya_detector.config import load_config  # noqa: E402
from cya_detector.data.task8b_archive import (  # noqa: E402
    Task8BArchiveError,
    extract_genimage_ai_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/colab.json"))
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    task8b = config["task8b"]
    paths = config["paths"]
    task8b_root = Path(paths["local_data_root"]) / task8b["source_relative_path"]
    output_root = args.output_root or task8b_root / "genimage_ai"
    artifact_root = Path(paths["local_artifact_root"]) / task8b["artifact_relative_path"]
    report = args.report or (
        artifact_root / "audits" / f"extract_{args.generator.replace(' ', '_')}.json"
    )
    seed = args.seed if args.seed is not None else int(config["runtime"]["seed"])
    try:
        result = extract_genimage_ai_sample(
            archive_path=args.archive,
            generator_name=args.generator,
            output_root=output_root,
            report_path=report,
            limit=args.limit,
            seed=seed,
            supported_extensions=set(task8b["supported_source_extensions"]),
        )
    except (Task8BArchiveError, zipfile.BadZipFile) as exc:
        print(f"TASK 8B ARCHIVE ERROR: {exc}", file=sys.stderr)
        return 2
    summary_keys = ("generator_name", "selected_count", "destination")
    print(json.dumps({key: result[key] for key in summary_keys}, indent=2))
    print(f"Extraction report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
