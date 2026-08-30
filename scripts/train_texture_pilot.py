#!/usr/bin/env python3
"""Train one Task 9 texture head from already-frozen feature caches."""

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
from cya_detector.data.dataset import ManifestExample  # noqa: E402
from cya_detector.training.texture_stage_d import (  # noqa: E402
    CachedTextureFeatures, LOCKED_TEXTURE_SEEDS, LOCKED_TEXTURE_VARIANTS, train_texture_head,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached-features", type=Path, required=True, help="JSON rows produced from Task 4 cache outputs")
    parser.add_argument("--variant", required=True, choices=LOCKED_TEXTURE_VARIANTS)
    parser.add_argument("--seed", type=int, required=True, choices=LOCKED_TEXTURE_SEEDS)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/task9"))
    parser.add_argument("--config", type=Path, default=Path("configs/colab.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--physical-batch-size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _cached_payload(path: Path) -> tuple[list[CachedTextureFeatures], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list) or not isinstance(payload.get("task4_extraction_report"), dict):
        raise ValueError("--cached-features must be a JSON object with rows and task4_extraction_report")
    rows = [
        CachedTextureFeatures(
            example=ManifestExample(**row["example"]),
            global_cache_path=Path(row["global_cache_path"]), patch_cache_path=Path(row["patch_cache_path"]),
            matching_policy=row.get("matching_policy", ""),
        )
        for row in payload["rows"]
    ]
    return rows, payload["task4_extraction_report"]


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    experiment_root = args.output_root / config["texture"]["experiment_name"]
    optimization = config["optimization"]
    rows, extraction_report = _cached_payload(args.cached_features)
    summary = train_texture_head(
        rows=rows, variant=args.variant, seed=args.seed,
        output_root=experiment_root, overwrite=args.overwrite, run_configuration=config,
        device=args.device, learning_rate=optimization["head_learning_rate"], weight_decay=optimization["weight_decay"],
        warmup_fraction=optimization["warmup_fraction"], max_epochs=optimization["max_head_epochs"],
        early_stopping_patience=optimization["early_stopping_patience"], physical_batch_size=args.physical_batch_size,
        effective_batch_size=optimization["effective_batch_size"], threshold=config["evaluation"]["threshold"],
        gradient_clip_norm=optimization["gradient_clip_norm"],
        task4_extraction_report=extraction_report,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
