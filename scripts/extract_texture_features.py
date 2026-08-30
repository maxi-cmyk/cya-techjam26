#!/usr/bin/env python3
"""Produce the versioned Task 4 cache handoff consumed by texture training."""

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
from cya_detector.data.dataset import load_examples  # noqa: E402
from cya_detector.models.clip_baseline import load_frozen_clip  # noqa: E402
from cya_detector.training.texture_stage_d import (  # noqa: E402
    APPROVED_MATCHING_POLICY,
    extract_texture_features,
    write_cached_texture_features_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-payload", type=Path, required=True, help="Task 4 handoff JSON for train_texture_pilot.py")
    parser.add_argument("--global-cache-root", type=Path, required=True)
    parser.add_argument("--patch-cache-root", type=Path, required=True)
    parser.add_argument("--matching-policy", choices=(APPROVED_MATCHING_POLICY,), default=APPROVED_MATCHING_POLICY)
    parser.add_argument("--config", type=Path, default=Path("configs/colab.json"))
    parser.add_argument("--physical-batch-size", type=int, default=4)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("torch is unavailable; run this script in Colab") from exc
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_config = config["model"]
    texture_config = config["texture"]
    examples = load_examples(args.manifest, splits={"seed_train", "selection_val"})
    loaded = load_frozen_clip(
        model_config["identifier"], revision=model_config["revision"], device=device
    )
    rows, report = extract_texture_features(
        loaded_clip=loaded,
        examples=examples,
        global_cache_root=args.global_cache_root,
        patch_cache_root=args.patch_cache_root,
        matching_policy=args.matching_policy,
        preprocessing_version=config["preprocessing"]["version"],
        rine_representation_version=model_config["rine_representation_version"],
        texture_extractor_version=texture_config["extractor_version"],
        layers=tuple(model_config["rine_layers"]),
        patch_size=int(texture_config["patch_size"]),
        patch_count=int(texture_config["patch_count"]),
        batch_size=args.physical_batch_size,
        device=device,
    )
    write_cached_texture_features_payload(
        args.cache_payload, rows=rows, task4_extraction_report=report
    )
    print(json.dumps({"cache_payload": str(args.cache_payload), "example_count": len(rows)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
