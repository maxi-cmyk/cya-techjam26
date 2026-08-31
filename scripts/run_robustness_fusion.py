#!/usr/bin/env python3
"""Train one predeclared auxiliary fusion candidate over controlled RINE."""

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
from cya_detector.data.dataset import ManifestExample, load_examples  # noqa: E402
from cya_detector.data.manifest import sha256_file  # noqa: E402
from cya_detector.models.clip_baseline import load_frozen_clip  # noqa: E402
from cya_detector.training.clip_stage_a import CachedEmbedding  # noqa: E402
from cya_detector.training.rine_stage_b import extract_rine_features  # noqa: E402
from cya_detector.training.robustness import validate_robustness_bank  # noqa: E402
from cya_detector.training.robustness_fusion import (  # noqa: E402
    FUSION_VARIANTS,
    load_tabular_feature_bank,
    train_controlled_rine_auxiliary_fusion,
)
from cya_detector.transforms.benchmark import benchmark_cells  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(FUSION_VARIANTS), required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--transform-manifest", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--frequency-table", type=Path)
    parser.add_argument("--auxiliary-table", type=Path)
    parser.add_argument("--prnu-table", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/colab.json"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/robustness"))
    parser.add_argument("--cache-root", type=Path, default=Path("/content/robustness_rine_cache"))
    parser.add_argument("--matching-policy", default="fixed_q96")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--physical-batch-size", type=int, default=4)
    parser.add_argument("--epoch-size", type=int)
    parser.add_argument("--device")
    return parser.parse_args()


def _bank(
    clean_manifest: Path,
    transform_manifest: Path,
    *,
    split: str,
    config: dict[str, object],
) -> tuple[ManifestExample, ...]:
    return validate_robustness_bank(
        load_examples(clean_manifest, splits={split}),
        load_examples(transform_manifest, splits={split}),
        benchmark_cells(config),
        split=split,
    ).all_examples


def _rows(
    cached: list[CachedEmbedding],
    examples: tuple[ManifestExample, ...],
) -> list[CachedEmbedding]:
    ids = {example.sample_id for example in examples}
    selected = [row for row in cached if row.example.sample_id in ids]
    if len(selected) != len(examples):
        raise ValueError("RINE cache does not match the robustness manifest")
    return selected


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    config["runtime"]["seed"] = args.seed
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "torch is unavailable; run this script in the prepared Colab runtime"
        ) from exc
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_examples = _bank(
        args.clean_manifest,
        args.transform_manifest,
        split="seed_train",
        config=config,
    )
    selection_examples = _bank(
        args.clean_manifest,
        args.transform_manifest,
        split="selection_val",
        config=config,
    )
    train_parents = tuple(example for example in train_examples if example.transform == "clean")
    model_config = config["model"]
    loaded = load_frozen_clip(
        model_config["identifier"],
        revision=model_config["revision"],
        device=device,
    )
    cached, extraction_report = extract_rine_features(
        loaded_clip=loaded,
        examples=list(train_examples + selection_examples),
        cache_root=args.cache_root,
        matching_policy=args.matching_policy,
        preprocessing_version=config["preprocessing"]["version"],
        representation_version=model_config["rine_representation_version"],
        layers=model_config["rine_layers"],
        batch_size=args.physical_batch_size,
        device=device,
    )
    feature_bank = load_tabular_feature_bank(
        variant=args.variant,
        frequency_table=args.frequency_table,
        auxiliary_table=args.auxiliary_table,
        prnu_table=args.prnu_table,
    )
    run_directory = args.output_root / f"rine_{args.variant}" / f"seed_{args.seed}"
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "extraction_report.json").write_text(
        json.dumps(extraction_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = train_controlled_rine_auxiliary_fusion(
        train_parent_rows=_rows(cached, train_parents),
        train_bank_rows=_rows(cached, train_examples),
        selection_rows=_rows(cached, selection_examples),
        feature_bank=feature_bank,
        parent_checkpoint=args.parent_checkpoint,
        cells=benchmark_cells(config),
        output_directory=run_directory,
        variant=args.variant,
        matching_policy=args.matching_policy,
        layers=model_config["rine_layers"],
        seed=args.seed,
        device=device,
        learning_rate=config["optimization"]["fusion_learning_rate"],
        weight_decay=config["optimization"]["weight_decay"],
        max_epochs=config["optimization"]["max_head_epochs"],
        early_stopping_patience=config["optimization"]["early_stopping_patience"],
        physical_batch_size=args.physical_batch_size,
        effective_batch_size=config["optimization"]["effective_batch_size"],
        threshold=config["evaluation"]["threshold"],
        epoch_size=args.epoch_size,
    )
    completion = {
        "complete": True,
        "variant": args.variant,
        "seed": args.seed,
        "clean_manifest_sha256": sha256_file(args.clean_manifest),
        "transform_manifest_sha256": sha256_file(args.transform_manifest),
        "parent_checkpoint_sha256": sha256_file(args.parent_checkpoint),
        "frequency_table_sha256": (
            sha256_file(args.frequency_table) if args.frequency_table else None
        ),
        "auxiliary_table_sha256": (
            sha256_file(args.auxiliary_table) if args.auxiliary_table else None
        ),
        "prnu_table_sha256": sha256_file(args.prnu_table) if args.prnu_table else None,
    }
    (run_directory / "complete.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Robustness fusion artifacts: {run_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
