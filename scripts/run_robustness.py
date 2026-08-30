#!/usr/bin/env python3
"""Evaluate existing CLIP/RINE heads or retrain RINE on Task 3 robustness views."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cya_detector.config import load_config  # noqa: E402
from cya_detector.data.dataset import ManifestExample, load_examples  # noqa: E402
from cya_detector.data.manifest import sha256_file  # noqa: E402
from cya_detector.evaluation.reporting import build_report, write_report  # noqa: E402
from cya_detector.models.clip_baseline import load_frozen_clip  # noqa: E402
from cya_detector.predictions import write_predictions  # noqa: E402
from cya_detector.reproducibility import (  # noqa: E402
    collect_run_metadata,
    write_run_metadata,
)
from cya_detector.training.clip_stage_a import (  # noqa: E402
    CachedEmbedding,
    extract_embeddings,
    predict_linear_probe_checkpoint,
)
from cya_detector.training.rine_stage_b import (  # noqa: E402
    extract_rine_features,
    predict_rine_checkpoint,
    train_controlled_rine_head,
)
from cya_detector.training.robustness import validate_robustness_bank  # noqa: E402
from cya_detector.transforms.benchmark import benchmark_cells  # noqa: E402


MODES = ("evaluate-stage-a", "evaluate-rine", "train-controlled-rine")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--transform-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/colab.json"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/robustness"))
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--matching-policy", default="fixed_q96")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--physical-batch-size", type=int, default=4)
    parser.add_argument("--epoch-size", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.mode != "train-controlled-rine" and args.checkpoint is None:
        parser.error(f"--checkpoint is required for {args.mode}")
    if args.mode == "train-controlled-rine" and args.checkpoint is not None:
        parser.error("--checkpoint is not used when training controlled RINE")
    return args


def _combined_manifest_hash(clean_manifest: Path, transform_manifest: Path) -> str:
    digest = hashlib.sha256()
    for path in (clean_manifest, transform_manifest):
        digest.update(str(path.resolve()).encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def _load_bank(
    *,
    clean_manifest: Path,
    transform_manifest: Path,
    split: str,
    config: dict[str, Any],
) -> tuple[ManifestExample, ...]:
    clean = load_examples(clean_manifest, splits={split})
    variants = load_examples(transform_manifest, splits={split})
    bank = validate_robustness_bank(
        clean,
        variants,
        benchmark_cells(config),
        split=split,
    )
    return bank.all_examples


def _rows_for_ids(
    cached: list[CachedEmbedding],
    examples: tuple[ManifestExample, ...],
) -> list[CachedEmbedding]:
    sample_ids = {example.sample_id for example in examples}
    rows = [row for row in cached if row.example.sample_id in sample_ids]
    if len(rows) != len(examples):
        raise ValueError("Cached robustness feature bank does not match its manifest examples")
    return rows


def _write_completion_marker(
    path: Path,
    *,
    mode: str,
    seed: int,
    clean_manifest: Path,
    transform_manifest: Path,
    checkpoint: Path | None,
) -> None:
    value = {
        "complete": True,
        "mode": mode,
        "seed": seed,
        "clean_manifest_sha256": sha256_file(clean_manifest),
        "transform_manifest_sha256": sha256_file(transform_manifest),
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
    }
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    config["runtime"]["seed"] = args.seed
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("torch is unavailable; run this script in the prepared Colab runtime") from exc

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_config = config["model"]
    cells = benchmark_cells(config)
    selection_examples = _load_bank(
        clean_manifest=args.clean_manifest,
        transform_manifest=args.transform_manifest,
        split="selection_val",
        config=config,
    )
    train_examples: tuple[ManifestExample, ...] = ()
    train_parents: tuple[ManifestExample, ...] = ()
    if args.mode == "train-controlled-rine":
        train_examples = _load_bank(
            clean_manifest=args.clean_manifest,
            transform_manifest=args.transform_manifest,
            split="seed_train",
            config=config,
        )
        train_parents = tuple(
            example for example in train_examples if example.transform == "clean"
        )

    loaded = load_frozen_clip(
        model_config["identifier"],
        revision=model_config["revision"],
        device=device,
    )
    run_directory = args.output_root / args.mode.replace("evaluate-", "existing-") / f"seed_{args.seed}"
    run_directory.mkdir(parents=True, exist_ok=True)
    cache_root = args.cache_root or Path(
        "/content/robustness_clip_cache"
        if args.mode == "evaluate-stage-a"
        else "/content/robustness_rine_cache"
    )
    all_examples = list(train_examples + selection_examples)

    if args.mode == "evaluate-stage-a":
        cached, extraction_report = extract_embeddings(
            loaded_clip=loaded,
            examples=all_examples,
            cache_root=cache_root,
            matching_policy=args.matching_policy,
            preprocessing_version=config["preprocessing"]["version"],
            batch_size=args.physical_batch_size,
            device=device,
        )
    else:
        cached, extraction_report = extract_rine_features(
            loaded_clip=loaded,
            examples=all_examples,
            cache_root=cache_root,
            matching_policy=args.matching_policy,
            preprocessing_version=config["preprocessing"]["version"],
            representation_version=model_config["rine_representation_version"],
            layers=model_config["rine_layers"],
            batch_size=args.physical_batch_size,
            device=device,
        )
    (run_directory / "extraction_report.json").write_text(
        json.dumps(extraction_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metadata = collect_run_metadata(
        config=config,
        repo_root=REPO_ROOT,
        distributions=("torch", "transformers", "Pillow", "numpy"),
    )
    metadata["model_revision"] = {
        "requested": loaded.requested_revision,
        "resolved": loaded.resolved_revision,
    }
    metadata["manifests"] = {
        "clean": {
            "path": str(args.clean_manifest.resolve()),
            "sha256": sha256_file(args.clean_manifest),
        },
        "transforms": {
            "path": str(args.transform_manifest.resolve()),
            "sha256": sha256_file(args.transform_manifest),
        },
    }
    write_run_metadata(run_directory / "run_metadata.json", metadata)

    selection_rows = _rows_for_ids(cached, selection_examples)
    if args.mode == "evaluate-stage-a":
        predictions, checkpoint_metadata = predict_linear_probe_checkpoint(
            checkpoint_path=args.checkpoint,
            rows=selection_rows,
            seed=args.seed,
            matching_policy=args.matching_policy,
            device=device,
        )
    elif args.mode == "evaluate-rine":
        predictions, checkpoint_metadata = predict_rine_checkpoint(
            checkpoint_path=args.checkpoint,
            rows=selection_rows,
            seed=args.seed,
            matching_policy=args.matching_policy,
            device=device,
        )
    else:
        train_rows = _rows_for_ids(cached, train_examples)
        train_parent_rows = _rows_for_ids(cached, train_parents)
        summary = train_controlled_rine_head(
            train_parent_rows=train_parent_rows,
            train_bank_rows=train_rows,
            selection_rows=selection_rows,
            cells=cells,
            output_directory=run_directory,
            matching_policy=args.matching_policy,
            layers=model_config["rine_layers"],
            resolved_revision=loaded.resolved_revision,
            manifest_sha256=_combined_manifest_hash(
                args.clean_manifest,
                args.transform_manifest,
            ),
            seed=args.seed,
            device=device,
            learning_rate=config["optimization"]["head_learning_rate"],
            weight_decay=config["optimization"]["weight_decay"],
            warmup_fraction=config["optimization"]["warmup_fraction"],
            max_epochs=config["optimization"]["max_head_epochs"],
            early_stopping_patience=config["optimization"]["early_stopping_patience"],
            physical_batch_size=args.physical_batch_size,
            effective_batch_size=config["optimization"]["effective_batch_size"],
            threshold=config["evaluation"]["threshold"],
            run_configuration=config,
            epoch_size=args.epoch_size,
        )
        _write_completion_marker(
            run_directory / "complete.json",
            mode=args.mode,
            seed=args.seed,
            clean_manifest=args.clean_manifest,
            transform_manifest=args.transform_manifest,
            checkpoint=None,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"Controlled RINE artifacts: {run_directory}")
        return 0

    write_predictions(run_directory / "predictions.csv", predictions)
    report = build_report(
        predictions,
        threshold=config["evaluation"]["threshold"],
        bootstrap_iterations=config["evaluation"]["bootstrap_iterations"],
        bootstrap_seed=args.seed,
    )
    report["checkpoint_metadata"] = checkpoint_metadata
    write_report(run_directory, report)
    _write_completion_marker(
        run_directory / "complete.json",
        mode=args.mode,
        seed=args.seed,
        clean_manifest=args.clean_manifest,
        transform_manifest=args.transform_manifest,
        checkpoint=args.checkpoint,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Robustness artifacts: {run_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
