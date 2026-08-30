#!/usr/bin/env python3
"""Extract frozen CLIP embeddings and train the Stage A binary head."""

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
from cya_detector.data.manifest import sha256_file  # noqa: E402
from cya_detector.models.clip_baseline import load_frozen_clip  # noqa: E402
from cya_detector.reproducibility import (  # noqa: E402
    collect_run_metadata,
    write_run_metadata,
)
from cya_detector.training.clip_stage_a import (  # noqa: E402
    extract_embeddings,
    train_linear_probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--matching-policy", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/colab.json"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/task4"))
    parser.add_argument("--cache-root", type=Path, default=Path("/content/clip_embedding_cache"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--physical-batch-size", type=int, default=8)
    parser.add_argument("--device")
    parser.add_argument("--hidden-dimension", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("torch is unavailable; run this script in the prepared Colab runtime") from exc

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_config = config["model"]
    loaded = load_frozen_clip(
        model_config["identifier"],
        revision=model_config["revision"],
        device=device,
    )
    train_examples = load_examples(args.manifest, splits={"seed_train"})
    selection_examples = load_examples(args.manifest, splits={"selection_val"})
    all_examples = train_examples + selection_examples
    cached, extraction_report = extract_embeddings(
        loaded_clip=loaded,
        examples=all_examples,
        cache_root=args.cache_root,
        matching_policy=args.matching_policy,
        preprocessing_version=config["preprocessing"]["version"],
        batch_size=args.physical_batch_size,
        device=device,
    )
    train_ids = {example.sample_id for example in train_examples}
    selection_ids = {example.sample_id for example in selection_examples}
    train_rows = [row for row in cached if row.example.sample_id in train_ids]
    selection_rows = [row for row in cached if row.example.sample_id in selection_ids]

    run_directory = args.output_root / args.matching_policy / f"seed_{args.seed}"
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "extraction_report.json").write_text(
        json.dumps(extraction_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config["runtime"]["seed"] = args.seed
    metadata = collect_run_metadata(
        config=config,
        repo_root=REPO_ROOT,
        distributions=("torch", "transformers", "Pillow", "numpy"),
    )
    metadata["model_revision"] = {
        "requested": loaded.requested_revision,
        "resolved": loaded.resolved_revision,
    }
    metadata["manifest"] = {
        "path": str(args.manifest.resolve()),
        "sha256": sha256_file(args.manifest),
    }
    write_run_metadata(run_directory / "run_metadata.json", metadata)

    summary = train_linear_probe(
        train_rows=train_rows,
        selection_rows=selection_rows,
        output_directory=run_directory,
        matching_policy=args.matching_policy,
        resolved_revision=loaded.resolved_revision,
        manifest_sha256=sha256_file(args.manifest),
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
        hidden_dimension=args.hidden_dimension,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Stage A artifacts: {run_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
