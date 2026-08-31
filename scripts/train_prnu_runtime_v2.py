#!/usr/bin/env python3
"""Train the reference-free PRNU-v2-only locked robustness diagnostic."""

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
from cya_detector.training.prnu_runtime_v2 import (  # noqa: E402
    load_prnu_runtime_rows,
    train_prnu_runtime_baseline,
)
from cya_detector.training.robustness import validate_robustness_bank  # noqa: E402
from cya_detector.transforms.benchmark import benchmark_cells  # noqa: E402


def _examples(clean: Path, transformed: Path, *, split: str, config: dict[str, object]):
    return validate_robustness_bank(
        load_examples(clean, splits={split}),
        load_examples(transformed, splits={split}),
        benchmark_cells(config),
        split=split,
    ).all_examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--transform-manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/robustness"))
    parser.add_argument("--matching-policy", default="fixed_q96")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch-size", type=int)
    parser.add_argument("--config", type=Path, default=Path("configs/colab.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    train_examples = _examples(
        args.clean_manifest, args.transform_manifest, split="seed_train", config=config
    )
    selection_examples = _examples(
        args.clean_manifest, args.transform_manifest, split="selection_val", config=config
    )
    train_rows = load_prnu_runtime_rows(table_path=args.features, examples=train_examples)
    selection_rows = load_prnu_runtime_rows(
        table_path=args.features, examples=selection_examples
    )
    clean_ids = {example.sample_id for example in train_examples if example.transform == "clean"}
    train_parents = [row for row in train_rows if row.example.sample_id in clean_ids]
    output = args.output_root / "prnu_v2_runtime" / f"seed_{args.seed}"
    summary = train_prnu_runtime_baseline(
        train_parent_rows=train_parents,
        train_bank_rows=train_rows,
        selection_rows=selection_rows,
        cells=benchmark_cells(config),
        output_directory=output,
        matching_policy=args.matching_policy,
        seed=args.seed,
        threshold=config["evaluation"]["threshold"],
        sampling_epochs=config["prnu_v2_runtime"]["sampling_epochs"],
        epoch_size=args.epoch_size,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"PRNU-v2 runtime diagnostic artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
