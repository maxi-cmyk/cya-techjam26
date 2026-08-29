#!/usr/bin/env python3
"""Compare RGB, Lab, and combined clean auxiliary representations."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cya_detector.evaluation.metrics import evaluate_predictions  # noqa: E402
from cya_detector.predictions import read_predictions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task8-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    variants = ("rgb", "lab", "rgb_lab")
    by_variant: dict[str, list[float]] = {variant: [] for variant in variants}
    per_seed: dict[str, dict[str, float]] = {}
    for seed in args.seeds:
        seed_values: dict[str, float] = {}
        for variant in variants:
            records = read_predictions(
                args.task8_root / variant / f"seed_{seed}" / "selection_predictions.csv"
            )
            clean = evaluate_predictions(records)["clean"]
            if clean is None:
                raise SystemExit("Color comparison requires clean predictions")
            by_variant[variant].append(clean["accuracy"])
            seed_values[variant] = clean["accuracy"]
        per_seed[str(seed)] = seed_values
    means = {variant: statistics.fmean(values) for variant, values in by_variant.items()}
    selected = max(variants, key=lambda variant: (means[variant], -variants.index(variant)))
    report = {
        "seeds": args.seeds,
        "selection_split": "selection_val",
        "by_seed": per_seed,
        "accuracy_mean": means,
        "selected_color_representation": selected,
        "physical_family_retention": "pending_eligible_data_and_task3",
        "final_retention": "pending_task3_and_rine_fusion",
        "final_test_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
