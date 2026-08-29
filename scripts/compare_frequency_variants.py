#!/usr/bin/env python3
"""Compare magnitude-only and bounded-phase frequency representations."""

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

from cya_detector.config import load_config  # noqa: E402
from cya_detector.evaluation.bootstrap import paired_bootstrap_difference  # noqa: E402
from cya_detector.evaluation.metrics import evaluate_predictions  # noqa: E402
from cya_detector.predictions import read_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task7-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/colab.json"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--bootstrap-iterations", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    tolerance = config["evaluation"]["max_per_class_accuracy_regression"]
    iterations = args.bootstrap_iterations or config["evaluation"]["bootstrap_iterations"]
    variants = ("magnitude", "magnitude_phase")
    values = {variant: [] for variant in variants}
    by_seed: dict[str, object] = {}
    authentic_deltas: list[float] = []
    ai_deltas: list[float] = []
    for seed in args.seeds:
        records = {
            variant: read_predictions(
                args.task7_root / variant / f"seed_{seed}" / "selection_predictions.csv"
            )
            for variant in variants
        }
        metrics = {variant: evaluate_predictions(rows) for variant, rows in records.items()}
        magnitude = metrics["magnitude"]["clean"]
        phase = metrics["magnitude_phase"]["clean"]
        if magnitude is None or phase is None:
            raise SystemExit("Frequency comparison requires clean predictions")
        for variant, clean in (("magnitude", magnitude), ("magnitude_phase", phase)):
            values[variant].append(clean["accuracy"])
        authentic_delta = phase["authentic_accuracy"] - magnitude["authentic_accuracy"]
        ai_delta = phase["ai_generated_accuracy"] - magnitude["ai_generated_accuracy"]
        authentic_deltas.append(authentic_delta)
        ai_deltas.append(ai_delta)
        by_seed[str(seed)] = {
            "magnitude_accuracy": magnitude["accuracy"],
            "magnitude_phase_accuracy": phase["accuracy"],
            "accuracy_delta": phase["accuracy"] - magnitude["accuracy"],
            "authentic_accuracy_delta": authentic_delta,
            "ai_generated_accuracy_delta": ai_delta,
            "paired_bootstrap_phase_minus_magnitude": paired_bootstrap_difference(
                records["magnitude_phase"],
                records["magnitude"],
                metric="clean_accuracy",
                iterations=iterations,
                seed=seed,
            ),
        }
    magnitude_mean = statistics.fmean(values["magnitude"])
    phase_mean = statistics.fmean(values["magnitude_phase"])
    authentic_delta_mean = statistics.fmean(authentic_deltas)
    ai_delta_mean = statistics.fmean(ai_deltas)
    keep_phase = (
        phase_mean > magnitude_mean
        and authentic_delta_mean >= -tolerance
        and ai_delta_mean >= -tolerance
    )
    report = {
        "seeds": args.seeds,
        "selection_split": "selection_val",
        "by_seed": by_seed,
        "aggregate": {
            "magnitude_accuracy_mean": magnitude_mean,
            "magnitude_phase_accuracy_mean": phase_mean,
            "phase_accuracy_mean_delta": phase_mean - magnitude_mean,
            "phase_authentic_accuracy_mean_delta": authentic_delta_mean,
            "phase_ai_generated_accuracy_mean_delta": ai_delta_mean,
        },
        "selected_representation": "magnitude_phase" if keep_phase else "magnitude",
        "stage1_early_exit_enabled": False,
        "final_retention": "pending_task3_and_rine_fusion",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
