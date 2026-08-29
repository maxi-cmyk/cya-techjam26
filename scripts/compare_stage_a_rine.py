#!/usr/bin/env python3
"""Compare Stage A final-layer CLIP with Stage B RINE on locked selection data."""

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
    parser.add_argument("--stage-a-root", type=Path, required=True)
    parser.add_argument("--stage-b-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/colab.json"))
    parser.add_argument("--matching-policy", default="fixed_q96")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--bootstrap-iterations", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    tolerance = config["evaluation"]["max_per_class_accuracy_regression"]
    iterations = args.bootstrap_iterations or config["evaluation"]["bootstrap_iterations"]
    by_seed: dict[str, object] = {}
    stage_a_clean: list[float] = []
    stage_b_clean: list[float] = []
    authentic_deltas: list[float] = []
    ai_deltas: list[float] = []

    for seed in args.seeds:
        relative = Path(args.matching_policy) / f"seed_{seed}" / "best_clean_predictions.csv"
        stage_a_records = read_predictions(args.stage_a_root / relative)
        stage_b_records = read_predictions(args.stage_b_root / relative)
        stage_a = evaluate_predictions(stage_a_records)
        stage_b = evaluate_predictions(stage_b_records)
        a_clean = stage_a["clean"]
        b_clean = stage_b["clean"]
        if a_clean is None or b_clean is None:
            raise SystemExit("Stage A/B comparison requires clean predictions")
        authentic_delta = b_clean["authentic_accuracy"] - a_clean["authentic_accuracy"]
        ai_delta = b_clean["ai_generated_accuracy"] - a_clean["ai_generated_accuracy"]
        stage_a_clean.append(a_clean["accuracy"])
        stage_b_clean.append(b_clean["accuracy"])
        authentic_deltas.append(authentic_delta)
        ai_deltas.append(ai_delta)
        by_seed[str(seed)] = {
            "stage_a_clean_accuracy": a_clean["accuracy"],
            "stage_b_clean_accuracy": b_clean["accuracy"],
            "clean_accuracy_delta": b_clean["accuracy"] - a_clean["accuracy"],
            "authentic_accuracy_delta": authentic_delta,
            "ai_generated_accuracy_delta": ai_delta,
            "paired_bootstrap_stage_b_minus_stage_a": paired_bootstrap_difference(
                stage_b_records,
                stage_a_records,
                metric="clean_accuracy",
                iterations=iterations,
                seed=seed,
            ),
        }

    stage_a_mean = statistics.fmean(stage_a_clean)
    stage_b_mean = statistics.fmean(stage_b_clean)
    authentic_delta_mean = statistics.fmean(authentic_deltas)
    ai_delta_mean = statistics.fmean(ai_deltas)
    provisional_retain = (
        stage_b_mean > stage_a_mean
        and authentic_delta_mean >= -tolerance
        and ai_delta_mean >= -tolerance
    )
    report = {
        "selection_split": "selection_val",
        "matching_policy": args.matching_policy,
        "seeds": args.seeds,
        "predeclared_layers": config["model"]["rine_layers"],
        "max_per_class_accuracy_regression": tolerance,
        "by_seed": by_seed,
        "aggregate": {
            "stage_a_clean_accuracy_mean": stage_a_mean,
            "stage_b_clean_accuracy_mean": stage_b_mean,
            "clean_accuracy_mean_delta": stage_b_mean - stage_a_mean,
            "authentic_accuracy_mean_delta": authentic_delta_mean,
            "ai_generated_accuracy_mean_delta": ai_delta_mean,
        },
        "provisional_clean_only_decision": "retain" if provisional_retain else "drop",
        "final_decision": "pending_task3_robustness_cells",
        "criterion": (
            "Stage B must improve mean clean accuracy and keep each mean per-class "
            f"regression within {tolerance:.3f}; final retention uses the 50/50 score."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
