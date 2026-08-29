#!/usr/bin/env python3
"""Select a Task 2 matching policy from multi-seed Stage A clean results."""

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

from cya_detector.evaluation.bootstrap import paired_bootstrap_difference  # noqa: E402
from cya_detector.predictions import read_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task4-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policies", nargs="+", default=["fixed_q96", "uniform_q95_q100"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policies: dict[str, dict[str, object]] = {}
    for policy in args.policies:
        values: list[float] = []
        runs: dict[str, float] = {}
        for seed in args.seeds:
            path = args.task4_root / policy / f"seed_{seed}" / "training_summary.json"
            if not path.is_file():
                raise SystemExit(f"Missing Stage A result: {path}")
            summary = json.loads(path.read_text(encoding="utf-8"))
            value = summary["best_values"]["clean"]
            if value is None:
                raise SystemExit(f"No clean selection metric in {path}")
            values.append(float(value))
            runs[str(seed)] = float(value)
        policies[policy] = {
            "clean_accuracy_by_seed": runs,
            "clean_accuracy_mean": statistics.fmean(values),
            "clean_accuracy_sample_stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }

    ranked = sorted(
        policies,
        key=lambda name: (-float(policies[name]["clean_accuracy_mean"]), name),
    )
    paired_comparisons: dict[str, object] = {}
    if len(ranked) >= 2:
        left, right = ranked[:2]
        for seed in args.seeds:
            left_path = (
                args.task4_root / left / f"seed_{seed}" / "best_clean_predictions.csv"
            )
            right_path = (
                args.task4_root / right / f"seed_{seed}" / "best_clean_predictions.csv"
            )
            paired_comparisons[str(seed)] = paired_bootstrap_difference(
                read_predictions(left_path),
                read_predictions(right_path),
                metric="clean_accuracy",
                iterations=args.bootstrap_iterations,
                seed=seed,
            )
    report = {
        "selection_split": "selection_val",
        "criterion": "highest_mean_clean_accuracy_across_declared_seeds",
        "seeds": args.seeds,
        "policies": policies,
        "selected_policy": ranked[0],
        "paired_clean_accuracy_comparison": {
            "left_policy": ranked[0],
            "right_policy": ranked[1] if len(ranked) >= 2 else None,
            "by_seed": paired_comparisons,
        },
        "note": "Clean-only pilot decision; rerun retention after Task 3 enables the locked 50/50 score.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
