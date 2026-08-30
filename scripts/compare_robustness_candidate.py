#!/usr/bin/env python3
"""Compare one robustness candidate with its direct parent across fixed seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cya_detector.config import load_config  # noqa: E402
from cya_detector.evaluation.metrics import evaluate_predictions  # noqa: E402
from cya_detector.predictions import read_predictions  # noqa: E402
from cya_detector.transforms.benchmark import benchmark_cells  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--parent-filename", default="best_50_50_predictions.csv")
    parser.add_argument("--candidate-filename", default="best_50_50_predictions.csv")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--config", type=Path, default=Path("configs/colab.json"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _cell_metrics(report: dict[str, Any]) -> list[dict[str, Any]]:
    cells = list(report["robustness"]["cells"].values())
    if report["clean"] is not None:
        cells.append(report["clean"])
    return cells


def _mean_class_accuracy(report: dict[str, Any], key: str) -> float:
    values = [cell[key] for cell in _cell_metrics(report) if cell[key] is not None]
    if not values:
        raise ValueError(f"No {key} values in robustness report")
    return statistics.fmean(values)


def _evaluate(
    path: Path,
    *,
    expected_cells: set[str],
    threshold: float,
) -> tuple[dict[str, Any], set[tuple[str, str, str]]]:
    rows = read_predictions(path)
    if {row.split for row in rows} != {"selection_val"}:
        raise ValueError(f"Robustness comparison accepts selection_val only: {path}")
    observed_cells = {row.evaluation_cell for row in rows if row.evaluation_cell != "clean"}
    if observed_cells != expected_cells:
        missing = sorted(expected_cells - observed_cells)
        extra = sorted(observed_cells - expected_cells)
        raise ValueError(f"Incomplete robustness cells in {path}: missing={missing}, extra={extra}")
    report = evaluate_predictions(rows, threshold=threshold)
    report["mean_authentic_accuracy"] = _mean_class_accuracy(
        report,
        "authentic_accuracy",
    )
    report["mean_ai_generated_accuracy"] = _mean_class_accuracy(
        report,
        "ai_generated_accuracy",
    )
    units = {(row.source_id or row.sample_id, row.evaluation_cell, row.label) for row in rows}
    return report, units


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be a nonempty unique list")
    expected_cells = {cell.cell_id for cell in benchmark_cells(config)}
    max_regression = config["evaluation"]["max_per_class_accuracy_regression"]
    seed_rows: list[dict[str, Any]] = []
    for seed in seeds:
        parent_path = args.parent_root / f"seed_{seed}" / args.parent_filename
        candidate_path = args.candidate_root / f"seed_{seed}" / args.candidate_filename
        parent, parent_units = _evaluate(
            parent_path,
            expected_cells=expected_cells,
            threshold=config["evaluation"]["threshold"],
        )
        candidate, candidate_units = _evaluate(
            candidate_path,
            expected_cells=expected_cells,
            threshold=config["evaluation"]["threshold"],
        )
        if parent_units != candidate_units:
            raise ValueError(f"Parent and candidate evaluation units differ for seed {seed}")
        seed_rows.append(
            {
                "seed": seed,
                "parent_score": parent["selection_score"],
                "candidate_score": candidate["selection_score"],
                "score_delta": candidate["selection_score"] - parent["selection_score"],
                "parent_clean": parent["clean"]["accuracy"],
                "candidate_clean": candidate["clean"]["accuracy"],
                "parent_robustness": parent["robustness"]["mean_accuracy"],
                "candidate_robustness": candidate["robustness"]["mean_accuracy"],
                "authentic_delta": (
                    candidate["mean_authentic_accuracy"] - parent["mean_authentic_accuracy"]
                ),
                "ai_generated_delta": (
                    candidate["mean_ai_generated_accuracy"] - parent["mean_ai_generated_accuracy"]
                ),
                "candidate_worst_cell": min(
                    candidate["robustness"]["cells"],
                    key=lambda cell: candidate["robustness"]["cells"][cell]["accuracy"],
                ),
                "candidate_worst_cell_accuracy": min(
                    cell["accuracy"] for cell in candidate["robustness"]["cells"].values()
                ),
            }
        )

    mean_score_delta = statistics.fmean(row["score_delta"] for row in seed_rows)
    mean_authentic_delta = statistics.fmean(row["authentic_delta"] for row in seed_rows)
    mean_ai_delta = statistics.fmean(row["ai_generated_delta"] for row in seed_rows)
    retain = (
        mean_score_delta > 0.0
        and mean_authentic_delta >= -max_regression
        and mean_ai_delta >= -max_regression
    )
    result = {
        "candidate": args.candidate_name,
        "seeds": seeds,
        "required_cell_count": len(expected_cells),
        "max_per_class_accuracy_regression": max_regression,
        "mean_parent_score": statistics.fmean(row["parent_score"] for row in seed_rows),
        "mean_candidate_score": statistics.fmean(row["candidate_score"] for row in seed_rows),
        "mean_score_delta": mean_score_delta,
        "mean_authentic_delta": mean_authentic_delta,
        "mean_ai_generated_delta": mean_ai_delta,
        "decision": "retain" if retain else "reject",
        "seed_results": seed_rows,
        "final_test_read": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "retention_decision.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output / "seed_comparison.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
