"""Argument parsing and wiring for the directory-inference CLI.

Owns the exit-code boundary: 0 full success, 1 fatal, 2 argparse usage error
(handled by argparse itself), 3 partial success.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from cya_detector.inference.c2pa import has_verified_ai_generation_claim
from cya_detector.inference.contracts import EXIT_FATAL, Predictor
from cya_detector.inference.output import publish
from cya_detector.inference.runner import InferenceRunFailure, run_inference


def stub_predictor(image: Image.Image) -> float:
    """Placeholder predictor for the Task 10A skeleton.

    Task 10B replaces this with the calibrated controlled-RINE seed-42
    adapter; everything else in this pipeline is unaffected by that swap.
    """

    return 0.5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score a directory of images for likelihood of being AI-generated."
    )
    parser.add_argument("image_dir", type=Path, help="Directory to scan recursively.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write predictions.json and report.json into.",
    )
    return parser


def main(argv: list[str] | None = None, *, predict_probability: Predictor = stub_predictor) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    def progress(line: str) -> None:
        print(line)

    try:
        result = run_inference(
            args.image_dir,
            predict_probability=predict_probability,
            has_verified_ai_generation_claim=has_verified_ai_generation_claim,
            print_progress=progress,
        )
    except InferenceRunFailure as exc:
        print(f"fatal: {exc}", file=sys.stderr)
        return EXIT_FATAL

    exit_code = publish(result, args.output_dir)
    print(
        f"discovered={result.summary.discovered} predicted={result.summary.predicted} "
        f"invalid={result.summary.invalid} exit={exit_code}"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
