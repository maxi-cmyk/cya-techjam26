#!/usr/bin/env python3
"""Run the sealed final_test evaluation exactly once. This is irreversible.

Requires --i-understand-this-is-irreversible in addition to everything else.
Refuses to run if a prior result already exists at --output-root; it is not
resumable or overwrite-able by design.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cya_detector.data.manifest import sha256_file  # noqa: E402
from cya_detector.evaluation.final_test import evaluate_final_test  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="fixed_q96_manifest.csv")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Controlled-RINE seed-42 best_50_50.pt")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument(
        "--i-understand-this-is-irreversible",
        dest="confirmed",
        action="store_true",
        help="Required. final_test may be evaluated only once, ever.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.confirmed:
        print(
            "Refusing to run: pass --i-understand-this-is-irreversible. "
            "final_test may be evaluated only once, ever.",
            file=sys.stderr,
        )
        return 2

    from cya_detector.inference.rine_predictor import RinePredictor

    predictor = RinePredictor(checkpoint_path=args.checkpoint, device=args.device)
    import torch

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_identity = {
        "stage": checkpoint.get("stage"),
        "seed": checkpoint.get("seed"),
        "matching_policy": checkpoint.get("matching_policy"),
        "layers": checkpoint.get("layers"),
        "resolved_model_revision": checkpoint.get("resolved_model_revision"),
        "manifest_sha256": checkpoint.get("manifest_sha256"),
        "checkpoint_sha256": sha256_file(args.checkpoint),
    }

    report = evaluate_final_test(
        manifest_path=args.manifest,
        predict_probability=predictor,
        threshold=args.threshold,
        output_root=args.output_root,
        checkpoint_identity=checkpoint_identity,
        confirm_final_test_read=True,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
