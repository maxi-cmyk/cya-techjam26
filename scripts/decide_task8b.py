#!/usr/bin/env python3
"""Record the fail-closed Task 8B physical-feature retention decision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cya_detector.data.manifest import sha256_file, write_json  # noqa: E402


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-readiness", type=Path, required=True)
    parser.add_argument("--prnu-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matched = _read(args.matched_readiness)
    prnu = _read(args.prnu_validation)
    matched_ready = bool(matched.get("training_ready"))
    prnu_validated = bool(prnu.get("signal_validated"))
    ca_validated = bool(matched.get("chromatic_aberration", {}).get("ready"))
    eligible_features = [
        feature
        for feature, ready in (("prnu", prnu_validated), ("chromatic_aberration", ca_validated))
        if ready
    ]
    fusion_training_eligible = matched_ready and bool(eligible_features)
    report = {
        "status": (
            "fusion_training_eligible"
            if fusion_training_eligible
            else "completed_no_physical_feature_retained"
        ),
        "matched_training_ready": matched_ready,
        "nuisance_maximum_balanced_accuracy": matched.get("nuisance", {}).get(
            "maximum_balanced_accuracy"
        ),
        "prnu_signal_validated": prnu_validated,
        "prnu_roc_auc": prnu.get("roc_auc"),
        "prnu_minimum_auc": prnu.get("minimum_auc"),
        "chromatic_aberration_validated": ca_validated,
        "eligible_physical_features": eligible_features,
        "fusion_training_eligible": fusion_training_eligible,
        "fusion_training_run": False,
        "rine_backbone_changed": False,
        "retained_features": [],
        "decision": (
            "Proceed to frozen-RINE projection/fusion training."
            if fusion_training_eligible
            else "Do not fit binary projection/fusion weights; no physical estimator passed its independent validation gate."
        ),
        "matched_readiness_report": str(args.matched_readiness.resolve()),
        "matched_readiness_sha256": sha256_file(args.matched_readiness),
        "prnu_validation_report": str(args.prnu_validation.resolve()),
        "prnu_validation_sha256": sha256_file(args.prnu_validation),
        "physical_claim": False,
        "camera_authentication_claim": False,
        "final_test_read": False,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
