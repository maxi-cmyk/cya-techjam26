#!/usr/bin/env python3
"""Validate configuration, imports, metadata, and accelerator visibility."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cya_detector.config import ConfigError, load_config  # noqa: E402
from cya_detector.reproducibility import (  # noqa: E402
    collect_run_metadata,
    write_run_metadata,
)


DEPENDENCIES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "scikit-image": "skimage",
    "opencv-python-headless": "cv2",
    "Pillow": "PIL",
    "c2pa-python": "c2pa",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/colab.json")
    parser.add_argument(
        "--allow-missing-dependencies",
        action="store_true",
        help="Report missing packages without failing; intended for bootstrap checks only.",
    )
    parser.add_argument(
        "--metadata-output",
        help="Optional JSON path for a run-environment snapshot.",
    )
    return parser.parse_args()


def check_imports() -> tuple[dict[str, str], list[str]]:
    results: dict[str, str] = {}
    missing: list[str] = []
    for distribution, module in DEPENDENCIES.items():
        try:
            importlib.import_module(module)
            results[distribution] = "ok"
        except Exception as exc:  # Import failures can include missing native libraries.
            results[distribution] = f"error: {type(exc).__name__}: {exc}"
            missing.append(distribution)
    return results, missing


def accelerator_status() -> dict[str, object]:
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return {"torch_available": False, "cuda_available": False, "device": "cpu"}

    cuda_available = bool(torch.cuda.is_available())
    return {
        "torch_available": True,
        "cuda_available": cuda_available,
        "device": torch.cuda.get_device_name(0) if cuda_available else "cpu",
    }


def main() -> int:
    args = parse_args()
    try:
        config = load_config(REPO_ROOT / args.config)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2

    imports, missing = check_imports()
    status = {
        "config": "ok",
        "schema_version": config["schema_version"],
        "runtime": config["runtime"]["platform"],
        "model": config["model"]["identifier"],
        "imports": imports,
        "accelerator": accelerator_status(),
    }
    print(json.dumps(status, indent=2, sort_keys=True))

    if args.metadata_output:
        metadata = collect_run_metadata(
            config=config,
            repo_root=REPO_ROOT,
            distributions=DEPENDENCIES,
        )
        written = write_run_metadata(REPO_ROOT / args.metadata_output, metadata)
        print(f"Wrote metadata: {written}")

    if missing and not args.allow_missing_dependencies:
        print(
            "Missing or broken dependencies: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

