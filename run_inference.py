#!/usr/bin/env python3
"""Submission entry point: score a directory of images for AI-generation likelihood.

Usage:
    python run_inference.py <image_dir> --output-dir <dir>

Writes ``<output-dir>/predictions.json`` (the public ``{"image_path", "pred"}``
contract) and ``<output-dir>/report.json`` (a validation summary plus any
per-image errors). All pipeline logic lives under
``src/cya_detector/inference/``; this file only wires up the entry point so
it is easy to find at the repository root.

Exit codes: 0 full success, 1 fatal run failure, 2 argument usage error,
3 partial success (some images were invalid; predictions.json still
contains every image that scored successfully).

Uses the committed controlled-RINE seed-42 checkpoint by default. The
``--no-checkpoint`` option explicitly selects a constant test stub and is not
part of the submission inference path.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cya_detector.inference.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
