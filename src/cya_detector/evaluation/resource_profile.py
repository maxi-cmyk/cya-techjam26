"""Latency, GPU memory, and checkpoint disk-footprint measurement.

Task 10B resource-measurement step: profiles an arbitrary
``predict_probability`` callable and reports the checkpoint files the
packaged model depends on, without importing or assuming anything about
which model that callable wraps. Torch is used opportunistically for GPU
peak-memory tracking when available; it is never a hard requirement, so
this module also profiles a pure-Python fixture predictor in tests.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from PIL import Image


class ResourceProfileError(ValueError):
    """Raised when the profiling boundary is violated."""


def checkpoint_disk_footprint(paths: Sequence[Path]) -> dict[str, Any]:
    """Report on-disk size for every checkpoint/asset file the packaged model needs."""

    resolved = [Path(path) for path in paths]
    if not resolved:
        raise ResourceProfileError("Cannot measure an empty checkpoint file list")
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise ResourceProfileError(f"Missing checkpoint file(s): {missing}")
    sizes = {str(path): path.stat().st_size for path in resolved}
    return {"file_sizes_bytes": sizes, "total_bytes": sum(sizes.values())}


def profile_predictor(
    predict_probability: Callable[[Image.Image], float],
    images: Sequence[Image.Image],
    *,
    warmup: int = 1,
) -> dict[str, Any]:
    """Measure per-call latency and peak GPU memory for a frozen predictor."""

    if not images:
        raise ResourceProfileError("Cannot profile against an empty image set")
    if warmup < 0:
        raise ResourceProfileError("warmup must be non-negative")

    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except ImportError:
        torch = None
        cuda_available = False

    for image in images[:warmup]:
        predict_probability(image)

    if cuda_available:
        torch.cuda.reset_peak_memory_stats()

    latencies: list[float] = []
    for image in images:
        started = time.perf_counter()
        prediction = predict_probability(image)
        latencies.append(time.perf_counter() - started)
        if isinstance(prediction, bool) or not isinstance(prediction, (int, float)):
            raise ResourceProfileError(
                "Predictor must return a numeric probability while profiling"
            )

    sorted_latencies = sorted(latencies)
    percentile_95_index = min(
        len(sorted_latencies) - 1, int(0.95 * (len(sorted_latencies) - 1))
    )
    peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated()) if cuda_available else 0

    return {
        "sample_count": len(images),
        "warmup_count": min(warmup, len(images)),
        "latency_seconds": {
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "p95": sorted_latencies[percentile_95_index],
            "max": max(latencies),
            "total": sum(latencies),
        },
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
    }
