"""Post-hoc temperature scaling fit once on clean selection_val logits.

The fitted temperature is reused unchanged for transformed data and at
inference; the binary decision threshold stays fixed regardless of the
fitted temperature.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from scipy.optimize import minimize_scalar

from cya_detector.predictions import PredictionRecord

_TEMPERATURE_BOUNDS = (0.05, 20.0)


class CalibrationError(ValueError):
    """Raised when the input predictions violate the calibration boundary."""


def _stable_sigmoid(value: float) -> float:
    if value >= 0:
        denominator = 1.0 + math.exp(-value)
    else:
        exponentiated = math.exp(value)
        return exponentiated / (1.0 + exponentiated)
    return 1.0 / denominator


def _log_sigmoid_pair(scaled_logit: float) -> tuple[float, float]:
    """Return (log(sigmoid(x)), log(1 - sigmoid(x))) without overflow."""

    if scaled_logit >= 0:
        log_sigmoid = -math.log1p(math.exp(-scaled_logit))
        log_one_minus_sigmoid = -scaled_logit - math.log1p(math.exp(-scaled_logit))
    else:
        log_sigmoid = scaled_logit - math.log1p(math.exp(scaled_logit))
        log_one_minus_sigmoid = -math.log1p(math.exp(scaled_logit))
    return log_sigmoid, log_one_minus_sigmoid


def negative_log_likelihood(
    temperature: float, logits: list[float], targets: list[int]
) -> float:
    if temperature <= 0:
        raise CalibrationError("Temperature must be positive")
    total = 0.0
    for logit, target in zip(logits, targets, strict=True):
        log_sigmoid, log_one_minus_sigmoid = _log_sigmoid_pair(logit / temperature)
        total -= target * log_sigmoid + (1 - target) * log_one_minus_sigmoid
    return total / len(logits)


def _expected_calibration_error(
    probabilities: list[float], targets: list[int], *, bins: int = 10
) -> float:
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for probability, target in zip(probabilities, targets, strict=True):
        index = min(int(probability * bins), bins - 1)
        buckets[index].append((probability, target))
    total = len(probabilities)
    error = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        mean_probability = sum(value for value, _ in bucket) / len(bucket)
        empirical_rate = sum(target for _, target in bucket) / len(bucket)
        error += len(bucket) / total * abs(mean_probability - empirical_rate)
    return error


def fit_temperature(records: Iterable[PredictionRecord]) -> dict[str, Any]:
    """Fit one scalar temperature minimizing NLL on clean selection_val logits."""

    rows = list(records)
    if not rows:
        raise CalibrationError("Cannot fit calibration on an empty prediction set")
    if {row.split for row in rows} != {"selection_val"}:
        raise CalibrationError("Calibration accepts selection_val predictions only")
    if {row.evaluation_cell for row in rows} != {"clean"}:
        raise CalibrationError("Calibration must be fit on clean predictions only")
    if len({row.seed for row in rows}) != 1:
        raise CalibrationError("Calibration requires exactly one seed")
    targets = [row.target for row in rows]
    if len(set(targets)) < 2:
        raise CalibrationError("Calibration requires both classes to be present")
    logits = [row.logit for row in rows]
    if not all(math.isfinite(value) for value in logits):
        raise CalibrationError("Calibration requires finite logits")

    result = minimize_scalar(
        lambda temperature: negative_log_likelihood(temperature, logits, targets),
        bounds=_TEMPERATURE_BOUNDS,
        method="bounded",
    )
    if not result.success:
        raise CalibrationError("Temperature optimization failed to converge")
    temperature = float(result.x)

    raw_probabilities = [_stable_sigmoid(logit) for logit in logits]
    calibrated_probabilities = [_stable_sigmoid(logit / temperature) for logit in logits]

    return {
        "temperature": temperature,
        "sample_count": len(rows),
        "seed": rows[0].seed,
        "nll_before": negative_log_likelihood(1.0, logits, targets),
        "nll_after": negative_log_likelihood(temperature, logits, targets),
        "ece_before": _expected_calibration_error(raw_probabilities, targets),
        "ece_after": _expected_calibration_error(calibrated_probabilities, targets),
    }
