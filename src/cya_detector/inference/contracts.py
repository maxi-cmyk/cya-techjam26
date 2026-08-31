"""Pure types shared across the inference pipeline. No I/O in this module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image

EXIT_SUCCESS = 0
EXIT_FATAL = 1
EXIT_USAGE_ERROR = 2
EXIT_PARTIAL_SUCCESS = 3

VALID_ERROR_CODES = frozenset(
    {
        "file_unreadable",
        "decode_failed",
        "unsupported_image",
        "invalid_dimensions",
        "decompression_bomb",
    }
)


class Predictor(Protocol):
    """Injected, not registered. Receives a normalized owned RGB image."""

    def __call__(self, image: Image.Image) -> float: ...


@dataclass(frozen=True)
class ValidationError:
    """One per-image failure. ``message`` must never contain absolute paths,
    tracebacks, or unstable exception text — it is always one of a small set
    of fixed templates per ``code``."""

    image_path: str
    code: str
    message: str

    def __post_init__(self) -> None:
        if self.code not in VALID_ERROR_CODES:
            raise ValueError(f"Unknown validation error code: {self.code!r}")


@dataclass(frozen=True)
class PredictionRecord:
    """One row of the public predictions.json contract."""

    image_path: str
    pred: float


@dataclass(frozen=True)
class RunSummary:
    discovered: int
    predicted: int
    invalid: int


@dataclass(frozen=True)
class RunResult:
    predictions: tuple[PredictionRecord, ...]
    errors: tuple[ValidationError, ...]
    summary: RunSummary

    @property
    def exit_code(self) -> int:
        return EXIT_PARTIAL_SUCCESS if self.summary.invalid > 0 else EXIT_SUCCESS
