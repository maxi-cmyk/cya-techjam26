"""Manifest selection and image loading for fixed-view CLIP extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from cya_detector.data.manifest import read_manifest
from cya_detector.predictions import LABEL_TO_TARGET


ALLOWED_TRAIN_SPLITS = {"seed_train"}
ALLOWED_SELECTION_SPLITS = {"selection_val"}


@dataclass(frozen=True)
class ManifestExample:
    sample_id: str
    source_id: str
    parent_id: str
    image_path: Path
    sha256: str
    label: str
    split: str
    image_view: str
    transform: str
    transform_parameter: str
    metadata: dict[str, str]

    @property
    def target(self) -> int:
        return LABEL_TO_TARGET[self.label]


def load_examples(
    manifest_path: Path,
    *,
    splits: set[str],
    require_paths: bool = True,
) -> list[ManifestExample]:
    """Load only explicitly allowed splits from a versioned image manifest."""

    if not splits:
        raise ValueError("At least one split must be requested")
    if "final_test" in splits:
        raise ValueError("The training dataset loader cannot read final_test")

    examples: list[ManifestExample] = []
    for row in read_manifest(manifest_path):
        if row.get("split") not in splits:
            continue
        label = row.get("label", "")
        if label not in LABEL_TO_TARGET:
            raise ValueError(f"Unsupported label {label!r} for {row.get('sample_id', '<unknown>')}")
        image_path = Path(row.get("image_path") or row.get("clean_image_path", ""))
        if require_paths and not image_path.is_file():
            raise FileNotFoundError(f"Manifest image is missing: {image_path}")
        examples.append(
            ManifestExample(
                sample_id=row["sample_id"],
                source_id=row.get("source_id", ""),
                parent_id=row.get("parent_id", ""),
                image_path=image_path,
                sha256=row.get("sha256", ""),
                label=label,
                split=row["split"],
                image_view=row.get("image_view", "matched_clean"),
                transform=row.get("transform", "clean") or "clean",
                transform_parameter=row.get("transform_parameter", ""),
                metadata={
                    "dataset_name": row.get("dataset_name", "unknown") or "unknown",
                    "generator_name": row.get("generator_name", "unknown") or "unknown",
                    "generator_checkpoint": row.get("generator_checkpoint", "unknown")
                    or "unknown",
                    "capture_source": row.get("capture_source", "unknown") or "unknown",
                },
            )
        )
    if not examples:
        raise ValueError(f"No records for splits {sorted(splits)} in {manifest_path}")
    return examples


class ClipImageDataset:
    """Lazy image dataset; importing this module does not require torch or Pillow."""

    def __init__(self, examples: list[ManifestExample], processor: Callable[..., Any]) -> None:
        self.examples = examples
        self.processor = processor

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        try:
            from PIL import Image, ImageOps
        except ImportError as exc:
            raise RuntimeError("Pillow is required to load CLIP images") from exc

        example = self.examples[index]
        with Image.open(example.image_path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            pixels = self.processor(images=rgb, return_tensors="pt")["pixel_values"][0]
        return {"pixel_values": pixels, "example": example}
