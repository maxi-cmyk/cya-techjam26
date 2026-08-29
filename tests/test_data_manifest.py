from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from cya_detector.data.manifest import (
    DatasetContractError,
    build_source_manifest,
    normalize_label,
)


def patterned_image(path: Path, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (48, 40), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 4, 20, 18), outline=(255, 255, 255), width=2)
    draw.line((0, 39, 47, 0), fill=(0, 0, 0), width=2)
    image.save(path, format="PNG")


def write_labels(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filename", "label", "label_name"])
        writer.writeheader()
        writer.writerows(rows)


class SourceManifestTests(unittest.TestCase):
    def test_binary_filter_corruption_and_duplicate_grouping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            patterned_image(images / "real.png", (180, 30, 30))
            patterned_image(images / "fake.png", (20, 40, 200))
            shutil.copy2(images / "fake.png", images / "fake-copy.png")
            patterned_image(images / "tampered.png", (40, 180, 60))
            (images / "broken.png").write_bytes(b"not an image")
            write_labels(
                root / "labels.csv",
                [
                    {"filename": "real.png", "label": 0, "label_name": "real"},
                    {"filename": "fake.png", "label": 1, "label_name": "synthetic"},
                    {"filename": "fake-copy.png", "label": 1, "label_name": "synthetic"},
                    {"filename": "tampered.png", "label": 2, "label_name": "tampered"},
                    {"filename": "broken.png", "label": 0, "label_name": "real"},
                ],
            )

            records, report = build_source_manifest(dataset_root=root, check_c2pa=False)

            self.assertEqual(len(records), 4)
            self.assertEqual(report["excluded_label_counts"], {"tampered": 1})
            self.assertEqual(report["corrupt_count"], 1)
            self.assertEqual(report["duplicate_groups"], 1)
            self.assertEqual(report["cross_label_duplicate_groups"], 0)
            duplicate_rows = [row for row in records if row["label"] == "ai_generated"]
            self.assertEqual(len({row["duplicate_group_id"] for row in duplicate_rows}), 1)
            self.assertEqual(
                sum(row["duplicate_is_primary"] == "true" for row in duplicate_rows), 1
            )
            self.assertEqual(
                sum(row["eligible_for_split"] == "true" for row in duplicate_rows), 1
            )

    def test_cross_label_exact_duplicate_requires_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            patterned_image(images / "real.png", (100, 120, 140))
            shutil.copy2(images / "real.png", images / "fake.png")
            write_labels(
                root / "labels.csv",
                [
                    {"filename": "real.png", "label": 0, "label_name": "real"},
                    {"filename": "fake.png", "label": 1, "label_name": "synthetic"},
                ],
            )

            records, report = build_source_manifest(dataset_root=root, check_c2pa=False)

            self.assertEqual(report["cross_label_duplicate_groups"], 1)
            self.assertTrue(all(row["review_required"] == "true" for row in records))
            self.assertTrue(all(row["eligible_for_split"] == "false" for row in records))

    def test_unknown_label_fails_closed(self) -> None:
        with self.assertRaises(DatasetContractError):
            normalize_label({"label": "7", "label_name": "mystery"})


if __name__ == "__main__":
    unittest.main()

