from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from cya_detector.data.manifest import MANIFEST_FIELDS, read_manifest, write_manifest
from cya_detector.data.matched import build_matched_clean, quality_for


class MatchedCleanTests(unittest.TestCase):
    def source_manifest(self, root: Path) -> Path:
        records = []
        for label, color in (("authentic", "red"), ("ai_generated", "blue")):
            source_path = root / f"{label}.png"
            Image.new("RGB", (72, 56), color).save(source_path)
            record = {field: "" for field in MANIFEST_FIELDS}
            record.update(
                {
                    "sample_id": f"{label}__source_original",
                    "source_id": label,
                    "source_path": str(source_path),
                    "image_path": str(source_path),
                    "image_view": "source_original",
                    "label": label,
                    "split": "seed_train",
                    "duplicate_is_primary": "true",
                    "review_required": "false",
                    "eligible_for_split": "true",
                    "c2pa_status": "no_manifest",
                    "width": 72,
                    "height": 56,
                    "original_width": 72,
                    "original_height": 56,
                    "format": "PNG",
                    "original_format": "PNG",
                }
            )
            records.append(record)
        path = root / "source.csv"
        write_manifest(path, records)
        return path

    def test_fixed_policy_is_symmetric_and_preserves_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_manifest = self.source_manifest(root)
            output_manifest = root / "matched.csv"
            report = build_matched_clean(
                source_manifest=source_manifest,
                output_root=root / "views",
                output_manifest=output_manifest,
                report_path=root / "report.json",
                policy="fixed_q96",
                seed=42,
            )

            rows = read_manifest(output_manifest)
            self.assertEqual(report["label_counts"], {"ai_generated": 1, "authentic": 1})
            self.assertEqual({row["normalization_quality"] for row in rows}, {"96"})
            self.assertEqual({(row["width"], row["height"]) for row in rows}, {("72", "56")})
            self.assertTrue(all(row["chroma_subsampling"] == "4:4:4" for row in rows))
            self.assertTrue(all(Path(row["image_path"]).is_file() for row in rows))
            for row in rows:
                self.assertEqual(
                    {field: row[field] for field in (
                        "parent_width",
                        "parent_height",
                        "parent_mode",
                        "parent_format",
                    )},
                    {
                        "parent_width": "",
                        "parent_height": "",
                        "parent_mode": "",
                        "parent_format": "",
                    },
                )

    def test_uniform_quality_is_deterministic(self) -> None:
        first = quality_for("uniform_q95_q100", source_id="sample", seed=42)
        second = quality_for("uniform_q95_q100", source_id="sample", seed=42)
        self.assertEqual(first, second)
        self.assertIn(first, range(95, 101))


if __name__ == "__main__":
    unittest.main()
