from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cya_detector.data.manifest import MANIFEST_FIELDS, write_manifest
from cya_detector.data.nuisance import audit_nuisance_manifest


class NuisanceAuditTests(unittest.TestCase):
    def test_audit_produces_bounded_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for label in ("authentic", "ai_generated"):
                for index in range(20):
                    record = {field: "" for field in MANIFEST_FIELDS}
                    record.update(
                        {
                            "sample_id": f"{label}-{index}",
                            "source_id": f"{label}-{index}",
                            "label": label,
                            "eligible_for_split": "true",
                            "width": 336,
                            "height": 336,
                            "file_size": (50000 if label == "authentic" else 20000) + index,
                            "normalization_quality": 96,
                            "format": "JPEG",
                            "original_format": "PNG" if label == "authentic" else "JPEG",
                            "normalization_codec": "JPEG",
                            "chroma_subsampling": "4:4:4",
                        }
                    )
                    records.append(record)
            manifest_path = root / "manifest.csv"
            write_manifest(manifest_path, records)

            report = audit_nuisance_manifest(
                manifest_path=manifest_path,
                output_path=root / "report.json",
                seed=42,
            )

            self.assertEqual(report["sample_count"], 40)
            self.assertGreaterEqual(report["test_accuracy"], 0.0)
            self.assertLessEqual(report["test_accuracy"], 1.0)
            self.assertGreater(report["test_roc_auc"], 0.5)


if __name__ == "__main__":
    unittest.main()

