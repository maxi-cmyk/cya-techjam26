from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cya_detector.data.dataset import load_examples
from cya_detector.data.manifest import MANIFEST_FIELDS, write_manifest


def row(sample_id: str, split: str, label: str) -> dict[str, str]:
    value = {field: "" for field in MANIFEST_FIELDS}
    value.update(
        {
            "sample_id": sample_id,
            "source_id": f"source-{sample_id}",
            "parent_id": f"parent-{sample_id}",
            "image_path": f"/{sample_id}.jpg",
            "sha256": f"hash-{sample_id}",
            "label": label,
            "split": split,
            "image_view": "matched_clean",
            "transform": "clean",
        }
    )
    return value


class TrainingDatasetTests(unittest.TestCase):
    def test_loader_filters_requested_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            write_manifest(
                path,
                [
                    row("train", "seed_train", "authentic"),
                    row("selection", "selection_val", "ai_generated"),
                ],
            )
            examples = load_examples(path, splits={"selection_val"}, require_paths=False)
        self.assertEqual([example.sample_id for example in examples], ["selection"])

    def test_loader_refuses_final_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            write_manifest(path, [row("final", "final_test", "authentic")])
            with self.assertRaises(ValueError):
                load_examples(path, splits={"final_test"}, require_paths=False)


if __name__ == "__main__":
    unittest.main()
