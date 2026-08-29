from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cya_detector.reproducibility import collect_run_metadata, write_run_metadata


REPO_ROOT = Path(__file__).resolve().parents[1]


class ReproducibilityTests(unittest.TestCase):
    def test_metadata_contains_required_provenance(self) -> None:
        config = {"runtime": {"seed": 42}}
        metadata = collect_run_metadata(
            config=config,
            repo_root=REPO_ROOT,
            distributions=["definitely-not-an-installed-distribution"],
        )
        self.assertEqual(metadata["seed"], 42)
        self.assertIn("git_commit", metadata)
        self.assertEqual(
            metadata["packages"]["definitely-not-an-installed-distribution"],
            "not-installed",
        )

    def test_metadata_write_is_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "metadata.json"
            write_run_metadata(output, {"seed": 42})
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"seed": 42})
            self.assertFalse(output.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()

