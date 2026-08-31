from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_INFERENCE = REPO_ROOT / "run_inference.py"


class InferenceCliTests(unittest.TestCase):
    """The break caught here is a submission script that isn't actually
    runnable, or that publishes output on a fatal run."""

    def setUp(self) -> None:
        self.root = Path(".tmp") / f"inference-cli-{uuid.uuid4().hex}"
        self.image_dir = self.root / "images"
        self.output_dir = self.root / "output"
        self.image_dir.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(RUN_INFERENCE), *args],
            cwd=REPO_ROOT, capture_output=True, check=False, text=True,
        )

    def test_help_exits_zero(self) -> None:
        result = self._run("--help")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_required_argument_is_a_usage_error(self) -> None:
        result = self._run(str(self.image_dir))
        self.assertEqual(result.returncode, 2, result.stderr)

    def test_full_success_scores_every_image_with_the_stub_predictor(self) -> None:
        Image.new("RGB", (4, 4), (1, 2, 3)).save(self.image_dir / "a.png")
        Image.new("RGB", (4, 4), (4, 5, 6)).save(self.image_dir / "b.jpg")

        result = self._run(str(self.image_dir), "--output-dir", str(self.output_dir), "--no-checkpoint")

        self.assertEqual(result.returncode, 0, result.stderr)
        predictions = json.loads((self.output_dir / "predictions.json").read_text())
        self.assertEqual(
            sorted(predictions, key=lambda row: row["image_path"]),
            [{"image_path": "a.png", "pred": 0.5}, {"image_path": "b.jpg", "pred": 0.5}],
        )
        report = json.loads((self.output_dir / "report.json").read_text())
        self.assertEqual(report["summary"], {"discovered": 2, "predicted": 2, "invalid": 0})
        self.assertEqual(report["errors"], [])

    def test_progress_and_summary_lines_are_printed_to_stdout(self) -> None:
        Image.new("RGB", (4, 4)).save(self.image_dir / "a.png")

        result = self._run(str(self.image_dir), "--output-dir", str(self.output_dir), "--no-checkpoint")

        self.assertIn("predicted a.png", result.stdout)
        self.assertIn("discovered=1 predicted=1 invalid=0 exit=0", result.stdout)

    def test_partial_success_publishes_both_files_with_exit_three(self) -> None:
        Image.new("RGB", (4, 4)).save(self.image_dir / "good.png")
        (self.image_dir / "bad.png").write_bytes(b"not an image")

        result = self._run(str(self.image_dir), "--output-dir", str(self.output_dir), "--no-checkpoint")

        self.assertEqual(result.returncode, 3, result.stderr)
        predictions = json.loads((self.output_dir / "predictions.json").read_text())
        self.assertEqual(predictions, [{"image_path": "good.png", "pred": 0.5}])
        report = json.loads((self.output_dir / "report.json").read_text())
        self.assertEqual(report["summary"], {"discovered": 2, "predicted": 1, "invalid": 1})
        self.assertEqual(report["errors"][0]["image_path"], "bad.png")
        self.assertEqual(report["errors"][0]["code"], "unsupported_image")
        self.assertNotIn("/", report["errors"][0]["message"])

    def test_missing_default_checkpoint_falls_back_to_stub_with_a_loud_warning(self) -> None:
        Image.new("RGB", (4, 4)).save(self.image_dir / "a.png")

        result = self._run(
            str(self.image_dir), "--output-dir", str(self.output_dir),
            "--checkpoint", str(self.root / "does-not-exist.pt"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("WARNING", result.stderr)
        self.assertIn("stub predictor", result.stderr)
        predictions = json.loads((self.output_dir / "predictions.json").read_text())
        self.assertEqual(predictions, [{"image_path": "a.png", "pred": 0.5}])

    def test_empty_directory_is_fatal_and_publishes_nothing(self) -> None:
        result = self._run(str(self.image_dir), "--output-dir", str(self.output_dir), "--no-checkpoint")

        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse((self.output_dir / "predictions.json").exists())
        self.assertFalse((self.output_dir / "report.json").exists())

    def test_fatal_run_never_touches_a_prior_successful_run(self) -> None:
        Image.new("RGB", (4, 4)).save(self.image_dir / "a.png")
        first = self._run(str(self.image_dir), "--output-dir", str(self.output_dir), "--no-checkpoint")
        self.assertEqual(first.returncode, 0, first.stderr)
        before = (self.output_dir / "predictions.json").read_bytes()

        # Remove all images so the second run hits the fatal empty-discovery path.
        (self.image_dir / "a.png").unlink()
        second = self._run(str(self.image_dir), "--output-dir", str(self.output_dir), "--no-checkpoint")

        self.assertEqual(second.returncode, 1, second.stderr)
        self.assertEqual((self.output_dir / "predictions.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
