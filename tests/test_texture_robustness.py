from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from cya_detector.config import ConfigError, load_config
from cya_detector.data.manifest import (
    MANIFEST_FIELDS,
    read_manifest,
    sha256_file,
    write_manifest,
)
from cya_detector.evaluation.texture_robustness import (
    STAGE1_CELL_IDS,
    TextureRobustnessError,
    materialize_texture_stage1,
    validate_robustness_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/colab.json"


class TextureRobustnessContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CONFIG_PATH)

    def test_texture_robustness_contract_is_exact_and_locked(self) -> None:
        contract = validate_robustness_contract(self.config)

        self.assertEqual(contract.cell_ids, STAGE1_CELL_IDS)
        self.assertEqual(
            contract.variants,
            ("global_only", "local_only", "global_local"),
        )
        self.assertEqual(contract.seeds, (42, 43, 44))
        self.assertEqual(
            contract.controlling_comparators,
            ("global_only", "controlled_rine"),
        )
        self.assertEqual(contract.aggregate_class_tolerance, 0.01)
        self.assertEqual(contract.worst_cell_tolerance, 0.03)

    def test_texture_robustness_contract_is_immutable(self) -> None:
        contract = validate_robustness_contract(self.config)

        with self.assertRaises(FrozenInstanceError):
            contract.worst_cell_tolerance = 0.5

    def test_contract_rejects_cells_absent_from_benchmark_contract(self) -> None:
        candidate = copy.deepcopy(self.config)
        candidate["benchmark_transforms"]["jpeg_quality"] = [70, 50, 30]

        with self.assertRaises(ConfigError):
            validate_robustness_contract(candidate)


class TextureRobustnessMaterializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = load_config(CONFIG_PATH)
        self.clean_parent_ids = {
            "authentic__matched_clean__fixed_q96",
            "ai_generated__matched_clean__fixed_q96",
        }
        self.input_manifest = self.root / "fixed_q96_manifest.csv"
        self.output_root = self.root / "images"
        self.output_manifest = self.root / "transformed_selection_val.csv"
        self.report_path = self.root / "materialization_report.json"
        self._write_parents()

    @property
    def materialize_args(self) -> dict[str, object]:
        return {
            "input_manifest": self.input_manifest,
            "output_root": self.output_root,
            "output_manifest": self.output_manifest,
            "report_path": self.report_path,
            "config": self.config,
        }

    def _write_parents(self) -> None:
        rows = []
        for label, color in (("authentic", "red"), ("ai_generated", "blue")):
            image_path = self.root / f"{label}.jpg"
            Image.new("RGB", (32, 24), color).save(
                image_path,
                format="JPEG",
                quality=96,
                subsampling=0,
                optimize=False,
                progressive=False,
                exif=b"",
            )
            row = {field: "" for field in MANIFEST_FIELDS}
            row.update(
                {
                    "sample_id": f"{label}__matched_clean__fixed_q96",
                    "source_id": label,
                    "parent_id": f"{label}__source_original",
                    "source_path": str((self.root / f"source-{label}.png").resolve()),
                    "image_path": str(image_path.resolve()),
                    "clean_image_path": str(image_path.resolve()),
                    "image_view": "matched_clean",
                    "sha256": sha256_file(image_path),
                    "label": label,
                    "split": "selection_val",
                    "dataset_name": "fixture",
                    "transform": "clean",
                    "transform_seed": "42",
                    "normalization_codec": "JPEG",
                    "normalization_quality": "96",
                    "output_storage_format": "JPEG",
                }
            )
            rows.append(row)
        write_manifest(self.input_manifest, list(reversed(rows)))

    def _mutate_parent(self, field: str, value: str) -> None:
        rows = read_manifest(self.input_manifest)
        rows[-1][field] = value
        write_manifest(self.input_manifest, rows)

    def test_materializes_exact_stage1_cells_directly_from_selection_clean(self) -> None:
        report = materialize_texture_stage1(**self.materialize_args)
        rows = read_manifest(self.output_manifest)

        self.assertEqual(set(report["cell_counts"]), set(STAGE1_CELL_IDS))
        self.assertEqual(len(rows), len(self.clean_parent_ids) * len(STAGE1_CELL_IDS))
        self.assertEqual({row["split"] for row in rows}, {"selection_val"})
        self.assertEqual({row["image_view"] for row in rows}, {"benchmark"})
        self.assertEqual({row["parent_id"] for row in rows}, self.clean_parent_ids)
        self.assertEqual(report["expected_image_count"], len(rows))
        self.assertEqual(report["stage1_cell_ids"], list(STAGE1_CELL_IDS))

    def test_records_verified_parent_and_output_hashes(self) -> None:
        report = materialize_texture_stage1(**self.materialize_args)

        parent_hashes = {
            row["sample_id"]: row["sha256"] for row in read_manifest(self.input_manifest)
        }
        for row in read_manifest(self.output_manifest):
            self.assertEqual(row["parent_sha256"], parent_hashes[row["parent_id"]])
            self.assertEqual(row["sha256"], sha256_file(Path(row["image_path"])))
        self.assertEqual(report["input_manifest_sha256"], sha256_file(self.input_manifest))
        self.assertEqual(report["output_manifest_sha256"], sha256_file(self.output_manifest))

    def test_rejects_forbidden_split_view_policy_or_chaining_before_writing(self) -> None:
        cases = (
            ("split", "final_test"),
            ("split", "self_train_pool"),
            ("image_view", "benchmark"),
            ("transform", "blur"),
            ("normalization_quality", "95"),
            ("sample_id", "wrong-policy-parent"),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                self._write_parents()
                self._mutate_parent(field, value)
                with self.assertRaises(TextureRobustnessError):
                    materialize_texture_stage1(**self.materialize_args)
                self.assertFalse(self.output_root.exists())
                self.assertFalse(self.output_manifest.exists())
                self.assertFalse(self.report_path.exists())

    def test_filters_seed_train_without_reading_or_materializing_it(self) -> None:
        rows = read_manifest(self.input_manifest)
        seed_row = dict(rows[0])
        seed_row.update(
            {
                "sample_id": "sealed-seed-parent__matched_clean__fixed_q96",
                "split": "seed_train",
                "image_path": str(self.root / "must-not-be-read.jpg"),
                "clean_image_path": str(self.root / "must-not-be-read.jpg"),
                "sha256": "0" * 64,
            }
        )
        write_manifest(self.input_manifest, rows + [seed_row])

        report = materialize_texture_stage1(**self.materialize_args)

        output_rows = read_manifest(self.output_manifest)
        self.assertEqual(report["parent_count"], 2)
        self.assertEqual(report["source_row_count"], 3)
        self.assertEqual(report["ignored_seed_train_count"], 1)
        self.assertEqual(len(output_rows), 18)
        self.assertNotIn(seed_row["sample_id"], {row["parent_id"] for row in output_rows})

    def test_rejects_parent_hash_mismatch_before_writing(self) -> None:
        self._mutate_parent("sha256", "0" * 64)

        with self.assertRaisesRegex(TextureRobustnessError, "SHA-256"):
            materialize_texture_stage1(**self.materialize_args)

        self.assertFalse(self.output_root.exists())
        self.assertFalse(self.output_manifest.exists())
        self.assertFalse(self.report_path.exists())

    def test_rerun_is_byte_identical(self) -> None:
        first_report = materialize_texture_stage1(**self.materialize_args)
        first_manifest_bytes = self.output_manifest.read_bytes()
        first_report_bytes = self.report_path.read_bytes()
        first_image_hashes = {
            row["sample_id"]: row["sha256"] for row in read_manifest(self.output_manifest)
        }

        second_report = materialize_texture_stage1(**self.materialize_args)

        self.assertEqual(second_report, first_report)
        self.assertEqual(self.output_manifest.read_bytes(), first_manifest_bytes)
        self.assertEqual(self.report_path.read_bytes(), first_report_bytes)
        self.assertEqual(
            {row["sample_id"]: row["sha256"] for row in read_manifest(self.output_manifest)},
            first_image_hashes,
        )

    def test_late_validation_failure_does_not_publish_csv_or_json(self) -> None:
        def invalid_materializer(**kwargs: object) -> dict[str, object]:
            write_manifest(Path(kwargs["output_manifest"]), [])
            Path(kwargs["report_path"]).write_text("{}", encoding="utf-8")
            return {"parent_count": 2, "image_count": 0, "cell_counts": {}}

        with patch(
            "cya_detector.evaluation.texture_robustness.materialize_benchmarks",
            side_effect=invalid_materializer,
        ):
            with self.assertRaises(TextureRobustnessError):
                materialize_texture_stage1(**self.materialize_args)

        self.assertFalse(self.output_manifest.exists())
        self.assertFalse(self.report_path.exists())
        self.assertEqual(list(self.root.glob(".*stage1*")), [])

    def test_cli_materializes_the_locked_matrix(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/materialize_texture_robustness.py",
                "--input-manifest",
                str(self.input_manifest),
                "--output-root",
                str(self.output_root),
                "--output-manifest",
                str(self.output_manifest),
                "--report",
                str(self.report_path),
                "--config",
                str(CONFIG_PATH),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(read_manifest(self.output_manifest)), 18)
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["stage1_cell_ids"], list(STAGE1_CELL_IDS))
        self.assertIn("Materialized 18 images", result.stdout)


if __name__ == "__main__":
    unittest.main()
