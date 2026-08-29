from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from cya_detector.config import load_config
from cya_detector.data.manifest import (
    MANIFEST_FIELDS,
    read_manifest,
    sha256_file,
    write_manifest,
)
from cya_detector.transforms.benchmark import (
    TransformContractError,
    apply_benchmark,
    benchmark_cells,
    derive_seed,
)
from cya_detector.transforms.materialize import (
    TransformMaterializationError,
    materialize_benchmarks,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/colab.json"


class TransformMaterializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = load_config(CONFIG_PATH)
        by_id = {cell.cell_id: cell for cell in benchmark_cells(self.config)}
        self.jpeg_cell = by_id["jpeg_q90"]
        self.resize_cell = by_id["resize_scale_0.5"]
        self.parent_ids = {
            "authentic__matched_clean__fixed_q96",
            "ai_generated__matched_clean__fixed_q96",
        }
        self.manifest = self._write_parents()

    def _write_parents(self) -> Path:
        records = []
        for label, color in (("authentic", "red"), ("ai_generated", "blue")):
            image_path = self.root / f"{label}.jpg"
            Image.new("RGB", (19, 17), color).save(
                image_path,
                format="JPEG",
                quality=96,
                subsampling=0,
            )
            parent_id = f"{label}__matched_clean__fixed_q96"
            record = {field: "" for field in MANIFEST_FIELDS}
            record.update(
                {
                    "sample_id": parent_id,
                    "source_id": label,
                    "parent_id": f"{label}__source_original",
                    "source_path": str((self.root / f"source-{label}.png").resolve()),
                    "image_path": str(image_path.resolve()),
                    "clean_image_path": str(image_path.resolve()),
                    "image_view": "matched_clean",
                    "sha256": "deliberately-not-trusted",
                    "label": label,
                    "split": "final_test",
                    "dataset_name": "fixture",
                    "license_status": "test-only",
                    "transform": "clean",
                    "width": 999,
                    "height": 997,
                    "format": "PNG",
                    "mode": "L",
                    "original_width": 101,
                    "original_height": 103,
                    "output_storage_format": "JPEG",
                }
            )
            records.append(record)
        manifest = self.root / "parents.csv"
        write_manifest(manifest, list(reversed(records)))
        return manifest

    def _materialize(
        self,
        *,
        output_root: Path | None = None,
        output_manifest: Path | None = None,
        report_path: Path | None = None,
        overwrite: bool = False,
    ) -> dict[str, object]:
        return materialize_benchmarks(
            input_manifest=self.manifest,
            output_root=output_root or self.root / "variants",
            output_manifest=output_manifest or self.root / "variants.csv",
            report_path=report_path or self.root / "report.json",
            config=self.config,
            cells=(self.resize_cell, self.jpeg_cell),
            overwrite=overwrite,
        )

    def test_materializes_cells_directly_from_clean_parents(self) -> None:
        report = self._materialize()

        rows = read_manifest(self.root / "variants.csv")
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["parent_id"] for row in rows}, self.parent_ids)
        self.assertEqual({row["image_view"] for row in rows}, {"benchmark"})
        self.assertEqual(
            report["cell_counts"],
            {"jpeg_q90": 2, "resize_scale_0.5": 2},
        )
        self.assertEqual(
            [(row["parent_id"], json.loads(row["transform_parameter"])["cell_id"])
             for row in rows],
            sorted(
                (parent_id, cell_id)
                for parent_id in self.parent_ids
                for cell_id in ("jpeg_q90", "resize_scale_0.5")
            ),
        )

    def test_records_exact_provenance_and_verified_storage_formats(self) -> None:
        self._materialize()
        rows = read_manifest(self.root / "variants.csv")

        for row in rows:
            cell_id = json.loads(row["transform_parameter"])["cell_id"]
            parent_path = self.root / f"{row['source_id']}.jpg"
            output_path = Path(row["image_path"])
            expected_format = "JPEG" if cell_id == "jpeg_q90" else "PNG"
            expected_suffix = ".jpg" if expected_format == "JPEG" else ".png"
            self.assertEqual(output_path.suffix, expected_suffix)
            with Image.open(output_path) as image:
                image.verify()
            with Image.open(output_path) as image:
                self.assertEqual(image.format, expected_format)
                self.assertEqual(image.mode, "RGB")
            self.assertEqual(row["sha256"], sha256_file(output_path))
            self.assertEqual(row["parent_sha256"], sha256_file(parent_path))
            self.assertEqual(row["clean_image_path"], str(parent_path.resolve()))
            self.assertEqual(row["parent_width"], "19")
            self.assertEqual(row["parent_height"], "17")
            self.assertEqual(row["parent_mode"], "RGB")
            self.assertEqual(row["parent_format"], "JPEG")
            self.assertEqual(row["original_width"], "101")
            self.assertEqual(row["original_height"], "103")
            self.assertEqual(row["transform_version"], "task3-v1")
            self.assertEqual(row["preprocessing_version"], "clip-crop-v1")
            self.assertEqual(row["output_storage_format"], expected_format)
            self.assertEqual(
                row["transform_seed"],
                str(derive_seed(42, row["parent_id"], cell_id)),
            )
            self.assertEqual(row["transform"], json.loads(row["transform_parameter"])["name"])
            self.assertEqual(
                row["transform_parameter"],
                json.dumps(
                    json.loads(row["transform_parameter"]),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            self.assertEqual(
                row["realized_parameters"],
                json.dumps(
                    json.loads(row["realized_parameters"]),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )

        self.assertFalse(list((self.root / "variants").rglob("*.tmp*")))

    def test_jpeg_materialization_persists_the_single_encoder_stream(self) -> None:
        self._materialize()
        jpeg_row = next(
            row
            for row in read_manifest(self.root / "variants.csv")
            if row["transform"] == "jpeg" and row["source_id"] == "authentic"
        )
        parent_id = jpeg_row["parent_id"]
        with Image.open(self.root / "authentic.jpg") as parent:
            expected = apply_benchmark(parent, self.jpeg_cell, parent_id, 42)

        self.assertIsNotNone(expected.encoded_bytes)
        self.assertEqual(Path(jpeg_row["image_path"]).read_bytes(), expected.encoded_bytes)

    def test_chained_parent_is_rejected_before_any_output_is_written(self) -> None:
        records = read_manifest(self.manifest)
        records[-1]["image_view"] = "benchmark"
        records[-1]["transform"] = "blur"
        write_manifest(self.manifest, records)

        with self.assertRaises(TransformContractError):
            self._materialize()

        self.assertFalse((self.root / "variants").exists())
        self.assertFalse((self.root / "variants.csv").exists())
        self.assertFalse((self.root / "report.json").exists())

    def test_differing_output_collision_requires_explicit_overwrite(self) -> None:
        self._materialize()
        rows = read_manifest(self.root / "variants.csv")
        output_path = Path(rows[0]["image_path"])
        original_hash = rows[0]["sha256"]
        Image.new("RGB", (19, 17), "green").save(output_path, format=rows[0]["format"])
        collision_hash = sha256_file(output_path)

        with self.assertRaisesRegex(TransformMaterializationError, "collision"):
            self._materialize()

        self.assertEqual(sha256_file(output_path), collision_hash)
        self._materialize(overwrite=True)
        self.assertEqual(sha256_file(output_path), original_hash)

    def test_regeneration_is_byte_identical_and_order_independent(self) -> None:
        first_root = self.root / "first"
        second_root = self.root / "second"
        first_manifest = self.root / "first.csv"
        second_manifest = self.root / "second.csv"
        first_report = self.root / "first.json"
        second_report = self.root / "second.json"

        first = self._materialize(
            output_root=first_root,
            output_manifest=first_manifest,
            report_path=first_report,
        )
        records = read_manifest(self.manifest)
        write_manifest(self.manifest, list(reversed(records)))
        second = materialize_benchmarks(
            input_manifest=self.manifest,
            output_root=second_root,
            output_manifest=second_manifest,
            report_path=second_report,
            config=self.config,
            cells=(self.jpeg_cell, self.resize_cell),
        )

        first_rows = read_manifest(first_manifest)
        second_rows = read_manifest(second_manifest)
        for left, right in zip(first_rows, second_rows, strict=True):
            self.assertEqual(
                {key: value for key, value in left.items() if key != "image_path"},
                {key: value for key, value in right.items() if key != "image_path"},
            )
            self.assertEqual(Path(left["image_path"]).read_bytes(), Path(right["image_path"]).read_bytes())
        for key in (
            "cell_counts",
            "image_count",
            "label_counts",
            "parent_count",
            "preprocessing_version",
            "seed",
            "transform_version",
        ):
            self.assertEqual(first[key], second[key])

    def test_late_parent_read_failure_does_not_publish_final_csv_or_json(self) -> None:
        records = read_manifest(self.manifest)
        Path(records[-1]["image_path"]).unlink()
        write_manifest(self.manifest, records)

        with self.assertRaises(TransformMaterializationError):
            self._materialize()

        self.assertFalse((self.root / "variants.csv").exists())
        self.assertFalse((self.root / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
