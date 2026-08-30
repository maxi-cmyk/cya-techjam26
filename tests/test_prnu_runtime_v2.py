from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from cya_detector.config import load_config
from cya_detector.data.dataset import ManifestExample
from cya_detector.data.manifest import write_manifest
from cya_detector.features.prnu_runtime_v2 import extract_prnu_runtime_v2
from cya_detector.training.prnu_runtime_v2 import (
    PrnuRuntimeRow,
    audit_prnu_runtime_rows,
    extract_prnu_runtime_manifest,
    train_prnu_runtime_baseline,
)
from cya_detector.transforms.benchmark import benchmark_cells


def _example(
    sample_id: str,
    label: str,
    split: str,
    *,
    source_id: str,
    parent_id: str = "",
    cell_id: str = "clean",
) -> ManifestExample:
    clean = cell_id == "clean"
    return ManifestExample(
        sample_id=sample_id,
        source_id=source_id,
        parent_id=parent_id,
        image_path=Path("unused.png"),
        sha256=f"sha-{sample_id}",
        label=label,
        split=split,
        image_view="matched_clean" if clean else "benchmark",
        transform="clean" if clean else cell_id.split("_", 1)[0],
        transform_parameter="" if clean else f'{{"cell_id":"{cell_id}"}}',
        metadata={
            "dataset_name": "fixture",
            "generator_name": "unknown",
            "generator_checkpoint": "unknown",
            "capture_source": "unknown",
        },
    )


class PrnuRuntimeV2Tests(unittest.TestCase):
    def test_runtime_vector_is_reference_free_and_small_images_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            random = np.random.default_rng(42)
            supported = root / "supported.png"
            small = root / "small.png"
            Image.fromarray(random.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)).save(
                supported
            )
            Image.new("RGB", (96, 96), (128, 128, 128)).save(small)

            result = extract_prnu_runtime_v2(
                supported, crop_size=128, wavelet_levels=2, block_size=32
            )
            unsupported = extract_prnu_runtime_v2(
                small, crop_size=128, wavelet_levels=2, block_size=32
            )

        self.assertTrue(result.valid["prnu_v2_runtime"])
        self.assertTrue(np.all(np.isfinite(result.values)))
        self.assertFalse(result.metadata["reference_comparison_used"])
        self.assertFalse(unsupported.valid["prnu_v2_runtime"])
        self.assertTrue(np.all(unsupported.values == 0.0))

    def test_readiness_is_balanced_and_extraction_never_reads_final_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            random = np.random.default_rng(7)
            for split in ("seed_train", "selection_val"):
                for label in ("authentic", "ai_generated"):
                    sample_id = f"{split}-{label}"
                    path = root / f"{sample_id}.png"
                    Image.fromarray(
                        random.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
                    ).save(path)
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "source_id": sample_id,
                            "image_path": str(path),
                            "sha256": sample_id,
                            "label": label,
                            "split": split,
                            "image_view": "matched_clean",
                            "transform": "clean",
                        }
                    )
            manifest = root / "manifest.csv"
            output = root / "features.csv"
            report_path = root / "report.json"
            write_manifest(manifest, rows)
            report = extract_prnu_runtime_manifest(
                manifest_path=manifest,
                output_path=output,
                report_path=report_path,
                cache_root=root / "cache",
                matching_policy="fixed_q96",
                configuration={
                    "extractor_version": "fixture",
                    "crop_size": 128,
                    "wavelet": "db2",
                    "wavelet_levels": 2,
                    "edge_keep_quantile": 0.75,
                    "block_size": 32,
                    "minimum_eligibility_rate": 1.0,
                    "maximum_label_gap": 0.0,
                    "workers": 1,
                    "sampling_epochs": 1,
                },
                workers=1,
            )
            with output.open(newline="", encoding="utf-8") as stream:
                feature_rows = list(csv.DictReader(stream))

        self.assertTrue(report["ready_for_binary_ablation"])
        self.assertFalse(report["reference_comparison_used"])
        self.assertFalse(report["device_identity_used"])
        self.assertFalse(report["final_test_read"])
        self.assertEqual(len(feature_rows), 4)
        self.assertTrue(all(row["feature_valid"] == "true" for row in feature_rows))

        with self.assertRaisesRegex(ValueError, "final_test"):
            audit_prnu_runtime_rows(
                [{**rows[0], "split": "final_test"}],
                crop_size=128,
                minimum_eligibility_rate=1.0,
                maximum_label_gap=0.0,
            )

    def test_readiness_gates_clean_rows_but_allows_downscaled_eval_views(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for split in ("seed_train", "selection_val"):
                for label in ("authentic", "ai_generated"):
                    for image_view, transform, size in (
                        ("matched_clean", "clean", 128),
                        ("benchmark", "resize", 64),
                    ):
                        sample_id = f"{split}-{label}-{image_view}"
                        path = root / f"{sample_id}.png"
                        Image.new("RGB", (size, size), (128, 128, 128)).save(path)
                        rows.append(
                            {
                                "sample_id": sample_id,
                                "source_id": f"source-{split}-{label}",
                                "image_path": str(path),
                                "sha256": sample_id,
                                "label": label,
                                "split": split,
                                "image_view": image_view,
                                "transform": transform,
                            }
                        )

            report = audit_prnu_runtime_rows(
                rows,
                crop_size=128,
                minimum_eligibility_rate=1.0,
                maximum_label_gap=0.0,
            )
            manifest = root / "manifest.csv"
            output = root / "features.csv"
            write_manifest(manifest, rows)
            extraction_report = extract_prnu_runtime_manifest(
                manifest_path=manifest,
                output_path=output,
                report_path=root / "report.json",
                cache_root=root / "cache",
                matching_policy="fixed_q96",
                configuration={
                    "extractor_version": "fixture-clean-gate",
                    "crop_size": 128,
                    "wavelet": "db2",
                    "wavelet_levels": 2,
                    "edge_keep_quantile": 0.75,
                    "block_size": 32,
                    "minimum_eligibility_rate": 1.0,
                    "maximum_label_gap": 0.0,
                    "workers": 1,
                    "sampling_epochs": 1,
                },
                workers=1,
            )
            with output.open(newline="", encoding="utf-8") as stream:
                feature_rows = list(csv.DictReader(stream))

        self.assertTrue(report["ready_for_binary_ablation"])
        self.assertEqual(report["readiness_scope"], "matched_clean_rows_only")
        self.assertTrue(all(group["eligibility_rate"] == 1.0 for group in report["groups"]))
        self.assertTrue(
            all(group["eligibility_rate"] == 0.5 for group in report["all_view_coverage_groups"])
        )
        self.assertTrue(extraction_report["ready_for_binary_ablation"])
        benchmark_rows = [row for row in feature_rows if row["image_view"] == "benchmark"]
        self.assertEqual(len(benchmark_rows), 4)
        self.assertTrue(all(row["feature_valid"] == "true" for row in benchmark_rows))
        self.assertTrue(all(float(row["prnu_v2_runtime_eligible"]) == 0.0 for row in benchmark_rows))
        self.assertTrue(all(float(row["prnu_v2_runtime_valid"]) == 0.0 for row in benchmark_rows))

    def test_prnu_only_diagnostic_uses_complete_locked_selection_bank(self) -> None:
        cells = benchmark_cells(load_config("configs/colab.json"))
        train_clean = tuple(
            _example(
                f"train-{label}-{index}",
                label,
                "seed_train",
                source_id=f"train-source-{label}-{index}",
            )
            for label in ("authentic", "ai_generated")
            for index in range(2)
        )
        selection_clean = tuple(
            _example(
                f"selection-{label}", label, "selection_val", source_id=f"selection-{label}"
            )
            for label in ("authentic", "ai_generated")
        )

        def bank(clean_rows: tuple[ManifestExample, ...]) -> list[PrnuRuntimeRow]:
            examples = [
                *clean_rows,
                *(
                    _example(
                        f"{parent.sample_id}--{cell.cell_id}",
                        parent.label,
                        parent.split,
                        source_id=parent.source_id,
                        parent_id=parent.sample_id,
                        cell_id=cell.cell_id,
                    )
                    for parent in clean_rows
                    for cell in cells
                ),
            ]
            return [
                PrnuRuntimeRow(
                    example=row,
                    values=np.asarray(
                        [1.0 if row.label == "ai_generated" else -1.0, 1.0, 1.0],
                        dtype=np.float32,
                    ),
                )
                for row in examples
            ]

        train_rows = bank(train_clean)
        selection_rows = bank(selection_clean)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            summary = train_prnu_runtime_baseline(
                train_parent_rows=train_rows[: len(train_clean)],
                train_bank_rows=train_rows,
                selection_rows=selection_rows,
                cells=cells,
                output_directory=output,
                matching_policy="fixed_q96",
                seed=42,
                threshold=0.5,
                sampling_epochs=1,
                epoch_size=56,
            )

            self.assertTrue((output / "best_50_50_predictions.csv").is_file())
            self.assertEqual(summary["selection_metrics"]["robustness"]["cell_count"], 14)
            self.assertFalse(summary["final_test_read"])


if __name__ == "__main__":
    unittest.main()
