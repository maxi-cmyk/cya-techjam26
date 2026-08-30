from __future__ import annotations

import json
import csv
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from cya_detector.config import load_config
from cya_detector.data.dataset import ManifestExample
from cya_detector.predictions import PredictionRecord
from cya_detector.training.robustness import (
    RobustnessContractError,
    controlled_epoch_rows,
    transform_cell_id,
    validate_robustness_bank,
)
from cya_detector.training.robustness_fusion import load_tabular_feature_bank
from cya_detector.training.clip_stage_a import CachedEmbedding
from cya_detector.training.rine_stage_b import train_controlled_rine_head
from cya_detector.transforms.benchmark import benchmark_cells


def example(
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
        image_path=Path(f"unused/{sample_id}.png"),
        sha256=f"sha-{sample_id}",
        label=label,
        split=split,
        image_view="matched_clean" if clean else "benchmark",
        transform="clean" if clean else cell_id.split("_", 1)[0],
        transform_parameter=(
            "" if clean else json.dumps({"cell_id": cell_id}, separators=(",", ":"))
        ),
        metadata={
            "dataset_name": "fixture",
            "generator_name": "unknown",
            "generator_checkpoint": "unknown",
            "capture_source": "unknown",
        },
    )


@dataclass(frozen=True)
class FakeCachedRow:
    example: ManifestExample


class RobustnessTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cells = benchmark_cells(load_config("configs/colab.json"))
        self.clean = tuple(
            example(
                f"parent-{label}-{index}",
                label,
                "selection_val",
                source_id=f"source-{label}-{index}",
            )
            for label in ("authentic", "ai_generated")
            for index in range(2)
        )
        self.variants = tuple(
            example(
                f"{parent.sample_id}--{cell.cell_id}",
                parent.label,
                parent.split,
                source_id=parent.source_id,
                parent_id=parent.sample_id,
                cell_id=cell.cell_id,
            )
            for parent in self.clean
            for cell in self.cells
        )

    def test_complete_bank_is_ordered_and_preserves_all_cells(self) -> None:
        bank = validate_robustness_bank(
            tuple(reversed(self.clean)),
            tuple(reversed(self.variants)),
            self.cells,
            split="selection_val",
        )

        self.assertEqual(len(bank.clean), 4)
        self.assertEqual(len(bank.variants), 4 * len(self.cells))
        self.assertEqual(bank.cell_ids, tuple(cell.cell_id for cell in self.cells))
        self.assertEqual(transform_cell_id(bank.variants[0]), self.cells[0].cell_id)

    def test_bank_rejects_missing_cells_crossed_labels_and_final_test(self) -> None:
        with self.assertRaisesRegex(RobustnessContractError, "missing"):
            validate_robustness_bank(
                self.clean,
                self.variants[:-1],
                self.cells,
                split="selection_val",
            )

        crossed = list(self.variants)
        row = crossed[0]
        crossed[0] = ManifestExample(**{**row.__dict__, "label": "ai_generated"})
        with self.assertRaisesRegex(RobustnessContractError, "crosses"):
            validate_robustness_bank(
                self.clean,
                crossed,
                self.cells,
                split="selection_val",
            )

        with self.assertRaisesRegex(RobustnessContractError, "final_test"):
            validate_robustness_bank([], [], self.cells, split="final_test")

    def test_controlled_epoch_selects_only_complete_cached_views(self) -> None:
        train_clean = tuple(
            ManifestExample(**{**row.__dict__, "split": "seed_train"}) for row in self.clean
        )
        train_variants = tuple(
            ManifestExample(**{**row.__dict__, "split": "seed_train"}) for row in self.variants
        )
        parent_rows = [FakeCachedRow(row) for row in train_clean]
        bank_rows = [FakeCachedRow(row) for row in (*train_clean, *train_variants)]

        selected = controlled_epoch_rows(
            parent_rows,
            bank_rows,
            self.cells,
            epoch_size=56,
            project_seed=42,
            epoch=0,
        )

        self.assertEqual(len(selected), 56)
        self.assertEqual(
            sum(transform_cell_id(row.example) == "clean" for row in selected),
            28,
        )
        self.assertEqual(
            {row.example.label for row in selected},
            {"authentic", "ai_generated"},
        )

    def test_prediction_uses_encoded_cell_id(self) -> None:
        record = PredictionRecord(
            sample_id="variant",
            source_id="source",
            parent_id="parent",
            split="selection_val",
            label="authentic",
            logit=0.0,
            probability=0.5,
            checkpoint="fixture",
            seed=42,
            matching_policy="fixed_q96",
            transform="jpeg",
            transform_parameter=json.dumps({"cell_id": "jpeg_q90", "quality": 90}),
        )
        self.assertEqual(record.evaluation_cell, "jpeg_q90")

    def test_feature_bank_keeps_magnitude_residual_and_lab_but_drops_phase_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frequency = root / "frequency.csv"
            auxiliary = root / "auxiliary.csv"
            base_rows = [
                ("a", "authentic", "seed_train"),
                ("b", "ai_generated", "selection_val"),
            ]
            with frequency.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "sample_id",
                        "label",
                        "split",
                        "feature_valid",
                        "magnitude_bin_0",
                        "residual_mean",
                        "phase_bin_0",
                    ],
                )
                writer.writeheader()
                for index, (sample_id, label, split) in enumerate(base_rows):
                    writer.writerow(
                        {
                            "sample_id": sample_id,
                            "label": label,
                            "split": split,
                            "feature_valid": "true",
                            "magnitude_bin_0": index + 1,
                            "residual_mean": index + 2,
                            "phase_bin_0": index + 3,
                        }
                    )
            with auxiliary.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "sample_id",
                        "label",
                        "split",
                        "feature_valid",
                        "rgb_r_g_global",
                        "lab_l_a_global",
                    ],
                )
                writer.writeheader()
                for index, (sample_id, label, split) in enumerate(base_rows):
                    writer.writerow(
                        {
                            "sample_id": sample_id,
                            "label": label,
                            "split": split,
                            "feature_valid": "true",
                            "rgb_r_g_global": index + 4,
                            "lab_l_a_global": index + 5,
                        }
                    )

            bank = load_tabular_feature_bank(
                variant="frequency_lab",
                frequency_table=frequency,
                auxiliary_table=auxiliary,
            )

        self.assertEqual(
            bank.names,
            (
                "frequency:magnitude_bin_0",
                "frequency:residual_mean",
                "lab:lab_l_a_global",
            ),
        )
        self.assertEqual(bank.values_by_sample_id["a"].tolist(), [1.0, 2.0, 5.0])

    def test_controlled_rine_selects_a_real_50_50_checkpoint(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def cache_rows(examples: tuple[ManifestExample, ...]) -> list[CachedEmbedding]:
                rows = []
                for index, item in enumerate(examples):
                    path = root / "cache" / f"{item.sample_id}.pt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    base = 1.0 if item.label == "ai_generated" else -1.0
                    torch.save(torch.full((4, 8), base + index * 0.0001), path)
                    rows.append(CachedEmbedding(item, f"key-{item.sample_id}", path))
                return rows

            train_clean = tuple(
                example(
                    f"train-{label}-{index}",
                    label,
                    "seed_train",
                    source_id=f"train-source-{label}-{index}",
                )
                for label in ("authentic", "ai_generated")
                for index in range(2)
            )
            train_variants = tuple(
                example(
                    f"{parent.sample_id}--{cell.cell_id}",
                    parent.label,
                    "seed_train",
                    source_id=parent.source_id,
                    parent_id=parent.sample_id,
                    cell_id=cell.cell_id,
                )
                for parent in train_clean
                for cell in self.cells
            )
            selection_clean = tuple(
                example(
                    f"selection-{label}",
                    label,
                    "selection_val",
                    source_id=f"selection-source-{label}",
                )
                for label in ("authentic", "ai_generated")
            )
            selection_variants = tuple(
                example(
                    f"{parent.sample_id}--{cell.cell_id}",
                    parent.label,
                    "selection_val",
                    source_id=parent.source_id,
                    parent_id=parent.sample_id,
                    cell_id=cell.cell_id,
                )
                for parent in selection_clean
                for cell in self.cells
            )
            train_parent_rows = cache_rows(train_clean)
            train_bank_rows = cache_rows((*train_clean, *train_variants))
            selection_rows = cache_rows((*selection_clean, *selection_variants))
            output = root / "run"

            summary = train_controlled_rine_head(
                train_parent_rows=train_parent_rows,
                train_bank_rows=train_bank_rows,
                selection_rows=selection_rows,
                cells=self.cells,
                output_directory=output,
                matching_policy="fixed_q96",
                layers=[1, 2, 3, 4],
                resolved_revision="fixture",
                manifest_sha256="fixture",
                seed=42,
                device="cpu",
                learning_rate=0.01,
                weight_decay=0.0,
                warmup_fraction=0.0,
                max_epochs=1,
                early_stopping_patience=1,
                physical_batch_size=8,
                effective_batch_size=8,
                threshold=0.5,
                run_configuration={},
                epoch_size=56,
            )

            self.assertIsNotNone(summary["best_values"]["selection_score"])
            self.assertTrue((output / "best_50_50.pt").is_file())
            self.assertTrue((output / "best_50_50_predictions.csv").is_file())
            self.assertFalse(summary["robustness_pending_task3"])


if __name__ == "__main__":
    unittest.main()
