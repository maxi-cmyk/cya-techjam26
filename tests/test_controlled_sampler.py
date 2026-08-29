from __future__ import annotations

import random
import unittest
from collections import Counter
from unittest.mock import patch

from PIL import Image

from cya_detector.config import load_config
from cya_detector.transforms import (
    TrainingView,
    apply_training_view,
    benchmark_cells,
    build_controlled_epoch,
)
from cya_detector.transforms.benchmark import TransformCell, TransformContractError, TransformResult


class ControlledSamplerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cells = benchmark_cells(load_config("configs/colab.json"))
        self.cells_by_id = {cell.cell_id: cell for cell in self.cells}
        self.records = tuple(
            {
                "sample_id": f"{label}-{index}",
                "label": label,
                "image_path": f"unused/{label}-{index}.jpg",
                "image_view": "matched_clean",
                "transform": "clean",
            }
            for label in ("authentic", "ai_generated")
            for index in range(3)
        )

    def schedule(self, *, epoch: int, epoch_size: int = 112) -> tuple[TrainingView, ...]:
        return build_controlled_epoch(
            self.records,
            self.cells,
            epoch_size=epoch_size,
            project_seed=42,
            epoch=epoch,
        )

    def test_epoch_is_label_and_view_balanced(self) -> None:
        schedule = self.schedule(epoch=0)

        counts = Counter((view.label, view.cell_id == "clean") for view in schedule)

        self.assertEqual(set(counts.values()), {28})

    def test_cells_are_uniform_per_label(self) -> None:
        schedule = self.schedule(epoch=0)

        for label in ("authentic", "ai_generated"):
            counts = Counter(
                view.cell_id
                for view in schedule
                if view.label == label and view.cell_id != "clean"
            )
            self.assertEqual(set(counts), {cell.cell_id for cell in self.cells})
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_parents_cycle_uniformly_per_label(self) -> None:
        schedule = self.schedule(epoch=0)

        for label in ("authentic", "ai_generated"):
            counts = Counter(view.sample_id for view in schedule if view.label == label)
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_schedule_repeats_by_epoch_without_consuming_global_rng(self) -> None:
        original_state = random.getstate()
        self.addCleanup(random.setstate, original_state)
        random.seed(7)
        state_before = random.getstate()

        first = self.schedule(epoch=3)
        repeated = self.schedule(epoch=3)
        next_epoch = self.schedule(epoch=4)

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, next_epoch)
        self.assertEqual(random.getstate(), state_before)

    def test_partial_epoch_keeps_label_and_view_counts_within_one(self) -> None:
        for epoch_size in range(1, 10):
            with self.subTest(epoch_size=epoch_size):
                schedule = self.schedule(epoch=0, epoch_size=epoch_size)
                label_counts = Counter(view.label for view in schedule)
                view_counts = Counter(view.cell_id == "clean" for view in schedule)
                self.assertLessEqual(
                    abs(label_counts["authentic"] - label_counts["ai_generated"]),
                    1,
                )
                self.assertLessEqual(abs(view_counts[True] - view_counts[False]), 1)

    def test_schedule_construction_never_reads_images(self) -> None:
        with patch(
            "cya_detector.transforms.controlled.Image.open",
            side_effect=AssertionError("schedule construction must stay lazy"),
        ):
            schedule = self.schedule(epoch=0)

        self.assertEqual(len(schedule), 112)

    def test_rejects_missing_or_unknown_labels(self) -> None:
        only_authentic = tuple(
            record for record in self.records if record["label"] == "authentic"
        )
        with self.assertRaisesRegex(ValueError, "ai_generated"):
            build_controlled_epoch(
                only_authentic,
                self.cells,
                epoch_size=4,
                project_seed=42,
                epoch=0,
            )

        records = list(self.records)
        records[0] = {**records[0], "label": "synthetic"}
        with self.assertRaisesRegex(ValueError, "synthetic"):
            build_controlled_epoch(
                records,
                self.cells,
                epoch_size=4,
                project_seed=42,
                epoch=0,
            )

    def test_rejects_chained_parent_before_scheduling(self) -> None:
        records = list(self.records)
        records[-1] = {
            **records[-1],
            "image_view": "benchmark",
            "transform": "blur",
        }

        with self.assertRaises(TransformContractError):
            build_controlled_epoch(
                records,
                self.cells,
                epoch_size=4,
                project_seed=42,
                epoch=0,
            )

    def test_rejects_invalid_cells_and_epoch_sizes(self) -> None:
        invalid_cell_sets = (
            (),
            (self.cells[0], self.cells[0]),
            (TransformCell("blur", 1.0, "clean", "PNG"),),
        )
        for cells in invalid_cell_sets:
            with self.subTest(cells=cells), self.assertRaisesRegex(ValueError, "cell"):
                build_controlled_epoch(
                    self.records,
                    cells,
                    epoch_size=4,
                    project_seed=42,
                    epoch=0,
                )

        for epoch_size in (0, -1):
            with self.subTest(epoch_size=epoch_size), self.assertRaisesRegex(
                ValueError, "epoch_size"
            ):
                build_controlled_epoch(
                    self.records,
                    self.cells,
                    epoch_size=epoch_size,
                    project_seed=42,
                    epoch=0,
                )

    def test_clean_application_reads_once_and_skips_transform(self) -> None:
        source = Image.new("RGB", (9, 7), "blue")
        view = TrainingView("authentic-0", "authentic", "parent.jpg", "clean", 123)

        with (
            patch("cya_detector.transforms.controlled.Image.open", return_value=source) as opener,
            patch(
                "cya_detector.transforms.controlled.apply_benchmark",
                side_effect=AssertionError("clean views must not transform"),
            ),
        ):
            result = apply_training_view(view, self.cells_by_id, input_size=8)

        opener.assert_called_once_with("parent.jpg")
        self.assertEqual(result.size, (8, 8))
        self.assertEqual(result.mode, "RGB")

    def test_transformed_application_reads_once_uses_decoded_image_and_separate_crop_seed(
        self,
    ) -> None:
        source = Image.new("RGB", (9, 7), "blue")
        decoded = Image.new("RGB", (9, 7), "red")
        view = TrainingView(
            "authentic-0",
            "authentic",
            "parent.jpg",
            self.cells[0].cell_id,
            123,
        )
        observed: dict[str, int] = {}

        def transform(
            image: Image.Image,
            cell: TransformCell,
            sample_id: str,
            project_seed: int,
        ) -> TransformResult:
            observed["transform_seed"] = project_seed
            return TransformResult(decoded, {}, encoded_bytes=b"not-an-image")

        def crop(image: Image.Image, size: int, *, seed: int) -> Image.Image:
            observed["crop_seed"] = seed
            return image.crop((0, 0, size, size))

        with (
            patch("cya_detector.transforms.controlled.Image.open", return_value=source) as opener,
            patch(
                "cya_detector.transforms.controlled.apply_benchmark", side_effect=transform
            ) as transformer,
            patch("cya_detector.transforms.controlled.random_crop_input", side_effect=crop),
        ):
            result = apply_training_view(view, self.cells_by_id, input_size=6)

        opener.assert_called_once_with("parent.jpg")
        transformer.assert_called_once()
        self.assertEqual(observed["transform_seed"], view.seed)
        self.assertNotEqual(observed["crop_seed"], observed["transform_seed"])
        self.assertEqual(result.getpixel((0, 0)), (255, 0, 0))

    def test_application_rejects_unknown_or_mismatched_cell_ids_before_reading(self) -> None:
        views_and_cells = (
            (
                TrainingView("a", "authentic", "parent.jpg", "missing", 1),
                self.cells_by_id,
            ),
            (
                TrainingView("a", "authentic", "parent.jpg", "alias", 1),
                {"alias": self.cells[0]},
            ),
        )

        with patch(
            "cya_detector.transforms.controlled.Image.open",
            side_effect=AssertionError("invalid views must fail before reading"),
        ):
            for view, cells_by_id in views_and_cells:
                with self.subTest(view=view), self.assertRaisesRegex(ValueError, "cell"):
                    apply_training_view(view, cells_by_id, input_size=8)


if __name__ == "__main__":
    unittest.main()
