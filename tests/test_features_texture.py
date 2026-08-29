from __future__ import annotations

import unittest

import numpy as np

from cya_detector.features.texture import select_texture_patches


class TexturePatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

    def test_selects_at_most_top_k_non_overlapping_patches(self) -> None:
        image = self.rng.random((128, 128, 3)).astype(np.float32)
        selection = select_texture_patches(image, patch_size=16, top_k=8)
        self.assertLessEqual(len(selection.patch_boxes), 8)
        self.assertEqual(len(selection.patch_boxes), len(selection.energy_scores))

        boxes = selection.patch_boxes
        for i in range(len(boxes)):
            top_i, left_i, height_i, width_i = boxes[i]
            for j in range(i + 1, len(boxes)):
                top_j, left_j, height_j, width_j = boxes[j]
                overlaps = not (
                    top_i + height_i <= top_j
                    or top_j + height_j <= top_i
                    or left_i + width_i <= left_j
                    or left_j + width_j <= left_i
                )
                self.assertFalse(overlaps, f"patches {boxes[i]} and {boxes[j]} overlap")

    def test_same_image_gives_identical_selection(self) -> None:
        image = self.rng.random((128, 128, 3)).astype(np.float32)
        first = select_texture_patches(image, patch_size=16, top_k=8)
        second = select_texture_patches(image, patch_size=16, top_k=8)
        self.assertEqual(first.patch_boxes, second.patch_boxes)
        self.assertEqual(first.energy_scores, second.energy_scores)

    def test_scores_are_sorted_descending(self) -> None:
        image = self.rng.random((128, 128, 3)).astype(np.float32)
        selection = select_texture_patches(image, patch_size=16, top_k=8)
        self.assertEqual(selection.energy_scores, sorted(selection.energy_scores, reverse=True))

    def test_high_detail_region_is_selected_over_flat_region(self) -> None:
        image = np.full((64, 64, 3), 0.5, dtype=np.float32)
        image[:16, :16, :] = self.rng.random((16, 16, 3)).astype(np.float32)
        selection = select_texture_patches(image, patch_size=16, top_k=1)
        self.assertEqual(len(selection.patch_boxes), 1)
        top, left, _, _ = selection.patch_boxes[0]
        self.assertEqual((top, left), (0, 0))

    def test_image_smaller_than_patch_size_returns_empty_selection(self) -> None:
        image = self.rng.random((8, 8, 3)).astype(np.float32)
        selection = select_texture_patches(image, patch_size=16, top_k=4)
        self.assertEqual(selection.patch_boxes, [])
        self.assertEqual(selection.energy_scores, [])

    def test_rejects_non_positive_parameters(self) -> None:
        image = self.rng.random((64, 64, 3)).astype(np.float32)
        with self.assertRaises(ValueError):
            select_texture_patches(image, patch_size=0, top_k=4)
        with self.assertRaises(ValueError):
            select_texture_patches(image, patch_size=16, top_k=0)


if __name__ == "__main__":
    unittest.main()
