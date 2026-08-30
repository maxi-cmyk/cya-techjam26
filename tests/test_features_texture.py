from __future__ import annotations

import unittest

import numpy as np

from cya_detector.features.texture import (
    prepare_texture_patch_views,
    select_texture_patches,
    texture_patch_cache_key,
)


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

    def test_prepares_exact_grid_and_fixed_availability_mask(self) -> None:
        image = np.zeros((336, 336, 3), dtype=np.float32)
        views = prepare_texture_patch_views(image, patch_size=112, patch_count=4)
        self.assertEqual(len(views.patches), 4)
        self.assertEqual(views.availability_mask, (True, True, True, True))
        self.assertTrue(all(patch.shape == (112, 112, 3) for patch in views.patches))

    def test_selector_candidates_are_nine_and_only_top_four_are_prepared(self) -> None:
        image = self.rng.random((336, 336, 3)).astype(np.float32)
        views = prepare_texture_patch_views(image, patch_size=112, patch_count=4)
        self.assertEqual(len(select_texture_patches(image, patch_size=112, top_k=9).patch_boxes), 9)
        self.assertEqual(len(views.patches), 4)
        self.assertEqual(views.patch_boxes, tuple(select_texture_patches(image, patch_size=112, top_k=4).patch_boxes))

    def test_small_source_is_padded_with_odd_remainder_on_right_and_bottom(self) -> None:
        image = np.ones((100, 101, 3), dtype=np.float32)
        views = prepare_texture_patch_views(image, patch_size=112, patch_count=4)
        self.assertEqual(views.original_shape, (100, 101))
        self.assertEqual(views.padded_shape, (112, 112))
        self.assertEqual(views.patch_boxes, ((0, 0, 112, 112),))
        np.testing.assert_array_equal(views.patches[0][6:106, 5:106], image)
        np.testing.assert_array_equal(views.patches[0][:6], 0.0)
        np.testing.assert_array_equal(views.patches[0][106:], 0.0)
        np.testing.assert_array_equal(views.patches[0][:, :5], 0.0)
        np.testing.assert_array_equal(views.patches[0][:, 106:], 0.0)

    def test_padding_has_one_real_patch_and_trailing_false_entries(self) -> None:
        image = self.rng.random((100, 101, 3)).astype(np.float32)
        views = prepare_texture_patch_views(image, patch_size=112, patch_count=4)
        self.assertEqual(len(views.patches), 1)
        self.assertEqual(views.availability_mask, (True, False, False, False))

    def test_accepts_rgba_but_rejects_grayscale_and_nonfinite(self) -> None:
        rgba = np.zeros((112, 112, 4), dtype=np.float32)
        self.assertEqual(len(prepare_texture_patch_views(rgba, patch_size=112, patch_count=1).patches), 1)
        with self.assertRaises(ValueError):
            prepare_texture_patch_views(np.zeros((112, 112), dtype=np.float32), patch_size=112, patch_count=1)
        bad = np.zeros((112, 112, 3), dtype=np.float32)
        bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            prepare_texture_patch_views(bad, patch_size=112, patch_count=1)

    def test_rejects_patch_count_above_fixed_maximum(self) -> None:
        image = np.zeros((336, 336, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            prepare_texture_patch_views(image, patch_size=112, patch_count=5)

    def test_prepared_patches_do_not_share_writable_memory(self) -> None:
        image = np.zeros((112, 112, 3), dtype=np.float32)
        views = prepare_texture_patch_views(image, patch_size=112, patch_count=1)
        self.assertFalse(np.shares_memory(image, views.patches[0]))

    def test_patch_cache_key_is_repeatable_and_identity_specific(self) -> None:
        args = dict(
            image_sha256="image", patch_boxes=((0, 0, 112, 112),),
            model_identifier="model", resolved_revision="rev",
            preprocessing_version="prep", extractor_version="extractor",
        )
        first = texture_patch_cache_key(**args)
        self.assertEqual(first, texture_patch_cache_key(**args))
        for field in args:
            changed = dict(args)
            value = changed[field]
            changed[field] = value + "-changed" if isinstance(value, str) else ((1, 1, 112, 112),)
            self.assertNotEqual(first, texture_patch_cache_key(**changed), field)


if __name__ == "__main__":
    unittest.main()
