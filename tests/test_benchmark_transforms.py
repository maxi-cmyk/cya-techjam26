from __future__ import annotations

import hashlib
import io
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from cya_detector.config import load_config
from cya_detector.transforms.benchmark import (
    TransformCell,
    TransformContractError,
    apply_benchmark,
    benchmark_cells,
    derive_seed,
    validate_parent_record,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/colab.json"


class BenchmarkTransformTests(unittest.TestCase):
    @staticmethod
    def gradient(size: tuple[int, int]) -> Image.Image:
        width, height = size
        values = np.arange(width * height * 3, dtype=np.uint32).reshape(height, width, 3)
        return Image.fromarray((values % 256).astype(np.uint8))

    def setUp(self) -> None:
        self.cells = benchmark_cells(load_config(CONFIG_PATH))
        self.by_id = {cell.cell_id: cell for cell in self.cells}
        self.resize_half = self.by_id["resize_scale_0.5"]
        self.crop = self.by_id["center_crop_0.8"]

    def test_config_expands_to_fourteen_stable_cells(self) -> None:
        self.assertEqual(
            [cell.cell_id for cell in self.cells],
            [
                "jpeg_q90",
                "jpeg_q70",
                "jpeg_q50",
                "jpeg_q30",
                "blur_sigma_0.5",
                "blur_sigma_1.0",
                "blur_sigma_2.0",
                "resize_scale_0.5",
                "resize_scale_0.25",
                "noise_sigma_0.02",
                "noise_sigma_0.05",
                "noise_sigma_0.1",
                "color_jitter_0.2",
                "center_crop_0.8",
            ],
        )
        self.assertEqual(len({cell.cell_id for cell in self.cells}), 14)
        self.assertEqual(
            [cell.output_format for cell in self.cells],
            ["JPEG"] * 4 + ["PNG"] * 10,
        )
        self.assertEqual(
            {cell.cell_id for cell in self.cells if cell.stochastic},
            {
                "noise_sigma_0.02",
                "noise_sigma_0.05",
                "noise_sigma_0.1",
                "color_jitter_0.2",
            },
        )

    def test_seed_is_stable_sha256_prefix(self) -> None:
        self.assertEqual(derive_seed(42, "a", "noise_sigma_0.02"), 17306699420435255152)
        self.assertNotEqual(
            derive_seed(42, "a", "noise_sigma_0.02"),
            derive_seed(42, "b", "noise_sigma_0.02"),
        )

    def test_rejects_transformed_or_non_clean_parent(self) -> None:
        invalid_rows = (
            {"sample_id": "x", "image_view": "benchmark", "transform": "blur"},
            {"sample_id": "x", "image_view": "matched_clean", "transform": "jpeg"},
        )
        for row in invalid_rows:
            with self.subTest(row=row), self.assertRaisesRegex(
                TransformContractError, "matched_clean.*clean"
            ):
                validate_parent_record(row)

    def test_accepts_matched_clean_parent(self) -> None:
        validate_parent_record(
            {"sample_id": "x", "image_view": "matched_clean", "transform": "clean"}
        )

    def test_resize_restores_odd_parent_dimensions(self) -> None:
        result = apply_benchmark(self.gradient((73, 57)), self.resize_half, "a", 42)
        self.assertEqual(result.image.size, (73, 57))
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.realized["intermediate_size"], [37, 29])
        self.assertEqual(result.realized["output_size"], [73, 57])
        self.assertEqual(result.realized["interpolation"], "bilinear")
        self.assertEqual(result.realized["dimension_rounding"], "floor(d * scale + 0.5)")

    def test_center_crop_retains_eighty_percent_per_dimension(self) -> None:
        result = apply_benchmark(self.gradient((73, 57)), self.crop, "a", 42)
        self.assertEqual(result.image.size, (58, 46))
        self.assertEqual(result.realized["crop_bounds"], [7, 5, 65, 51])
        self.assertEqual(result.realized["output_size"], [58, 46])

    def test_noise_is_local_reproducible_and_uses_fixed_rounding(self) -> None:
        source = Image.fromarray(
            np.array(
                [
                    [[0, 127, 255], [10, 128, 240]],
                    [[250, 1, 100], [30, 200, 70]],
                ],
                dtype=np.uint8,
            )
        )
        cell = self.by_id["noise_sigma_0.02"]

        first = apply_benchmark(source, cell, "a", 42)
        repeated = apply_benchmark(source, cell, "a", 42)
        other_sample = apply_benchmark(source, cell, "b", 42)

        self.assertEqual(first.image.tobytes(), repeated.image.tobytes())
        self.assertEqual(first.realized, repeated.realized)
        self.assertNotEqual(first.image.tobytes(), other_sample.image.tobytes())
        self.assertEqual(
            np.asarray(first.image).tolist(),
            [
                [[0, 137, 255], [11, 128, 246]],
                [[250, 12, 91], [25, 202, 74]],
            ],
        )
        self.assertEqual(first.realized["seed"], 17306699420435255152)
        self.assertEqual(first.realized["rounding"], "floor(value + 0.5)")
        self.assertEqual(first.realized["noise_space"], "normalized_float_rgb")

    def test_color_jitter_is_local_reproducible_and_applied_in_fixed_order(self) -> None:
        source = self.gradient((19, 17))
        cell = self.by_id["color_jitter_0.2"]

        first = apply_benchmark(source, cell, "a", 42)
        repeated = apply_benchmark(source, cell, "a", 42)
        other_sample = apply_benchmark(source, cell, "b", 42)

        self.assertEqual(first.image.tobytes(), repeated.image.tobytes())
        self.assertEqual(first.realized, repeated.realized)
        self.assertNotEqual(first.image.tobytes(), other_sample.image.tobytes())
        self.assertEqual(first.realized["order"], ["brightness", "contrast", "saturation"])
        for name in first.realized["order"]:
            self.assertGreaterEqual(first.realized[name], 0.8)
            self.assertLessEqual(first.realized[name], 1.2)

        expected = ImageEnhance.Brightness(source.convert("RGB")).enhance(
            first.realized["brightness"]
        )
        expected = ImageEnhance.Contrast(expected).enhance(first.realized["contrast"])
        expected = ImageEnhance.Color(expected).enhance(first.realized["saturation"])
        self.assertEqual(first.image.tobytes(), expected.tobytes())

    def test_jpeg_round_trip_is_rgb_and_records_four_four_four_settings(self) -> None:
        source = self.gradient((8, 8))
        result = apply_benchmark(source, self.by_id["jpeg_q70"], "a", 42)

        expected_stream = io.BytesIO()
        source.convert("RGB").save(
            expected_stream,
            format="JPEG",
            quality=70,
            subsampling=0,
            optimize=False,
            progressive=False,
            exif=b"",
        )

        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.image.size, source.size)
        self.assertEqual(result.encoded_bytes, expected_stream.getvalue())
        self.assertEqual(
            hashlib.sha256(result.image.tobytes()).hexdigest(),
            "e3d9bb013a66e09b93bd0ae917d4c0f7c6b56671ab03120f6e4aed758061b9ec",
        )
        self.assertEqual(
            result.realized,
            {
                "quality": 70,
                "subsampling": "4:4:4",
                "optimize": False,
                "progressive": False,
                "exif": "",
                "output_format": "JPEG",
            },
        )

    def test_blur_and_alpha_inputs_produce_rgb(self) -> None:
        source = Image.new("RGBA", (9, 7), (10, 20, 30, 128))
        result = apply_benchmark(source, self.by_id["blur_sigma_1.0"], "a", 42)
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.image.size, source.size)
        self.assertEqual(result.realized, {"sigma": 1.0, "output_format": "PNG"})
        self.assertIsNone(result.encoded_bytes)

    def test_unknown_transform_fails_closed(self) -> None:
        cell = TransformCell("rotate", 90, "rotate_90", "PNG")
        with self.assertRaisesRegex(TransformContractError, "rotate"):
            apply_benchmark(self.gradient((5, 5)), cell, "a", 42)


if __name__ == "__main__":
    unittest.main()
