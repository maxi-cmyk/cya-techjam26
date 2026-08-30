from __future__ import annotations

import random
import unittest

from PIL import Image

from cya_detector.transforms.preprocessing import (
    center_crop_input,
    pad_to_minimum,
    random_crop_input,
    to_rgb,
)


class PreprocessingTests(unittest.TestCase):
    @staticmethod
    def gradient(width: int = 400, height: int = 380) -> Image.Image:
        image = Image.new("RGB", (width, height))
        pixels = image.load()
        for y in range(height):
            for x in range(width):
                pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
        return image

    def test_small_image_is_padded_without_resizing(self) -> None:
        result = center_crop_input(Image.new("L", (333, 331), 255), size=336)

        self.assertEqual(result.size, (336, 336))
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(result.getpixel((1, 2)), (255, 255, 255))
        self.assertEqual(result.getpixel((333, 332)), (255, 255, 255))
        self.assertEqual(result.getpixel((334, 333)), (0, 0, 0))

    def test_odd_padding_remainder_is_on_right_and_bottom(self) -> None:
        source = Image.new("RGB", (2, 2), (10, 20, 30))

        result = pad_to_minimum(source, size=5)

        self.assertEqual(result.size, (5, 5))
        self.assertEqual(result.getpixel((1, 1)), (10, 20, 30))
        self.assertEqual(result.getpixel((2, 2)), (10, 20, 30))
        self.assertEqual(result.getpixel((3, 3)), (0, 0, 0))
        self.assertEqual(result.getpixel((4, 4)), (0, 0, 0))

    def test_rgba_is_composited_on_stable_black_background(self) -> None:
        source = Image.new("RGBA", (2, 1))
        source.putpixel((0, 0), (255, 0, 0, 128))
        source.putpixel((1, 0), (255, 0, 0, 0))

        result = to_rgb(source)

        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.getpixel((0, 0)), (128, 0, 0))
        self.assertEqual(result.getpixel((1, 0)), (0, 0, 0))

    def test_center_crop_is_deterministic_and_does_not_upscale(self) -> None:
        source = self.gradient(5, 7)

        result = center_crop_input(source, size=3)

        self.assertEqual(result.size, (3, 3))
        self.assertEqual(result.getpixel((0, 0)), source.getpixel((1, 2)))
        self.assertEqual(result.getpixel((2, 2)), source.getpixel((3, 4)))

    def test_seeded_random_crop_repeats(self) -> None:
        first = random_crop_input(self.gradient(), 336, seed=123)
        second = random_crop_input(self.gradient(), 336, seed=123)

        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(first.size, (336, 336))

    def test_random_crop_uses_local_seeded_generator(self) -> None:
        source = self.gradient()
        random.seed(7)
        state_before = random.getstate()

        random_crop_input(source, 336, seed=123)

        self.assertEqual(random.getstate(), state_before)

    def test_nonpositive_size_is_rejected(self) -> None:
        for function in (pad_to_minimum, center_crop_input):
            with self.subTest(function=function.__name__), self.assertRaises(ValueError):
                function(Image.new("RGB", (2, 2)), size=0)

        with self.assertRaises(ValueError):
            random_crop_input(Image.new("RGB", (2, 2)), size=-1, seed=1)


if __name__ == "__main__":
    unittest.main()
