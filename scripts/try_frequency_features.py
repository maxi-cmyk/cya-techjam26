"""Manual sanity-check script for the Task 7 frequency extractor.

Not part of the automated test suite - run it directly to see the feature
values on your own images and build intuition for what they mean:

    python scripts/try_frequency_features.py path/to/image.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

from cya_detector.features.common import load_image_array
from cya_detector.features.frequency import extract_frequency_features


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/try_frequency_features.py <image_path>")
        raise SystemExit(1)

    image = load_image_array(Path(sys.argv[1]))
    result = extract_frequency_features(image)

    print(f"valid: {result.valid}")
    print(f"confidence: {result.confidence:.3f}")
    if result.notes:
        print(f"notes: {result.notes}")
    for key, value in result.values.items():
        print(f"  {key}: {value:.5f}")


if __name__ == "__main__":
    main()
