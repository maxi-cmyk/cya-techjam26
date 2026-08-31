from __future__ import annotations

import io
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from backend.app import app, get_predictor


def _fake_predictor(image: Image.Image) -> float:
    return 0.9 if image.getpixel((0, 0)) == (200, 0, 0) else 0.1


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buf, format="PNG")
    return buf.getvalue()


class BackendAppTests(unittest.TestCase):
    """The break caught here is the API silently diverging from the CLI's
    discovery/validation/prediction contract, or failing a whole batch upload
    because of one bad file."""

    def setUp(self) -> None:
        app.dependency_overrides[get_predictor] = lambda: _fake_predictor
        self.addCleanup(app.dependency_overrides.clear)
        self.client = TestClient(app)

    def test_health_reports_ok(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_predict_with_no_files_is_a_usage_error(self) -> None:
        response = self.client.post("/predict", files=[])

        self.assertEqual(response.status_code, 422)

    def test_predict_returns_labels_and_confidence_for_each_upload(self) -> None:
        response = self.client.post(
            "/predict",
            files=[
                ("files", ("ai.png", _png_bytes((200, 0, 0)), "image/png")),
                ("files", ("real.png", _png_bytes((10, 10, 10)), "image/png")),
            ],
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["errors"], [])
        predictions = {row["filename"]: row for row in body["predictions"]}
        self.assertEqual(predictions["ai.png"]["label"], "ai_generated")
        self.assertEqual(predictions["ai.png"]["confidence"], 0.9)
        self.assertEqual(predictions["real.png"]["label"], "authentic")
        self.assertEqual(predictions["real.png"]["confidence"], 0.9)

    def test_predict_reports_per_file_validation_errors_without_failing_the_whole_batch(
        self,
    ) -> None:
        response = self.client.post(
            "/predict",
            files=[
                ("files", ("ai.png", _png_bytes((200, 0, 0)), "image/png")),
                ("files", ("bad.png", b"not an image", "image/png")),
            ],
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body["predictions"]), 1)
        self.assertEqual(body["predictions"][0]["filename"], "ai.png")
        self.assertEqual(len(body["errors"]), 1)
        self.assertEqual(body["errors"][0]["filename"], "bad.png")


if __name__ == "__main__":
    unittest.main()
