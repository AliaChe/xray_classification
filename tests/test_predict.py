from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
from pathlib import Path
from src.predict import load_image

client = TestClient(app)

@patch("app.main.predict_image")
def test_predict_success(mock_predict):
    mock_predict.return_value = {
        "prediction": "NORMAL",
        "confidence": 0.98,
        "raw_score": 0.02,
    }

    with open("tests/data/test_image.jpeg", "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("test.jpeg", f, "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.json() == {
        "prediction": "NORMAL",
        "confidence": 0.98,
        "raw_score": 0.02,
    }


def test_predict_rejects_non_image():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "File must be an image."


def test_load_image_returns_batch():
    image_path = Path("tests/data/test_image.jpeg")

    img_array = load_image(image_path, image_size=224)

    assert img_array.shape == (1, 224, 224, 3)