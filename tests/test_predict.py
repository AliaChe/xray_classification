from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_predict_rejects_non_image():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "File must be an image."