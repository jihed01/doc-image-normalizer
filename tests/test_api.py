import base64
import io

import cv2
import numpy as np
from fastapi.testclient import TestClient

from doc_normalizer.api.main import app

client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_normalize_endpoint_success() -> None:
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    success, encoded_image = cv2.imencode(".png", image)

    assert success

    image_bytes = io.BytesIO(encoded_image.tobytes())
    response = client.post(
        "/v1/normalize",
        files={"file": ("test.png", image_bytes, "image/png")},
    )

    assert response.status_code == 200

    data = response.json()
    assert "image_base64" in data
    assert "detected_angle" in data
    assert "processing_time_ms" in data
    assert data["processing_time_ms"] > 0

    decoded_image = base64.b64decode(data["image_base64"])
    assert (
        cv2.imdecode(
            np.frombuffer(decoded_image, dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        is not None
    )


def test_normalize_endpoint_rejects_invalid_image() -> None:
    response = client.post(
        "/v1/normalize",
        files={"file": ("invalid.png", b"not an image", "image/png")},
    )

    assert response.status_code == 400


def test_normalize_invalid_file_type():
    file_bytes = io.BytesIO(b"ceci est un fichier texte")
    response = client.post(
        "/v1/normalize",
        files={"file": ("test.txt", file_bytes, "text/plain")},
    )

    assert response.status_code == 415
    assert "Format non supporte" in response.json()["detail"]


def test_normalize_file_too_large():
    large_data = b"0" * (11 * 1024 * 1024)
    file_bytes = io.BytesIO(large_data)
    response = client.post(
        "/v1/normalize",
        files={"file": ("large_image.png", file_bytes, "image/png")},
    )

    assert response.status_code == 413
    assert "depasse la limite" in response.json()["detail"]


def test_normalize_rejects_corrupted_image():
    file_bytes = io.BytesIO(b"this is not a valid image")

    response = client.post(
        "/v1/normalize",
        files={"file": ("corrupted.png", file_bytes, "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid or unsupported image file."
