import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def client(monkeypatch):
    # Read by resolve_and_download_champion() during the app's lifespan startup (triggered below
    # by the `with TestClient(...)` block) - short-circuits to a random-init model, no network,
    # no real trained weight needed. This is what lets this test (and CI's docker-smoke job) run
    # without CarDD's license-restricted data or a hosted weight ever being involved.
    monkeypatch.setenv("CARDD_MODEL_SOURCE", "random-init")
    from cardd.serving.app import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["source"] == "https://github.com/JShi12/vehicle-damage-detection"


def test_predict_returns_well_formed_response(client):
    image = Image.new("RGB", (64, 64), color=(120, 50, 200))
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)

    resp = client.post("/predict", files={"file": ("test.jpg", buf, "image/jpeg")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["image_width"] == 64
    assert body["image_height"] == 64
    assert isinstance(body["detections"], list)
    assert body["inference_ms"] >= 0


def test_predict_rejects_non_image_upload(client):
    files = {"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 400
