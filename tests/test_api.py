"""API tests: /health and /predict contract."""
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

SAMPLE_CAR = {
    "brand": "Toyota",
    "model": "Camry SE",
    "model_year": 2019,
    "milage": 45000,
    "fuel_type": "gasoline",
    "engine": "2.5L 4-Cylinder",
    "transmission": "Automatic",
    "ext_col": "Black",
    "int_col": "Black",
    "accident": "none reported",
    "clean_title": "yes",
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["label"] == "price"
    assert isinstance(body["features"], list) and len(body["features"]) == 11


def test_predict_schema():
    r = client.post("/predict", json=SAMPLE_CAR)
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["price"], (int, float)) and body["price"] > 0
    assert body["interval_low"] <= body["price"] <= body["interval_high"]
    assert body["interval_low"] >= 0
    factors = body["top_factors"]
    assert isinstance(factors, list) and len(factors) == 5
    for f in factors:
        assert set(f.keys()) == {"feature", "value", "importance"}
        assert isinstance(f["importance"], (int, float))


def test_predict_missing_field_returns_422():
    bad = dict(SAMPLE_CAR)
    del bad["brand"]
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
