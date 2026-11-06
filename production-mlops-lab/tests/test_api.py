from __future__ import annotations

from fastapi.testclient import TestClient

from heart_disease_mlops import api


class FakeRuntime:
    def predict(self, request):
        return 1, 0.75


def test_health_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(api, "runtime", FakeRuntime())

    with TestClient(api.app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(api, "runtime", FakeRuntime())

    with TestClient(api.app) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_prediction_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(api, "runtime", FakeRuntime())

    payload = {
        "age": 63,
        "trestbps": 145,
        "chol": 233,
        "thalach": 150,
        "oldpeak": 2.3,
        "sex": 1,
        "cp": 1,
        "fbs": 1,
        "restecg": 2,
        "exang": 0,
        "slope": 3,
        "ca": 0,
        "thal": 6,
    }

    with TestClient(api.app) as client:
        response = client.post("/v1/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["prediction"] == 1
    assert response.json()["probability"] == 0.75


def test_unknown_fields_are_rejected(monkeypatch) -> None:
    monkeypatch.setattr(api, "runtime", FakeRuntime())

    payload = {
        "age": 63,
        "trestbps": 145,
        "chol": 233,
        "thalach": 150,
        "oldpeak": 2.3,
        "sex": 1,
        "cp": 1,
        "fbs": 1,
        "restecg": 2,
        "exang": 0,
        "slope": 3,
        "ca": 0,
        "thal": 6,
        "extra": 1,
    }

    with TestClient(api.app) as client:
        response = client.post("/v1/predict", json=payload)

    assert response.status_code == 422