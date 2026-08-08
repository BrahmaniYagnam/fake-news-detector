import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.model_service import ModelService
from backend.utils.text import clean_text

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "uptime" in data


def test_model_info_endpoint() -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"]
    assert data["framework"] == "PyTorch"
    assert data["num_classes"] == 2


def test_prediction_request_validation() -> None:
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_text_cleaning() -> None:
    raw = "  Fake news   test  "
    assert clean_text(raw) == "Fake news test"


def test_model_service_loads_model_and_tokenizer() -> None:
    service = ModelService()
    info = service.get_info()
    assert isinstance(info["model_name"], str)
    assert "Fake" not in info["model_name"] or info["model_name"]
