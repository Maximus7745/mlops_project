import pytest
from fastapi.testclient import TestClient
from fastapi_app import app

client = TestClient(app)


def test_predict_happy():
    response = client.post("/predict/", json={"text": "I am so happy today!"})
    assert response.status_code == 200
    assert response.json() == {
        "text": "I am so happy today!",
        "predicted_emotion": "joy",
    }


def test_predict_sad():
    response = client.post("/predict/", json={"text": "I am feeling very sad."})
    assert response.status_code == 200
    assert response.json() == {
        "text": "I am feeling very sad.",
        "predicted_emotion": "sadness",
    }


def test_predict_empty():
    response = client.post("/predict/", json={"text": ""})
    assert response.status_code == 200
    assert response.json()["predicted_emotion"]


def test_invalid_request():
    response = client.post("/predict/", json={})
    assert response.status_code == 422
