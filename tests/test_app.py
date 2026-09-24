"""
Tests for Flask Web Application routes and API endpoints.
"""
import os
import sys
import io
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"FloraScan" in response.data
    assert b"Intelligent Plant Disease Detection" in response.data

def test_predict_no_file(client):
    response = client.post("/predict", data={}, follow_redirects=True)
    assert response.status_code == 200
    assert b"No image file provided" in response.data

def test_predict_empty_filename(client):
    data = {"file": (io.BytesIO(b""), "")}
    response = client.post("/predict", data=data, follow_redirects=True)
    assert response.status_code == 200
    assert b"No file was selected" in response.data

def test_api_predict_no_file(client):
    response = client.post("/api/predict", data={})
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["status"] == "error"
