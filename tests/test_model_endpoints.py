import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
from src.api.main import app

client = TestClient(app)

# Test data
VALID_API_KEY = "CRED$#bil1@"

def test_retrain_model_success():
    """Test successful model retraining"""
    response = client.post(
        "/api/v1/models/retrain",
        params={"model_type": "risk", "force": True},
        headers={"X-API-Key": VALID_API_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "accuracy" in data
    assert "precision" in data
    assert "recall" in data
    assert "f1_score" in data
    assert "last_updated" in data
    assert "training_data_size" in data
    assert "performance_metrics" in data

def test_retrain_model_invalid_type():
    """Test model retraining with invalid model type"""
    response = client.post(
        "/api/v1/models/retrain",
        params={"model_type": "invalid", "force": True},
        headers={"X-API-Key": VALID_API_KEY}
    )
    assert response.status_code == 400
    assert "Invalid model type" in response.json()["detail"]

def test_retrain_model_recent_training():
    """Test model retraining when recently trained"""
    response = client.post(
        "/api/v1/models/retrain",
        params={"model_type": "risk", "force": False},
        headers={"X-API-Key": VALID_API_KEY}
    )
    assert response.status_code == 409
    assert "Model was trained recently" in response.json()["detail"]

def test_get_model_metrics_success():
    """Test successful model metrics retrieval"""
    response = client.get(
        "/api/v1/models/metrics",
        params={"model_type": "risk", "days": 7},
        headers={"X-API-Key": VALID_API_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if data:  # If metrics exist
        assert "model_name" in data[0]
        assert "accuracy" in data[0]
        assert "precision" in data[0]
        assert "recall" in data[0]
        assert "f1_score" in data[0]
        assert "last_updated" in data[0]
        assert "training_data_size" in data[0]
        assert "performance_metrics" in data[0]

def test_get_model_metrics_invalid_type():
    """Test model metrics retrieval with invalid model type"""
    response = client.get(
        "/api/v1/models/metrics",
        params={"model_type": "invalid", "days": 7},
        headers={"X-API-Key": VALID_API_KEY}
    )
    assert response.status_code == 400
    assert "model_type must be either 'risk' or 'fraud'" in response.json()["detail"]

def test_get_model_metrics_invalid_days():
    """Test model metrics retrieval with invalid days parameter"""
    response = client.get(
        "/api/v1/models/metrics",
        params={"model_type": "risk", "days": 0},
        headers={"X-API-Key": VALID_API_KEY}
    )
    assert response.status_code == 422
    assert "ge=1" in response.json()["detail"]  # Check for greater than or equal to 1 validation

def test_get_model_metrics_missing_api_key():
    """Test model metrics retrieval without API key"""
    response = client.get(
        "/api/v1/models/metrics",
        params={"model_type": "risk", "days": 7}
    )
    assert response.status_code == 401
    assert "Invalid or missing API key" in response.json()["detail"]

def test_get_model_metrics_invalid_api_key():
    """Test model metrics retrieval with invalid API key"""
    response = client.get(
        "/api/v1/models/metrics",
        params={"model_type": "risk", "days": 7},
        headers={"X-API-Key": "invalid_key"}
    )
    assert response.status_code == 401
    assert "Invalid or missing API key" in response.json()["detail"] 