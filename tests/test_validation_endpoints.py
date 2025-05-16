import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
from src.api.main import app
import io

client = TestClient(app)

# Test data
VALID_API_KEY = "CRED$#bil1@"

def test_validate_paystub_success():
    """Test successful paystub validation"""
    # Create a test image file
    test_image = io.BytesIO(b"fake image data")
    test_image.name = "test.jpg"
    
    response = client.post(
        "/api/v1/validate/paystub",
        files={"paystub_image": ("test.jpg", test_image, "image/jpeg")},
        data={"application_id": "test_app_123"},
        headers={"X-API-Key": VALID_API_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    assert "is_valid" in data
    assert "confidence_score" in data
    assert "extracted_data" in data
    assert "validation_details" in data

def test_validate_paystub_invalid_file():
    """Test paystub validation with invalid file"""
    # Create an invalid file
    test_file = io.BytesIO(b"not an image")
    test_file.name = "test.txt"
    
    response = client.post(
        "/api/v1/validate/paystub",
        files={"paystub_image": ("test.txt", test_file, "text/plain")},
        data={"application_id": "test_app_123"},
        headers={"X-API-Key": VALID_API_KEY}
    )
    assert response.status_code == 422
    assert "File must be an image" in response.json()["detail"]

def test_validate_paystub_missing_application_id():
    """Test paystub validation without application ID"""
    test_image = io.BytesIO(b"fake image data")
    test_image.name = "test.jpg"
    
    response = client.post(
        "/api/v1/validate/paystub",
        files={"paystub_image": ("test.jpg", test_image, "image/jpeg")},
        headers={"X-API-Key": VALID_API_KEY}
    )
    assert response.status_code == 422
    assert "Application ID is required" in response.json()["detail"]

def test_validate_bank_transactions_success():
    """Test successful bank transaction validation"""
    test_transactions = {
        "application_id": "test_app_123",
        "transactions": [
            {
                "date": "2024-03-01",
                "amount": 5000.00,
                "description": "Salary"
            },
            {
                "date": "2024-03-15",
                "amount": 5000.00,
                "description": "Salary"
            }
        ]
    }
    
    response = client.post(
        "/api/v1/validate/bank-transactions",
        json=test_transactions,
        headers={"X-API-Key": VALID_API_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    assert "is_valid" in data
    assert "stability_score" in data
    assert "income_metrics" in data
    assert "anomalies" in data
    assert "validation_details" in data

def test_validate_bank_transactions_invalid_data():
    """Test bank transaction validation with invalid data"""
    invalid_transactions = {
        "application_id": "test_app_123",
        "transactions": [
            {
                "date": "2024-03-01",
                "amount": "invalid",  # Invalid amount type
                "description": "Salary"
            }
        ]
    }
    
    response = client.post(
        "/api/v1/validate/bank-transactions",
        json=invalid_transactions,
        headers={"X-API-Key": VALID_API_KEY}
    )
    assert response.status_code == 422

def test_validate_bank_transactions_missing_fields():
    """Test bank transaction validation with missing fields"""
    invalid_transactions = {
        "application_id": "test_app_123",
        "transactions": [
            {
                "date": "2024-03-01",
                # Missing amount and description
            }
        ]
    }
    
    response = client.post(
        "/api/v1/validate/bank-transactions",
        json=invalid_transactions,
        headers={"X-API-Key": VALID_API_KEY}
    )
    assert response.status_code == 422
    assert "Missing required field" in response.json()["detail"]

def test_validate_employment_success():
    """Test successful employment validation"""
    test_employment = {
        "application_id": "test_app_123",
        "employer": "Test Company",
        "start_date": "2021-01-01",
        "job_title": "Software Engineer"
    }
    
    response = client.post(
        "/api/v1/validate/employment",
        json=test_employment,
        headers={"X-API-Key": VALID_API_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    assert "is_valid" in data
    assert "validation_score" in data
    assert "employer_validation" in data
    assert "duration_validation" in data
    assert "role_validation" in data

def test_validate_employment_invalid_dates():
    """Test employment validation with invalid dates"""
    invalid_employment = {
        "application_id": "test_app_123",
        "employer": "Test Company",
        "start_date": "2025-01-01",  # Future date
        "job_title": "Software Engineer"
    }
    
    response = client.post(
        "/api/v1/validate/employment",
        json=invalid_employment,
        headers={"X-API-Key": VALID_API_KEY}
    )
    assert response.status_code == 422
    assert "Start date cannot be in the future" in response.json()["detail"]

def test_validate_employment_invalid_end_date():
    """Test employment validation with invalid end date"""
    invalid_employment = {
        "application_id": "test_app_123",
        "employer": "Test Company",
        "start_date": "2021-01-01",
        "end_date": "2020-01-01",  # End date before start date
        "job_title": "Software Engineer"
    }
    
    response = client.post(
        "/api/v1/validate/employment",
        json=invalid_employment,
        headers={"X-API-Key": VALID_API_KEY}
    )
    assert response.status_code == 422
    assert "End date must be after start date" in response.json()["detail"]

def test_validate_employment_invalid_employer():
    """Test employment validation with invalid employer"""
    invalid_employment = {
        "application_id": "test_app_123",
        "employer": "A",  # Too short
        "start_date": "2021-01-01",
        "job_title": "Software Engineer"
    }
    
    response = client.post(
        "/api/v1/validate/employment",
        json=invalid_employment,
        headers={"X-API-Key": VALID_API_KEY}
    )
    assert response.status_code == 422
    assert "Valid employer name is required" in response.json()["detail"] 