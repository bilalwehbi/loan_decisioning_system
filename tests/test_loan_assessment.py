import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
from src.api.main import app
from src.api.models import LoanApplication, CreditData, BankingData, BehavioralData, EmploymentData, ApplicationData

client = TestClient(app)

# Test data
VALID_API_KEY = "CRED$#bil1@"
VALID_APPLICATION = {
    "loan_amount": 50000,
    "loan_term_months": 36,
    "loan_purpose": "home_improvement",
    "credit_data": {
        "credit_score": 750,
        "delinquencies": 0,
        "inquiries_last_6m": 2,
        "tradelines": 5,
        "utilization": 0.3,
        "payment_history_score": 0.95,
        "credit_age_months": 60,
        "credit_mix_score": 0.8,
        "num_accounts": 5,
        "num_active_accounts": 4,
        "credit_utilization": 0.3,
        "num_delinquencies_30d": 0,
        "num_delinquencies_60d": 0,
        "num_delinquencies_90d": 0
    },
    "banking_data": {
        "avg_monthly_income": 8000,
        "income_stability_score": 0.9,
        "spending_pattern_score": 0.85,
        "transaction_count": 100,
        "avg_account_balance": 5000,
        "overdraft_frequency": 0,
        "savings_rate": 0.2
    },
    "behavioral_data": {
        "application_completion_time": 600,
        "device_trust_score": 0.95,
        "location_risk_score": 0.1,
        "digital_footprint_score": 0.9
    },
    "employment_data": {
        "employment_length_months": 36,
        "employment_verification_score": 0.95,
        "income_verification_score": 0.9
    },
    "application_data": {
        "application_timestamp": datetime.now().isoformat(),
        "device_fingerprint": "test_fingerprint",
        "browser_fingerprint": "test_browser",
        "device_id": "test_device",
        "email_domain": "example.com",
        "phone_number": "1234567890",
        "network_info": {
            "proxy_score": 0,
            "vpn_score": 0,
            "tor_score": 0
        }
    },
    "applicant_income": 96000,
    "applicant_employment_length_years": 3
}

def test_assess_loan_application_success():
    """Test successful loan application assessment"""
    response = client.post(
        "/api/v1/assess",
        json=VALID_APPLICATION,
        headers={"X-API-Key": VALID_API_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    assert "application_id" in data
    assert "risk_assessment" in data
    assert "fraud_assessment" in data
    assert "final_decision" in data
    assert "risk_score" in data
    assert "fraud_score" in data
    assert "fraud_flags" in data
    assert "explanations" in data

def test_assess_loan_application_invalid_amount():
    """Test loan application with invalid amount"""
    invalid_application = VALID_APPLICATION.copy()
    invalid_application["loan_amount"] = -1000
    response = client.post(
        "/api/v1/assess",
        json=invalid_application,
        headers={"X-API-Key": VALID_API_KEY}
    )
    assert response.status_code == 422
    assert "Loan amount must be positive" in response.json()["detail"]

def test_assess_loan_application_invalid_term():
    """Test loan application with invalid term"""
    invalid_application = VALID_APPLICATION.copy()
    invalid_application["loan_term_months"] = 6
    response = client.post(
        "/api/v1/assess",
        json=invalid_application,
        headers={"X-API-Key": VALID_API_KEY}
    )
    assert response.status_code == 422
    assert "Loan term must be between 12 and 84 months" in response.json()["detail"]

def test_assess_loan_application_invalid_credit_score():
    """Test loan application with invalid credit score"""
    invalid_application = VALID_APPLICATION.copy()
    invalid_application["credit_data"]["credit_score"] = 200
    response = client.post(
        "/api/v1/assess",
        json=invalid_application,
        headers={"X-API-Key": VALID_API_KEY}
    )
    assert response.status_code == 422
    assert "Credit score must be between 300 and 850" in response.json()["detail"]

def test_assess_loan_application_missing_api_key():
    """Test loan application without API key"""
    response = client.post(
        "/api/v1/assess",
        json=VALID_APPLICATION
    )
    assert response.status_code == 401
    assert "Invalid or missing API key" in response.json()["detail"]

def test_assess_loan_application_invalid_api_key():
    """Test loan application with invalid API key"""
    response = client.post(
        "/api/v1/assess",
        json=VALID_APPLICATION,
        headers={"X-API-Key": "invalid_key"}
    )
    assert response.status_code == 401
    assert "Invalid or missing API key" in response.json()["detail"] 