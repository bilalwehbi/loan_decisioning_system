import requests
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api_test.log')
    ]
)

logger = logging.getLogger(__name__)

# API endpoint
BASE_URL = "http://localhost:8000"
ASSESS_ENDPOINT = f"{BASE_URL}/api/v1/assess"

def test_health_check():
    print("\nTesting health check endpoint...")
    response = requests.get("http://localhost:8000/api/v1/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

def test_loan_assessment():
    """Test the loan assessment endpoint with valid data"""
    
    # Test data
    test_data = {
        "credit_data": {
            "credit_score": 720,
            "credit_history_length": 5,
            "payment_history": [1, 1, 1, 1, 1],  # 1 = on time, 0 = late
            "total_debt": 15000,
            "debt_to_income_ratio": 0.35
        },
        "banking_data": {
            "account_balance": 5000,
            "transaction_history": [
                {"date": "2024-01-01", "amount": 1000, "type": "deposit"},
                {"date": "2024-01-15", "amount": -500, "type": "withdrawal"}
            ],
            "average_monthly_balance": 4500,
            "overdraft_count": 0
        },
        "application_data": {
            "income": 60000,
            "employment_status": "employed",
            "employment_length": 3,
            "loan_amount": 25000,
            "loan_term": 36,
            "purpose": "debt_consolidation"
        }
    }
    
    try:
        # Make request to the endpoint
        logger.info("Sending loan assessment request...")
        response = requests.post(ASSESS_ENDPOINT, json=test_data)
        
        # Log response status
        logger.info(f"Response status code: {response.status_code}")
        
        # Check if request was successful
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        # Log key information
        logger.info("Loan assessment successful!")
        logger.info(f"Application ID: {result.get('application_id')}")
        logger.info(f"Final Decision: {result.get('final_decision')}")
        logger.info(f"Credit Risk Score: {result.get('credit_risk', {}).get('score')}")
        logger.info(f"Fraud Score: {result.get('fraud_detection', {}).get('score')}")
        
        # Validate response structure
        required_fields = [
            'application_id',
            'timestamp',
            'processing_time_ms',
            'credit_risk',
            'fraud_detection',
            'final_decision'
        ]
        
        for field in required_fields:
            if field not in result:
                logger.error(f"Missing required field in response: {field}")
                return False
        
        # Validate credit risk data
        credit_risk = result.get('credit_risk', {})
        if not all(key in credit_risk for key in ['score', 'probability_of_default', 'decision', 'top_factors']):
            logger.error("Missing required fields in credit_risk data")
            return False
        
        # Validate fraud detection data
        fraud_detection = result.get('fraud_detection', {})
        if not all(key in fraud_detection for key in ['score', 'flags']):
            logger.error("Missing required fields in fraud_detection data")
            return False
        
        logger.info("All response validations passed!")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        return False
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse response JSON: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False

def test_invalid_data():
    """Test the loan assessment endpoint with invalid data"""
    
    # Test cases with missing required fields
    test_cases = [
        {
            "name": "Missing credit_data",
            "data": {
                "banking_data": {
                    "account_balance": 5000,
                    "transaction_history": []
                },
                "application_data": {
                    "income": 60000,
                    "employment_status": "employed",
                    "loan_amount": 25000,
                    "loan_term": 36
                }
            }
        },
        {
            "name": "Missing required field in credit_data",
            "data": {
                "credit_data": {
                    "credit_score": 720,
                    "credit_history_length": 5
                    # Missing payment_history
                },
                "banking_data": {
                    "account_balance": 5000,
                    "transaction_history": []
                },
                "application_data": {
                    "income": 60000,
                    "employment_status": "employed",
                    "loan_amount": 25000,
                    "loan_term": 36
                }
            }
        }
    ]
    
    for test_case in test_cases:
        try:
            logger.info(f"\nTesting invalid data case: {test_case['name']}")
            response = requests.post(ASSESS_ENDPOINT, json=test_case['data'])
            
            # Should return 422 Unprocessable Entity
            if response.status_code != 422:
                logger.error(f"Expected status code 422, got {response.status_code}")
                continue
                
            error_data = response.json()
            logger.info(f"Received expected error: {error_data.get('detail')}")
            
        except Exception as e:
            logger.error(f"Error testing invalid data: {str(e)}")

def test_model_metrics():
    print("\nTesting model metrics endpoint...")
    response = requests.get("http://localhost:8000/api/v1/models/metrics")
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception:
        print(f"Raw Response: {response.text}")

def test_retrain_models():
    print("\nTesting model retrain endpoint...")
    response = requests.post(
        "http://localhost:8000/api/v1/models/retrain",
        params={"model_type": "risk", "force": True}
    )
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception:
        print(f"Raw Response: {response.text}")

def test_application_history():
    print("\nTesting application history endpoint...")
    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    response = requests.get(
        "http://localhost:8000/api/v1/applications/history",
        params={
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "limit": 5
        }
    )
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception:
        print(f"Raw Response: {response.text}")

if __name__ == "__main__":
    logger.info("Starting API tests...")
    
    # Test valid data
    logger.info("\nTesting with valid data...")
    valid_test_result = test_loan_assessment()
    logger.info(f"Valid data test {'passed' if valid_test_result else 'failed'}")
    
    # Test invalid data
    logger.info("\nTesting with invalid data...")
    test_invalid_data()
    
    logger.info("\nAPI tests completed!") 