import pytest
import requests
import time
from urllib.parse import urljoin
import os
import sys
from src.config import API_KEY

# API Configuration
BASE_URL = "http://localhost:8000"
HEALTH_ENDPOINT = urljoin(BASE_URL, "/api/v1/health")
API_KEY = "CRED$#bil1@"

def wait_for_server(timeout=30):
    """Wait for the server to be ready"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(HEALTH_ENDPOINT, headers={"X-API-Key": API_KEY})
            if response.status_code == 200:
                print("Server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            print("Waiting for server to start...")
            time.sleep(1)
    return False

def main():
    """Main function to run tests"""
    # Wait for server to be ready
    if not wait_for_server():
        print("Server failed to start within timeout period")
        sys.exit(1)
    
    try:
        # Run the tests
        print("\nRunning tests...")
        pytest.main(["-v", "test_api.py"])
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 