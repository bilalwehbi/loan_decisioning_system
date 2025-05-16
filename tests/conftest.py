import pytest
import os
import shutil
from pathlib import Path

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment before running tests."""
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/explanations", exist_ok=True)
    
    # Create empty model files if they don't exist
    Path("models/risk_model.joblib").touch()
    Path("models/fraud_model.joblib").touch()
    
    yield
    
    # Clean up test data after tests
    if os.path.exists("data/experiments.json"):
        os.remove("data/experiments.json")
    if os.path.exists("data/metrics_history.json"):
        os.remove("data/metrics_history.json")
    if os.path.exists("data/application_history.json"):
        os.remove("data/application_history.json")
    
    # Clean up explanations directory
    if os.path.exists("data/explanations"):
        shutil.rmtree("data/explanations") 