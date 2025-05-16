# src/models/test_integrated_service.py
import os
import sys
import logging
import json
import pandas as pd
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.loan_decisioning_service import LoanDecisioningService
from src.models.test_fraud_detection_inference import create_sample_applications
from src.configs.config import PERFORMANCE_TARGETS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('integrated_service_test.log')
    ]
)

logger = logging.getLogger(__name__)

def test_integrated_service():
    """Test the integrated loan decisioning service"""
    # Initialize service
    service = LoanDecisioningService()
    
    try:
        # Load models
        service.load_models()
        logger.info("Successfully loaded all models")
    except FileNotFoundError as e:
        logger.error(f"Error loading models: {str(e)}")
        logger.error("Please train both credit risk and fraud detection models first")
        return
    
    # Create sample applications
    applications = create_sample_applications(num_samples=5)
    
    logger.info(f"Testing with {len(applications)} sample applications")
    
    # Process each application
    for i, (application, actual) in enumerate(applications, 1):
        logger.info(f"Processing application {i}")
        logger.info(f"Loan details: Amount=${application['loan_amount']:.2f}, Purpose={application['loan_purpose']}")
        
        # Make decision
        decision = service.make_decision(application)
        
        # Log results
        logger.info(f"Decision: {decision['decision']}")
        logger.info(f"Credit score: {decision['credit_score']}, Default probability: {decision['probability_of_default']:.4f}")
        logger.info(f"Fraud score: {decision['fraud_score']:.4f}, Fraud flags: {', '.join(decision['fraud_flags']) if decision['fraud_flags'] else 'None'}")
        logger.info(f"Processing time: {decision['processing_time']:.3f} seconds")
        
        # Check if meets latency requirement
        if decision['processing_time'] > PERFORMANCE_TARGETS['latency']:
            logger.warning(f"Processing time exceeds target latency of {PERFORMANCE_TARGETS['latency']} seconds")
        
        logger.info("---")

if __name__ == "__main__":
    logger.info("Testing integrated loan decisioning service")
    test_integrated_service()
    logger.info("Testing completed")