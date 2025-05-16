# src/models/test_credit_risk_inference.py
import os
import sys
import logging
import json
import random
import pandas as pd
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.credit_risk_model import CreditRiskModel
from src.configs.config import MODEL_DIR, DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('inference_test.log')
    ]
)

logger = logging.getLogger(__name__)

def load_latest_model():
    """Load the most recently trained model"""
    model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('credit_risk_model_') and f.endswith('.pkl')]
    
    if not model_files:
        raise FileNotFoundError("No trained credit risk models found. Run training first.")
    
    # Get the most recent model
    latest_model = max(model_files)
    model_path = os.path.join(MODEL_DIR, latest_model)
    
    # Load model
    model = CreditRiskModel()
    model.load(model_path)
    
    return model

def create_sample_applications(num_samples=5):
    """Create sample loan applications for testing"""
    # Load a few rows from our dataset to create realistic samples
    df = pd.read_csv(os.path.join(DATA_DIR, "merged_dataset.csv"))
    
    # Select random samples
    samples = df.sample(num_samples)
    
    # Create test applications
    test_applications = []
    
    for _, row in samples.iterrows():
        application = {
            'loan_amount': float(row['loan_amount']),
            'loan_term_months': int(row['loan_term_months']),
            'loan_purpose': row['loan_purpose'],
            'applicant_income': float(row['applicant_income']),
            'applicant_employment_length_years': float(row['applicant_employment_length_years']),
            
            'credit_data': {
                'credit_score': int(row['credit_score']) if pd.notna(row['credit_score']) else None,
                'num_accounts': int(row['num_accounts']),
                'num_active_accounts': int(row['num_active_accounts']),
                'credit_utilization': float(row['credit_utilization']),
                'num_delinquencies_30d': int(row['num_delinquencies_30d']),
                'num_delinquencies_60d': int(row['num_delinquencies_60d']),
                'num_delinquencies_90d': int(row['num_delinquencies_90d']),
                'num_collections': int(row['num_collections']),
                'total_debt': float(row['total_debt']),
                'inquiries_last_6mo': int(row['inquiries_last_6mo']),
                'longest_credit_length_months': int(row['longest_credit_length_months']),
            },
            
            'banking_data': {
                'avg_monthly_deposits': float(row['avg_monthly_deposits']),
                'avg_monthly_withdrawals': float(row['avg_monthly_withdrawals']),
                'monthly_income': float(row['monthly_income']),
                'income_stability_score': float(row['income_stability_score']),
                'num_income_sources': int(row['num_income_sources']),
                'avg_daily_balance': float(row['avg_daily_balance']),
                'num_nsf_transactions': int(row['num_nsf_transactions']),
                'num_overdrafts': int(row['num_overdrafts']),
            },
            
            'application_data': {
                'ip_address': row['ip_address'],
                'device_id': row['device_id'],
                'email_domain': row['email_domain'],
                'application_timestamp': row['application_timestamp'],
                'time_spent_on_application': int(row['time_spent_on_application']),
                'num_previous_applications': int(row['num_previous_applications']),
            }
        }
        
        # Also include the actual outcome for comparison
        actual_outcome = {
            'actual_loan_status': row['loan_status'],
            'actual_default': 1 if row['loan_status'] == 'defaulted' else 0
        }
        
        test_applications.append((application, actual_outcome))
    
    return test_applications

def test_inference():
    """Test model inference with sample applications"""
    # Load model
    try:
        model = load_latest_model()
        logger.info(f"Loaded model {model.model_id}")
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    # Create sample applications
    applications = create_sample_applications(num_samples=5)
    
    logger.info(f"Testing with {len(applications)} sample applications")
    
    # Make predictions
    for i, (application, actual) in enumerate(applications, 1):
        logger.info(f"Sample {i}: Loan Amount: ${application['loan_amount']:.2f}, Purpose: {application['loan_purpose']}")
        
        # Make prediction
        prediction = model.predict(application)
        
        # Log results
        logger.info(f"Prediction: Credit Score: {prediction['credit_score']}, Decision: {prediction['decision']}")
        logger.info(f"Default Probability: {prediction['probability_of_default']:.4f}, Actual Default: {actual['actual_default']}")
        logger.info("---")

if __name__ == "__main__":
    logger.info("Testing credit risk model inference")
    test_inference()
    logger.info("Testing completed")