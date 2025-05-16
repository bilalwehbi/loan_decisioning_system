import os
import sys
import logging
import pandas as pd
import joblib

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from src.models.credit_risk_model import CreditRiskModel
from src.models.fraud_detection_model import FraudDetectionModel
from src.data.preprocessing import preprocess_loan_application
from src.configs.config import MODEL_DIR
from src.models.training import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)

logger = logging.getLogger(__name__)

def load_and_preprocess_data():
    """Load and preprocess all data sources"""
    logger.info("Loading data files...")
    
    # Load all data sources
    credit_data = pd.read_csv(os.path.join(project_root, 'data', 'credit_data.csv'))
    banking_data = pd.read_csv(os.path.join(project_root, 'data', 'banking_data.csv'))
    application_data = pd.read_csv(os.path.join(project_root, 'data', 'application_data.csv'))
    outcome_data = pd.read_csv(os.path.join(project_root, 'data', 'outcome_data.csv'))
    loan_data = pd.read_csv(os.path.join(project_root, 'data', 'loan_data.csv'))
    
    # Merge all data using left joins to preserve all records
    merged_data = pd.merge(credit_data, banking_data, on='applicant_id', how='left')
    merged_data = pd.merge(merged_data, application_data, on='applicant_id', how='left')
    merged_data = pd.merge(merged_data, loan_data, on='applicant_id', how='left')
    merged_data = pd.merge(merged_data, outcome_data, on='applicant_id', how='left')
    
    logger.info(f"Loaded and merged data shape: {merged_data.shape}")
    
    # Add target columns
    if 'loan_status' in merged_data.columns:
        merged_data['is_default'] = (merged_data['loan_status'] == 'defaulted').astype(int)
    if 'fraud_flag' in merged_data.columns:
        merged_data['is_fraud'] = merged_data['fraud_flag'].fillna(0).astype(int)

    # Drop leakage columns
    drop_cols = [
        "is_default", "is_fraud", "applicant_id", "loan_status", "days_to_default",
        "fraud_flag", "application_timestamp", "device_id", "ip_address", "email_domain"
    ]
    features = merged_data.drop([col for col in drop_cols if col in merged_data.columns], axis=1)
    targets = merged_data[['is_default', 'is_fraud']]

    # Preprocess features only
    processed_features = features.copy()
    
    # Fill missing values with median for numeric columns
    numeric_cols = processed_features.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if processed_features[col].isnull().any():
            median_val = processed_features[col].median()
            processed_features[col] = processed_features[col].fillna(median_val)

    # Concatenate processed features and targets
    processed_data = pd.concat([processed_features, targets], axis=1)
    
    return processed_data

def train_credit_risk_model(data):
    """Train the credit risk model"""
    logger.info("Training credit risk model...")
    
    # Use ModelTrainer for risk
    trainer = ModelTrainer("risk")
    trainer.train(data, test_size=0.2, batch_size=50000)  # Increased batch size
    model_path = os.path.join(MODEL_DIR, 'risk_model.joblib')
    trainer.save_model(model_path)
    logger.info(f"Credit risk model trained and saved to {model_path}")
    logger.info(f"Model metrics: {trainer.metrics}")
    return trainer

def train_fraud_detection_model(data):
    """Train the fraud detection model"""
    logger.info("Training fraud detection model...")
    
    # Use ModelTrainer for fraud
    trainer = ModelTrainer("fraud")
    trainer.train(data, test_size=0.2, batch_size=50000)  # Increased batch size
    model_path = os.path.join(MODEL_DIR, 'fraud_model.joblib')
    trainer.save_model(model_path)
    logger.info(f"Fraud detection model trained and saved to {model_path}")
    logger.info(f"Model metrics: {trainer.metrics}")
    return trainer

def main():
    """Main training function"""
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load and preprocess data
    data = load_and_preprocess_data()
    
    # Train models
    credit_model = train_credit_risk_model(data)
    fraud_model = train_fraud_detection_model(data)
    
    logger.info("Model training completed successfully!")

if __name__ == "__main__":
    main() 