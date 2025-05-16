import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.fraud_detection_model import FraudDetectionModel
from src.configs.config import DATA_DIR, MODEL_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fraud_testing.log')
    ]
)

logger = logging.getLogger(__name__)

def load_test_data():
    """Load test data for fraud detection"""
    logger.info("Loading test data for fraud detection...")
    
    # Load test data
    test_df = pd.read_csv(os.path.join(DATA_DIR, "fraud_detection_test.csv"), low_memory=False)
    
    logger.info(f"Loaded {len(test_df)} test rows")
    
    # Add missing required columns with default values
    required_columns = {
        'num_nsf_transactions': 0,
        'email_domain': 'example.com',
        'num_overdrafts': 0,
        'applicant_income': 50000,
        'loan_purpose': 'personal',
        'applicant_employment_length_years': 5
    }
    
    for col, default_value in required_columns.items():
        if col not in test_df.columns:
            test_df[col] = default_value
            logger.info(f"Added missing column {col} with default value {default_value}")
    
    # Define features and target
    exclude_cols = ['is_fraud']  # Target column
    
    # Define features and target for testing
    X_test = test_df.drop(columns=exclude_cols)
    y_test = test_df['is_fraud']
    
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance"""
    logger.info("Evaluating model performance...")
    
    try:
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Log metrics
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        logger.info("\nConfusion Matrix:")
        logger.info(conf_matrix)
        logger.info(f"\nROC AUC: {roc_auc:.4f}")
        
        return report, conf_matrix, roc_auc
        
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        return None, None, None

def main():
    """Main function to test the fraud detection model"""
    logger.info("Starting fraud detection model testing")
    
    # Load test data
    X_test, y_test = load_test_data()
    
    # Find the most recent model file
    model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('fraud_detection_model_') and f.endswith('.pkl')]
    if not model_files:
        logger.error("No trained fraud detection models found in models directory")
        return None, None, None
        
    latest_model = max(model_files)  # Get the most recent model file
    model_filepath = os.path.join(MODEL_DIR, latest_model)
    logger.info(f"Using model file: {latest_model}")
    
    # Load the trained model
    model = FraudDetectionModel()
    model.load(model_filepath)
    
    # Evaluate model
    report, conf_matrix, roc_auc = evaluate_model(model, X_test, y_test)
    
    logger.info("Fraud detection model testing completed")
    
    return report, conf_matrix, roc_auc

if __name__ == "__main__":
    main() 