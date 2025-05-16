# src/models/test_feature_mapping.py
import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.loan_decisioning_service import LoanDecisioningService
from src.models.feature_mapping import get_display_name, create_feature_index_mapping
from src.configs.config import DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('feature_mapping_test.log')
    ]
)

logger = logging.getLogger(__name__)

def test_feature_mapping():
    """Test feature name mapping for visualizations"""
    # Initialize service
    service = LoanDecisioningService()
    
    try:
        # Load models
        service.load_models()
        logger.info("Successfully loaded models")
        
        # Get feature names from the model
        credit_feature_names = service.risk_model.feature_names
        fraud_feature_names = service.fraud_model.feature_names

        # Create feature mapping
        feature_mapping = {
            'credit_risk': dict(zip(credit_feature_names, credit_feature_names)),
            'fraud_detection': dict(zip(fraud_feature_names, fraud_feature_names))
        }

        # Test feature mapping
        test_features = {
            'credit_risk': {
                'income': 50000,
                'employment_length': 5,
                'debt_to_income': 0.3,
                'credit_score': 700,
                'payment_history': 0.95,
                'credit_utilization': 0.4,
                'number_of_accounts': 3,
                'age': 35,
                'education': 'bachelor',
                'marital_status': 'married'
            },
            'fraud_detection': {
                'transaction_amount': 1000,
                'time_of_day': 14,
                'location_mismatch': 0,
                'device_type': 'mobile',
                'ip_risk_score': 0.1,
                'velocity_checks': 0,
                'account_age_days': 365,
                'previous_chargebacks': 0,
                'email_domain_risk': 0.1,
                'browser_mismatch': 0
            }
        }

        # Test credit risk feature mapping
        credit_features = test_features['credit_risk']
        feature_df = pd.DataFrame([credit_features])
        X_processed = service.risk_model.pipeline.transform(feature_df)
        print("\nCredit Risk Feature Mapping:")
        for orig, mapped in feature_mapping['credit_risk'].items():
            print(f"{orig} -> {mapped}")

        # Test fraud detection feature mapping
        fraud_features = test_features['fraud_detection']
        feature_df = pd.DataFrame([fraud_features])
        X_processed = service.fraud_model.pipeline.transform(feature_df)
        print("\nFraud Detection Feature Mapping:")
        for orig, mapped in feature_mapping['fraud_detection'].items():
            print(f"{orig} -> {mapped}")

        # Test SHAP values
        print("\nTesting SHAP values...")
        feature_df = pd.DataFrame([credit_features])
        X_processed = service.risk_model.pipeline.transform(feature_df)
        explainer = shap.TreeExplainer(service.risk_model.model)
        shap_values = explainer.shap_values(X_processed)
        print("SHAP values shape:", np.array(shap_values).shape)
        
        # Map feature names to display names
        display_names = [get_display_name(name) for name in credit_feature_names]
        
        # Print sample of the mapping
        logger.info("Sample feature name mapping:")
        for i, (orig, display) in enumerate(zip(credit_feature_names[:10], display_names[:10])):
            logger.info(f"  {i}: {orig} -> {display}")
        
        # Create visualization with display names
        # Load a sample of data
        df = pd.read_csv(os.path.join(DATA_DIR, "merged_dataset.csv"), low_memory=False)
        sample_df = df.sample(min(100, len(df)))
        
        # Prepare features
        exclude_cols = [col for col in ['applicant_id', 'loan_status', 'days_to_default', 
                        'default', 'fraud_flag'] if col in sample_df.columns]
        feature_df = sample_df.drop(columns=exclude_cols)
        
        # Get processed features
        X_processed = service.risk_model.pipeline.transform(feature_df)
        
        # Create explainer
        explainer = shap.TreeExplainer(service.risk_model.model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_processed)
        
        # If multiple outputs, use the positive class
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1]
        
        # Create summary plot with mapped feature names
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_values, 
            X_processed,
            feature_names=display_names[:X_processed.shape[1]],
            show=False
        )
        plt.title("Credit Risk Model with Display Names")
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_DIR, "mapped_feature_names_plot.png"))
        plt.close()
        
        logger.info(f"SHAP plot with display names saved to {os.path.join(DATA_DIR, 'mapped_feature_names_plot.png')}")
        
    except Exception as e:
        logger.error(f"Error testing feature mapping: {str(e)}")

if __name__ == "__main__":
    logger.info("Testing feature name mapping")
    test_feature_mapping()
    logger.info("Testing completed")