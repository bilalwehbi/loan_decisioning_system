# src/models/test_explainability.py
import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.loan_decisioning_service import LoanDecisioningService
from src.models.test_fraud_detection_inference import create_sample_applications
from src.configs.config import DATA_DIR
from src.models.feature_mapping import create_feature_index_mapping, get_display_name

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('explainability_test.log')
    ]
)

logger = logging.getLogger(__name__)

def test_explainability():
    """Test the explainability layer of the loan decisioning service"""
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
    applications = create_sample_applications(num_samples=3)
    
    logger.info(f"Testing with {len(applications)} sample applications")
    
    # Process each application and examine explanations
    for i, (application, actual) in enumerate(applications, 1):
        logger.info(f"\nProcessing application {i}")
        logger.info(f"Loan details: Amount=${application['loan_amount']:.2f}, Purpose={application['loan_purpose']}")
        
        # Make decision with explanations
        decision = service.make_decision(application)
        
        # Log results
        logger.info(f"Decision: {decision['decision']}")
        logger.info(f"Credit score: {decision['credit_score']}, Default probability: {decision['probability_of_default']:.4f}")
        logger.info(f"Fraud score: {decision['fraud_score']:.4f}, Fraud flags: {', '.join(decision['fraud_flags']) if decision['fraud_flags'] else 'None'}")
        
        # Log explanations
        logger.info("\nExplanations:")
        for j, exp in enumerate(decision['explanations'], 1):
            logger.info(f"  {j}. {exp['description']} (feature: {exp['feature']}, impact: {exp['contribution']:.4f})")
        
        logger.info(f"Processing time: {decision['processing_time']:.3f} seconds")
        logger.info("-" * 80)
    
    # Generate a SHAP summary plot for visualization (optional)
    try:
        # Load a sample of data for SHAP summary plot
        df = pd.read_csv(os.path.join(DATA_DIR, "merged_dataset.csv"), low_memory=False)
        sample_df = df.sample(min(100, len(df)))
        
        # Prepare features
        exclude_cols = [col for col in ['applicant_id', 'loan_status', 'days_to_default', 
                        'default', 'fraud_flag'] if col in sample_df.columns]
        feature_df = sample_df.drop(columns=exclude_cols)
        
        # Get processed features
        X_processed = service.risk_model.pipeline.transform(feature_df)
        
        # Get feature names
        if hasattr(service.risk_model, 'feature_names') and service.risk_model.feature_names:
            feature_names = service.risk_model.feature_names
        else:
            feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
        
        # Create explainer
        explainer = shap.TreeExplainer(service.risk_model.model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_processed)
        
        # If multiple outputs, use the positive class
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1]
        
        # Create summary plot with mapped feature names
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, 
            X_processed,
            feature_names=feature_names[:X_processed.shape[1]],  # Ensure we don't exceed feature count
            show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_DIR, "shap_summary_plot.png"))
        plt.close()
        
        logger.info(f"SHAP summary plot saved to {os.path.join(DATA_DIR, 'shap_summary_plot.png')}")
    except Exception as e:
        logger.error(f"Error generating SHAP summary plot: {str(e)}")
        logger.error(f"Exception details: {traceback.format_exc()}")
    
if __name__ == "__main__":
    logger.info("Testing explainability layer")
    test_explainability()
    logger.info("Testing completed")
    logger.info("Testing completed")
    logger.info("Testing completed")