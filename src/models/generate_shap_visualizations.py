# src/models/generate_shap_visualizations.py
import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.credit_risk_model import CreditRiskModel
from src.models.fraud_detection_model import FraudDetectionModel
from src.configs.config import MODEL_DIR, DATA_DIR
from src.models.feature_mapping import create_feature_index_mapping, get_display_name
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('shap_visualizations.log')
    ]
)

logger = logging.getLogger(__name__)

def load_models():
    """Load the trained models."""
    try:
        # Find the latest model files
        credit_model_files = [f for f in os.listdir(MODEL_DIR) 
                            if f.startswith('credit_risk_model_') and f.endswith('.pkl')]
        fraud_model_files = [f for f in os.listdir(MODEL_DIR) 
                           if f.startswith('fraud_detection_model_') and f.endswith('.pkl')]
        
        if not credit_model_files or not fraud_model_files:
            raise FileNotFoundError("No model files found")
        
        # Load the latest models
        credit_model_path = os.path.join(MODEL_DIR, sorted(credit_model_files)[-1])
        fraud_model_path = os.path.join(MODEL_DIR, sorted(fraud_model_files)[-1])
        
        risk_model = CreditRiskModel()
        risk_model.load(credit_model_path)
        logger.info(f"Loaded credit risk model: {risk_model.model_id}")
        
        fraud_model = FraudDetectionModel()
        fraud_model.load(fraud_model_path)
        logger.info(f"Loaded fraud detection model: {fraud_model.model_id}")
        
        return risk_model, fraud_model
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

def generate_credit_risk_visualizations(risk_model):
    """Generate SHAP visualizations for credit risk model."""
    try:
        # Load test data
        test_data = load_test_data()
        if test_data is None:
            return
        
        # Prepare features
        feature_df = test_data.drop('default', axis=1)
        
        # Get processed features
        X_processed = risk_model.pipeline.transform(feature_df)
        
        # Create explainer
        explainer = shap.TreeExplainer(risk_model.model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_processed)
        
        # Generate visualizations
        generate_summary_plot(shap_values, X_processed, risk_model.feature_names)
        generate_dependence_plots(shap_values, X_processed, risk_model.feature_names)
        generate_force_plots(shap_values, X_processed, risk_model.feature_names)
        
    except Exception as e:
        logger.error(f"Error generating credit risk visualizations: {str(e)}")
        raise

def generate_fraud_detection_visualizations(model):
    """Generate SHAP visualizations for fraud detection model"""
    # Create visualization output directory
    viz_dir = os.path.join(DATA_DIR, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load a sample of data
    df = pd.read_csv(os.path.join(DATA_DIR, "merged_dataset.csv"), low_memory=False)
    sample_df = df.sample(min(500, len(df)))
    
    # Prepare features
    exclude_cols = ['applicant_id', 'loan_status', 'days_to_default', 
                    'default', 'fraud_flag']
    feature_df = sample_df.drop(columns=[col for col in exclude_cols if col in sample_df.columns])
    
    # Get processed features
    X_processed = model.pipeline.transform(feature_df)
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
    
    # Create explainer
    explainer = shap.TreeExplainer(model.model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_processed)
    
    # If multiple outputs, use the positive class
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values = shap_values[1]
    
    # 1. Summary Plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values, 
        X_processed,
        feature_names=feature_names,
        show=False
    )
    plt.title("Fraud Detection Model SHAP Summary")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "fraud_detection_shap_summary.png"))
    plt.close()
    
    # 2. Bar Plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values, 
        X_processed,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.title("Fraud Detection Model Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "fraud_detection_feature_importance.png"))
    plt.close()
    
    # 3. Dependence Plots for top features
    # Get mean absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Get indices of top 5 features
    top_indices = np.argsort(mean_abs_shap)[-5:]
    
    for idx in top_indices:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            idx, 
            shap_values, 
            X_processed,
            feature_names=feature_names,
            show=False
        )
        plt.title(f"Dependence Plot for {feature_names[idx]}")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"fraud_detection_dependence_{feature_names[idx]}.png"))
        plt.close()
    
    logger.info(f"Fraud detection SHAP visualizations saved to {viz_dir}")

def main():
    """Main function to generate SHAP visualizations."""
    try:
        # Load models
        risk_model, fraud_model = load_models()
        
        # Generate visualizations
        generate_credit_risk_visualizations(risk_model)
        generate_fraud_detection_visualizations(fraud_model)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()