# src/models/train_credit_risk_model.py
import os
import sys
import logging
import pandas as pd
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.credit_risk_model import CreditRiskModel
from src.configs.config import DATA_DIR, MODEL_DIR, PERFORMANCE_TARGETS

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

def load_data():
    """Load and prepare training data"""
    logger.info("Loading training data...")
    
    # Load training and test data
    train_df = pd.read_csv(os.path.join(DATA_DIR, "credit_risk_train.csv"), low_memory=False)
    test_df = pd.read_csv(os.path.join(DATA_DIR, "credit_risk_test.csv"), low_memory=False)
    
    logger.info(f"Loaded {len(train_df)} training rows and {len(test_df)} test rows")
    
    # Define features and target
    exclude_cols = ['is_default']  # Target column
    
    # Define features and target for training
    X_train = train_df.drop(columns=exclude_cols)
    y_train = train_df['is_default']
    
    # Define features and target for testing
    X_test = test_df.drop(columns=exclude_cols)
    y_test = test_df['is_default']
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate():
    """Train and evaluate the credit risk model"""
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Initialize model
    model = CreditRiskModel()
    
    # Train model
    metrics = model.train(X_train, y_train, X_test, y_test)
    
    # Log performance metrics
    logger.info("Model Performance Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value}")
    
    # Evaluate against performance targets
    targets_met = True
    
    if metrics['precision'] < PERFORMANCE_TARGETS['precision']:
        logger.warning(f"Precision ({metrics['precision']:.4f}) is below target ({PERFORMANCE_TARGETS['precision']})")
        targets_met = False
        
    if metrics['false_positive_rate'] > PERFORMANCE_TARGETS['false_positive_rate']:
        logger.warning(f"False positive rate ({metrics['false_positive_rate']:.4f}) is above target ({PERFORMANCE_TARGETS['false_positive_rate']})")
        targets_met = False
    
    # Save model
    model_path = model.save()
    
    logger.info(f"Model {'meets' if targets_met else 'does not meet'} performance targets")
    logger.info(f"Model saved to {model_path}")
    
    return model, metrics

if __name__ == "__main__":
    logger.info("Starting credit risk model training")
    model, metrics = train_and_evaluate()
    logger.info("Credit risk model training completed")