# src/models/train_fraud_detection_model.py
import os
import sys
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.preprocessing import StandardScaler

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.fraud_detection_model import FraudDetectionModel
from src.configs.config import DATA_DIR, MODEL_DIR, PERFORMANCE_TARGETS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fraud_training.log')
    ]
)

logger = logging.getLogger(__name__)

def load_data():
    """Load and prepare training data for fraud detection"""
    logger.info("Loading training data for fraud detection...")
    
    # Load training and test data
    train_df = pd.read_csv(os.path.join(DATA_DIR, "fraud_detection_train.csv"), low_memory=False)
    test_df = pd.read_csv(os.path.join(DATA_DIR, "fraud_detection_test.csv"), low_memory=False)
    
    logger.info(f"Loaded {len(train_df)} training rows and {len(test_df)} test rows")
    
    # Define features and target
    exclude_cols = ['is_fraud']  # Target column
    
    # Define features and target for training
    X_train = train_df.drop(columns=exclude_cols)
    y_train = train_df['is_fraud']
    
    # Define features and target for testing
    X_test = test_df.drop(columns=exclude_cols)
    y_test = test_df['is_fraud']
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate():
    """Train and evaluate the fraud detection model"""
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Log columns for debugging
    logger.info(f"X_train columns: {list(X_train.columns)}")
    logger.info(f"X_test columns: {list(X_test.columns)}")
    
    # Feature Engineering
    # Time-based features
    X_train['time_risk'] = X_train['hour_of_day'] * X_train['day_of_week']
    X_test['time_risk'] = X_test['hour_of_day'] * X_test['day_of_week']
    
    # Device and contact features
    X_train['device_contact_risk'] = X_train['device_id_length'] * X_train['email_domain_length'] * X_train['phone_number_length']
    X_test['device_contact_risk'] = X_test['device_id_length'] * X_test['email_domain_length'] * X_test['phone_number_length']
    
    # Credit-related features
    X_train['credit_risk'] = (X_train['credit_score'] / 850) * (1 - X_train['utilization_rate'])
    X_test['credit_risk'] = (X_test['credit_score'] / 850) * (1 - X_test['utilization_rate'])
    
    X_train['income_risk'] = (X_train['monthly_income'] / 20000) * X_train['income_stability_score']
    X_test['income_risk'] = (X_test['monthly_income'] / 20000) * X_test['income_stability_score']
    
    X_train['loan_risk'] = (X_train['loan_amount'] / 10000) * (X_train['loan_term'] / 60)
    X_test['loan_risk'] = (X_test['loan_amount'] / 10000) * (X_test['loan_term'] / 60)
    
    # Behavioral features
    X_train['behavioral_risk'] = X_train['spending_pattern_score'] * (1 - X_train['avg_daily_balance'] / X_train['avg_monthly_deposits'])
    X_test['behavioral_risk'] = X_test['spending_pattern_score'] * (1 - X_test['avg_daily_balance'] / X_test['avg_monthly_deposits'])
    
    # Application behavior features
    X_train['application_risk'] = (1 - X_train['application_completion_rate']) * (X_train['form_fill_time'] / 15)
    X_test['application_risk'] = (1 - X_test['application_completion_rate']) * (X_test['form_fill_time'] / 15)
    
    # Risk score combinations
    X_train['combined_risk_score'] = (
        X_train['device_risk_score'] * 0.2 +
        X_train['ip_risk_score'] * 0.2 +
        X_train['location_risk_score'] * 0.2 +
        X_train['behavioral_risk_score'] * 0.4
    )
    X_test['combined_risk_score'] = (
        X_test['device_risk_score'] * 0.2 +
        X_test['ip_risk_score'] * 0.2 +
        X_test['location_risk_score'] * 0.2 +
        X_test['behavioral_risk_score'] * 0.4
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled arrays back to DataFrames
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Data Augmentation using SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled_df, y_train)
    
    # Initialize base model
    base_model = FraudDetectionModel()
    
    # Feature Selection based on coefficients
    base_model.model.fit(X_train_resampled, y_train_resampled)
    coefficients = base_model.model.coef_[0]
    feature_importance = pd.Series(coefficients, index=X_train_resampled.columns)
    selected_features = feature_importance[abs(feature_importance) > 0.01].index

    X_train_selected = X_train_resampled[selected_features]
    X_test_selected = X_test_scaled_df[selected_features]
    
    # Train the model using the selected features
    metrics = base_model.train(X_train_selected, y_train_resampled, X_test_selected, y_test)
    
    # Log performance metrics
    logger.info("Fraud Detection Model Performance Metrics:")
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
    model_path = base_model.save()
    
    logger.info(f"Model {'meets' if targets_met else 'does not meet'} performance targets")
    logger.info(f"Model saved to {model_path}")
    
    return base_model, metrics

if __name__ == "__main__":
    logger.info("Starting fraud detection model training")
    model, metrics = train_and_evaluate()
    logger.info("Fraud detection model training completed")