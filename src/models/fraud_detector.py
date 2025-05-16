import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
import re
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)

class FraudDetector:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.pipeline = None
        self.feature_names = None
        self.threshold = 0.8  # Fraud score threshold
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def _prepare_features(self, data):
        """Prepare features for model training or prediction."""
        # Define numeric and categorical features
        numeric_features = [
            'loan_amount', 'loan_term', 'credit_score', 'delinquencies',
            'inquiries_last_6m', 'tradelines', 'utilization', 'average_balance',
            'income', 'employment_length', 'avg_monthly_income',
            'income_stability_score', 'spending_pattern_score', 'transaction_count'
        ]
        
        categorical_features = ['purpose', 'has_co_applicant', 'email_domain']
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return preprocessor, numeric_features + categorical_features
    
    def train(self, train_data, test_data):
        """Train the fraud detection model."""
        # Prepare features
        preprocessor, feature_names = self._prepare_features(train_data)
        self.feature_names = feature_names
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', IsolationForest(
                n_estimators=100,
                contamination=0.1,
                random_state=42
            ))
        ])
        
        # Prepare data
        X_train = train_data.drop(['default', 'fraud'], axis=1)
        y_train = train_data['fraud']
        X_test = test_data.drop(['default', 'fraud'], axis=1)
        y_test = test_data['fraud']
        
        # Train model
        self.pipeline.fit(X_train)
        self.model = self.pipeline.named_steps['classifier']
        
        # Calculate metrics
        train_score = self.pipeline.score(X_train)
        test_score = self.pipeline.score(X_test)
        
        metrics = {
            'train_score': train_score,
            'test_score': test_score
        }
        
        return metrics
    
    def detect_fraud(self, application_data):
        """Detect potential fraud in a loan application."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Convert to DataFrame if dict
        if isinstance(application_data, dict):
            df = pd.DataFrame([application_data])
        else:
            df = application_data
        
        # Get prediction
        fraud_score = self.pipeline.score_samples(df)[0]
        
        # Convert to 0-1 scale (higher score means more likely to be fraud)
        fraud_score = 1 - (fraud_score - self.model.offset_) / self.model.threshold_
        
        # Generate fraud flags
        fraud_flags = []
        
        # Check specific patterns
        if isinstance(application_data, dict):
            # Check credit score vs income
            if (application_data['credit_data']['credit_score'] < 600 and 
                application_data['banking_data']['monthly_income'] > 10000):
                fraud_flags.append("Suspicious income for credit profile")
                fraud_score = max(fraud_score, 0.7)
            
            # Check utilization
            if application_data['credit_data']['credit_utilization'] > 0.9:
                fraud_flags.append("Very high credit utilization")
                fraud_score = max(fraud_score, 0.5)
            
            # Check inquiries
            if application_data['credit_data']['inquiries_last_6mo'] > 5:
                fraud_flags.append("Multiple recent credit inquiries")
                fraud_score = max(fraud_score, 0.6)
            
            # Check email domain
            if application_data['application_data']['email_domain'] in ['tempmail.com', 'throwaway.com', 'temp-mail.org']:
                fraud_flags.append("Suspicious email domain")
                fraud_score = max(fraud_score, 0.8)
            
            # Check employment length
            if application_data['applicant_employment_length_years'] < 1:
                fraud_flags.append("Short employment history")
                fraud_score = max(fraud_score, 0.4)
            
            # Check income stability
            if application_data['banking_data']['income_stability_score'] < 0.4:
                fraud_flags.append("Unstable income pattern")
                fraud_score = max(fraud_score, 0.6)
            
            # Check application velocity
            if application_data['application_data']['num_previous_applications'] > 3:
                fraud_flags.append("High application velocity")
                fraud_score = max(fraud_score, 0.7)
            
            # Check time spent on application
            if application_data['application_data']['time_spent_on_application'] < 300:
                fraud_flags.append("Suspiciously fast application completion")
                fraud_score = max(fraud_score, 0.5)
        
        return fraud_score, fraud_flags
    
    def save_model(self, path):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        joblib.dump(self.pipeline, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model from disk."""
        self.pipeline = joblib.load(path)
        self.model = self.pipeline.named_steps['classifier']
        logger.info(f"Model loaded from {path}")

    def preprocess_features(self, application_data: Dict) -> pd.DataFrame:
        """Convert application data into fraud detection features."""
        features = {}
        
        # Device and IP features
        features.update(self._extract_device_ip_features(application_data))
        
        # Email and phone features
        features.update(self._extract_contact_features(application_data))
        
        # Income consistency features
        features.update(self._extract_income_features(application_data))
        
        # Application velocity features
        features.update(self._extract_velocity_features(application_data))
        
        return pd.DataFrame([features])

    def _extract_device_ip_features(self, application_data: Dict) -> Dict:
        """Extract features related to device and IP."""
        device_id = application_data['application_data']['device_id']
        ip_address = application_data['application_data']['ip_address']
        
        # Placeholder for actual device velocity calculation
        device_velocity = 1.0  # Should be calculated based on historical data
        
        # Simple IP risk scoring (placeholder)
        ip_risk_score = 0.5  # Should be calculated based on IP reputation
        
        return {
            'device_velocity': device_velocity,
            'ip_risk_score': ip_risk_score
        }

    def _extract_contact_features(self, application_data: Dict) -> Dict:
        """Extract features related to email and phone."""
        email = application_data['application_data']['email_domain']
        phone = application_data['application_data']['phone_number']
        
        # Email domain risk (placeholder)
        email_domain_risk = 0.5  # Should be calculated based on domain reputation
        
        # Phone number risk (placeholder)
        phone_risk_score = 0.5  # Should be calculated based on phone number patterns
        
        return {
            'email_domain_risk': email_domain_risk,
            'phone_risk_score': phone_risk_score
        }

    def _extract_income_features(self, application_data: Dict) -> Dict:
        """Extract features related to income consistency."""
        banking_data = application_data['banking_data']
        
        # Calculate income consistency score
        income_consistency = banking_data['income_stability_score']
        
        return {
            'income_consistency': income_consistency
        }

    def _extract_velocity_features(self, application_data: Dict) -> Dict:
        """Extract features related to application velocity."""
        # Placeholder for actual velocity calculation
        application_velocity = 1.0  # Should be calculated based on application patterns
        
        return {
            'application_velocity': application_velocity
        }

    def _generate_fraud_flags(self, fraud_score: float, application_data: Dict) -> List[str]:
        """Generate human-readable fraud flags based on risk score and features."""
        flags = []
        
        if fraud_score > 0.8:
            flags.append("High Risk: Multiple fraud indicators detected")
        elif fraud_score > 0.6:
            flags.append("Medium Risk: Suspicious patterns detected")
        
        # Check specific fraud indicators
        if self._check_synthetic_identity(application_data):
            flags.append("Synthetic ID suspected")
        
        if self._check_income_mismatch(application_data):
            flags.append("Income mismatch detected")
        
        if self._check_velocity_risk(application_data):
            flags.append("High application velocity")
        
        return flags

    def _check_synthetic_identity(self, application_data: Dict) -> bool:
        """Check for synthetic identity indicators."""
        # Placeholder implementation
        return False

    def _check_income_mismatch(self, application_data: Dict) -> bool:
        """Check for income inconsistencies."""
        banking_data = application_data['banking_data']
        return banking_data['income_stability_score'] < 0.3

    def _check_velocity_risk(self, application_data: Dict) -> bool:
        """Check for suspicious application velocity."""
        # Placeholder implementation
        return False 