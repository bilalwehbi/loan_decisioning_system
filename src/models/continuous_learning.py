import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Tuple
from .training import ModelTrainer
import joblib
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class ContinuousLearner:
    def __init__(self, risk_model_path: str, fraud_model_path: str):
        """Initialize the continuous learner with existing models."""
        self.risk_trainer = ModelTrainer.load_model(risk_model_path)
        self.fraud_trainer = ModelTrainer.load_model(fraud_model_path)
        self.min_samples_for_update = 1000  # Minimum samples needed for an update
        self.max_update_frequency = timedelta(days=1)  # Maximum update frequency
        
    def prepare_application_data(self, applications: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert application data to training format."""
        features_list = []
        
        for app in applications:
            features = {
                "credit_score": app["credit_data"]["credit_score"],
                "delinquencies": app["credit_data"]["delinquencies"],
                "inquiries_last_6m": app["credit_data"]["inquiries_last_6m"],
                "tradelines": app["credit_data"]["tradelines"],
                "utilization": app["credit_data"]["utilization"],
                "payment_history_score": app["credit_data"]["payment_history_score"],
                "credit_age_months": app["credit_data"]["credit_age_months"],
                "credit_mix_score": app["credit_data"]["credit_mix_score"],
                "avg_monthly_income": app["banking_data"]["avg_monthly_income"],
                "income_stability_score": app["banking_data"]["income_stability_score"],
                "spending_pattern_score": app["banking_data"]["spending_pattern_score"],
                "transaction_count": app["banking_data"]["transaction_count"],
                "avg_account_balance": app["banking_data"]["avg_account_balance"],
                "overdraft_frequency": app["banking_data"]["overdraft_frequency"],
                "savings_rate": app["banking_data"]["savings_rate"],
                "application_completion_time": app["behavioral_data"]["application_completion_time"],
                "device_trust_score": app["behavioral_data"]["device_trust_score"],
                "location_risk_score": app["behavioral_data"]["location_risk_score"],
                "digital_footprint_score": app["behavioral_data"]["digital_footprint_score"],
                "employment_length_months": app["employment_data"]["employment_length_months"],
                "employment_verification_score": app["employment_data"]["employment_verification_score"],
                "income_verification_score": app["employment_data"]["income_verification_score"],
                "loan_amount": app["loan_amount"],
                "loan_term": app["loan_term"]
            }
            
            # Add outcomes if available
            if "outcome" in app:
                features["is_default"] = 1 if app["outcome"].get("defaulted", False) else 0
                features["is_fraud"] = 1 if app["outcome"].get("fraud", False) else 0
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def should_update_model(self, trainer: ModelTrainer) -> bool:
        """Check if model should be updated based on time and data availability."""
        if trainer.last_training_date is None:
            return True
        
        time_since_last_update = datetime.utcnow() - trainer.last_training_date
        return time_since_last_update >= self.max_update_frequency
    
    def update_models(self, new_applications: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Update models with new application data."""
        if len(new_applications) < self.min_samples_for_update:
            logger.info(f"Not enough samples for update. Need {self.min_samples_for_update}, got {len(new_applications)}")
            return {
                "risk_model": self.risk_trainer.metrics,
                "fraud_model": self.fraud_trainer.metrics
            }
        
        # Convert applications to training data
        new_data = self.prepare_application_data(new_applications)
        
        # Drop leakage columns
        drop_cols = [
            "is_default", "is_fraud", "applicant_id", "loan_status", "days_to_default",
            "fraud_flag", "application_timestamp", "device_id", "ip_address", "email_domain"
        ]
        X = new_data.drop([col for col in drop_cols if col in new_data.columns], axis=1)
        y = new_data["is_default"]  # or "is_fraud" for fraud model
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Update risk model if needed
        risk_metrics = self.risk_trainer.metrics
        if self.should_update_model(self.risk_trainer):
            logger.info("Updating risk model...")
            risk_metrics = self.risk_trainer.update_model(X_train, y_train)
            self.risk_trainer.save_model("models/risk_model.joblib")
        
        # Update fraud model if needed
        fraud_metrics = self.fraud_trainer.metrics
        if self.should_update_model(self.fraud_trainer):
            logger.info("Updating fraud model...")
            fraud_metrics = self.fraud_trainer.update_model(X_train, y_train)
            self.fraud_trainer.save_model("models/fraud_model.joblib")
        
        return {
            "risk_model": risk_metrics,
            "fraud_model": fraud_metrics
        }
    
    def get_model_performance_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get the performance history of both models."""
        return {
            "risk_model": self.risk_trainer.training_history,
            "fraud_model": self.fraud_trainer.training_history
        }
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance for both models."""
        return {
            "risk_model": self.risk_trainer.feature_importance,
            "fraud_model": self.fraud_trainer.feature_importance
        }

    def prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        drop_cols = [
            "is_default", "is_fraud", "applicant_id", "loan_status", "days_to_default",
            "fraud_flag", "application_timestamp", "device_id", "ip_address", "email_domain"
        ]
        features = data.drop([col for col in drop_cols if col in data.columns], axis=1)
        # Drop all object (string) columns except those you want to encode
        features = features.select_dtypes(exclude=['object'])
        if self.model_type == "risk":
            target = data["is_default"]
        else:
            target = data["is_fraud"]
        return features, target

@classmethod
def load_model(cls, path: str) -> "ModelTrainer":
    model_data = joblib.load(path)
    if isinstance(model_data, dict) and "model_type" in model_data:
        trainer = cls(model_data["model_type"])
        trainer.model = model_data["model"]
        trainer.feature_importance = model_data["feature_importance"]
        trainer.metrics = model_data["metrics"]
        trainer.training_history = model_data.get("training_history", [])
        trainer.last_training_date = model_data.get("last_training_date")
        return trainer
    else:
        # Assume it's just a model object
        trainer = cls("risk")  # or "fraud" as appropriate
        trainer.model = model_data
        return trainer 

print("Feature columns:", X.columns)
print("Target correlation with features:")
print(X.corrwith(y)) 