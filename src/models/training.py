import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import logging
from typing import Dict, Any, Tuple, List
from datetime import datetime
from ..data.data_generator import LoanDataGenerator

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model_type: str):
        """Initialize the model trainer."""
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        self.metrics = {}
        self.training_history = []
        self.last_training_date = None
        
    def prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training."""
        # Select features based on model type
        drop_cols = [
            "is_default", "is_fraud", "applicant_id", "loan_status", "application_timestamp", "device_id", "ip_address", "email_domain"
        ]
        features = data.drop([col for col in drop_cols if col in data.columns], axis=1)
        # Drop all object (string) columns
        features = features.select_dtypes(exclude=['object'])
        if self.model_type == "risk":
            target = data["is_default"]
        else:
            target = data["is_fraud"]
        return features, target
    
    def train(self, data: pd.DataFrame, test_size: float = 0.2, batch_size: int = 50000) -> Dict[str, Any]:
        """Train the model and evaluate its performance."""
        # Prepare data
        X, y = self.prepare_data(data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
        
        # Initialize model with parameters optimized for large datasets
        self.model = RandomForestClassifier(
            n_estimators=200,  # Increased for better performance
            max_depth=15,      # Increased for more complex patterns
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,         # Use all available CPU cores
            verbose=1          # Show progress
        )
        
        # Train model in batches if dataset is large
        if len(X_train) > batch_size:
            logger.info(f"Training on {len(X_train)} samples in batches of {batch_size}")
            for i in range(0, len(X_train), batch_size):
                end_idx = min(i + batch_size, len(X_train))
                logger.info(f"Training batch {i//batch_size + 1} of {(len(X_train) + batch_size - 1)//batch_size}")
                self.model.fit(X_train[i:end_idx], y_train[i:end_idx])
        else:
            self.model.fit(X_train, y_train)
        
        # Get feature importance
        self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))
        print("FEATURES USED FOR TRAINING:", list(X.columns))
        
        # Evaluate model on test set
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "training_date": datetime.utcnow().isoformat(),
            "n_samples": len(X_train),
            "test_samples": len(X_test)
        }
        
        # Store training history
        self.training_history.append(self.metrics)
        self.last_training_date = datetime.utcnow()
        
        return self.metrics
    
    def update_model(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """Update the model with new data (continuous learning)."""
        if self.model is None:
            return self.train(new_data)
        
        # Prepare new data
        X_new, y_new = self.prepare_data(new_data)
        
        # Update model with new data
        self.model.fit(X_new, y_new)
        
        # Evaluate on new data
        y_pred = self.model.predict(X_new)
        y_pred_proba = self.model.predict_proba(X_new)[:, 1]
        
        # Calculate metrics for new data
        new_metrics = {
            "accuracy": accuracy_score(y_new, y_pred),
            "precision": precision_score(y_new, y_pred),
            "recall": recall_score(y_new, y_pred),
            "f1_score": f1_score(y_new, y_pred),
            "roc_auc": roc_auc_score(y_new, y_pred_proba),
            "confusion_matrix": confusion_matrix(y_new, y_pred).tolist(),
            "training_date": datetime.utcnow().isoformat(),
            "n_samples": len(X_new),
            "update_type": "incremental"
        }
        
        # Update metrics and history
        self.metrics = new_metrics
        self.training_history.append(new_metrics)
        self.last_training_date = datetime.utcnow()
        
        return new_metrics
    
    def save_model(self, path: str):
        """Save the trained model and its metadata."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            "model": self.model,
            "feature_importance": self.feature_importance,
            "metrics": self.metrics,
            "training_history": self.training_history,
            "last_training_date": self.last_training_date,
            "model_type": self.model_type
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str) -> "ModelTrainer":
        """Load a trained model and its metadata."""
        model_data = joblib.load(path)
        
        trainer = cls(model_data["model_type"])
        trainer.model = model_data["model"]
        trainer.feature_importance = model_data["feature_importance"]
        trainer.metrics = model_data["metrics"]
        trainer.training_history = model_data.get("training_history", [])
        trainer.last_training_date = model_data.get("last_training_date")
        
        return trainer

def train_models(n_samples: int = 200000) -> Dict[str, Dict[str, Any]]:
    """Train both risk and fraud models using realistic data."""
    logger.info(f"Generating {n_samples} samples for training")
    
    # Generate training data
    data_generator = LoanDataGenerator()
    data = data_generator.generate_training_dataset(n_samples)
    
    # Train risk model
    logger.info("Training risk model...")
    risk_trainer = ModelTrainer("risk")
    risk_metrics = risk_trainer.train(data, batch_size=10000)
    risk_trainer.save_model("models/risk_model.joblib")
    
    # Train fraud model
    logger.info("Training fraud model...")
    fraud_trainer = ModelTrainer("fraud")
    fraud_metrics = fraud_trainer.train(data, batch_size=10000)
    fraud_trainer.save_model("models/fraud_model.joblib")
    
    return {
        "risk_model": risk_metrics,
        "fraud_model": fraud_metrics
    } 