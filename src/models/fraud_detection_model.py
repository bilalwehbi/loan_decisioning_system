# src/models/fraud_detection_model.py
import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union
import uuid
from datetime import datetime

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

from src.data.preprocessing import create_fraud_detection_pipeline, preprocess_loan_application
from src.configs.config import MODEL_CONFIG, MODEL_DIR

logger = logging.getLogger(__name__)

class FraudDetectionModel:
    """
    Fraud Detection model to identify potentially fraudulent loan applications
    """
    
    def __init__(self):
        """Initialize the model with default values"""
        self.model = None
        self.pipeline = create_fraud_detection_pipeline()
        self.model_config = MODEL_CONFIG['fraud_detection']
        self.threshold = 0.5  # Adjusted threshold for testing
        
        # For tracking model metadata
        self.model_id = f"fraud_detection_model_{uuid.uuid4().hex[:8]}"
        self.feature_names = []
        self.original_feature_names = []
        self.feature_name_mapping = {}  # Map from transformed to original features
        self.training_date = None
        self.model_metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'training_data_size': 0
        }
        self.fraud_tags = {
            'synthetic_id': "Synthetic identity suspected",
            'income_mismatch': "Income verification failed",
            'velocity': "Multiple applications detected",
            'suspicious_behavior': "Suspicious application behavior",
            'credit_identity_mismatch': "Credit and application identity mismatch",
            'identity_fraud': "Identity fraud suspected"
        }
        
        # Initialize a simpler model (e.g., Logistic Regression)
        self.model = LogisticRegression(max_iter=1000)  # Use a simpler model
    
    def _create_feature_name_mapping(self, X, pipeline):
        """
        Create mapping between transformed feature indices and original feature names
        
        Args:
            X: Original feature DataFrame or NumPy array
            pipeline: Fitted preprocessing pipeline
            
        Returns:
            None (updates self.feature_name_mapping)
        """
        # Store original feature names
        if isinstance(X, pd.DataFrame):
            self.original_feature_names = X.columns.tolist()
        else:
            self.original_feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # Only use the columns present in the generated data
        numeric_features = [
            'hour_of_day',
            'day_of_week',
            'device_id_length',
            'email_domain_length',
            'phone_number_length',
            'income_verification_ratio',
            'num_previous_applications',
            'time_spent_on_application'
        ]

        feature_mapping = {}
        for i, feature in enumerate(numeric_features):
            feature_mapping[i] = f"numeric_{feature}"

        self.feature_name_mapping = feature_mapping
        logger.info(f"Created feature mapping with {len(feature_mapping)} features")
    
    def _get_mapped_feature_name(self, index: int) -> str:
        """Get mapped feature name for a given index"""
        if index in self.feature_name_mapping:
            return self.feature_name_mapping[index]
        else:
            return f"feature_{index}"
    
    def _get_fraud_tags(self, features: pd.DataFrame, fraud_score: float) -> List[str]:
        """
        Generate specific fraud tags based on feature values and fraud score
        
        Args:
            features: Preprocessed feature DataFrame
            fraud_score: Model's fraud probability score
            
        Returns:
            tags: List of applicable fraud tags
        """
        tags = []
        
        # Only apply tags if fraud score is above threshold
        if fraud_score < self.threshold:
            return tags
            
        # Check for identity fraud
        if 'email_domain' in features.columns:
            if features['email_domain'].iloc[0] in ['gmail.com', 'yahoo.com']:
                tags.append(self.fraud_tags['identity_fraud'])
        
        # Check for income misrepresentation
        if 'income_verification_ratio' in features.columns:
            ratio = features['income_verification_ratio'].iloc[0]
            if ratio > 2.0:
                tags.append(self.fraud_tags['income_mismatch'])
        
        # Check for synthetic ID
        if 'num_previous_applications' in features.columns:
            prev_apps = features['num_previous_applications'].iloc[0]
            if prev_apps >= 3:
                tags.append(self.fraud_tags['synthetic_id'])
        
        # Check for suspicious application behavior
        if 'time_spent_on_application' in features.columns:
            time_spent = features['time_spent_on_application'].iloc[0]
            if time_spent < 120:
                tags.append(self.fraud_tags['suspicious_behavior'])
        
        return tags
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Train the fraud detection model
        
        Args:
            X_train: Training feature DataFrame
            y_train: Training target Series (1 = fraud, 0 = legitimate)
            X_test: Test feature DataFrame
            y_test: Test target Series
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        logger.info(f"Training fraud detection model {self.model_id}")
        
        # Create feature name mapping
        self._create_feature_name_mapping(X_train, self.pipeline)
        
        logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
        
        # Fit preprocessing pipeline
        X_train_processed = self.pipeline.fit_transform(X_train)
        X_test_processed = self.pipeline.transform(X_test)
        
        # Get feature names from pipeline if possible
        try:
            if hasattr(self.pipeline, 'get_feature_names_out'):
                # For scikit-learn >= 1.0
                self.feature_names = list(self.pipeline.get_feature_names_out())
            else:
                # Fallback to indices with our mapping
                self.feature_names = [self._get_mapped_feature_name(i) for i in range(X_train_processed.shape[1])]
        except Exception as e:
            logger.warning(f"Could not extract feature names from pipeline: {str(e)}")
            self.feature_names = [self._get_mapped_feature_name(i) for i in range(X_train_processed.shape[1])]
            
        logger.info(f"Model will use {len(self.feature_names)} features after preprocessing")
        
        # Train model
        self.model.fit(
            X_train_processed,
            y_train,
            **self.model_config['training_params']
        )
        
        # Make predictions
        y_pred = self.model.predict(X_test_processed)
        y_pred_proba = self.model.predict_proba(X_test_processed)[:, 1]
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel() if confusion_matrix(y_test, y_pred).size == 4 else (0,0,0,0)
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'training_data_size': len(X_train),
            'false_positive_rate': false_positive_rate
        }
        
        # Store metrics
        self.model_metrics = metrics
        self.training_date = datetime.now()
        
        logger.info(f"Model training completed. Metrics: {metrics}")
        
        return metrics
    
    def predict(self, loan_application: Union[Dict, pd.DataFrame]) -> Dict:
        """
        Generate fraud detection prediction for a loan application
        
        Args:
            loan_application: Loan application data as dict or DataFrame
            
        Returns:
            result: Dict with prediction results
        """
        try:
            # Preprocess the application
            if isinstance(loan_application, dict):
                df = preprocess_loan_application(loan_application)
            else:
                df = loan_application
                
            # Apply preprocessing pipeline
            X_processed = self.pipeline.transform(df)
            
            # Get prediction
            fraud_prob = self.model.predict_proba(X_processed)[0, 1]
            
            # Get fraud tags
            fraud_tags = self._get_fraud_tags(df, fraud_prob)
            
            # Return prediction result
            result = {
                'fraud_score': float(fraud_prob),
                'fraud_flags': fraud_tags
            }
            
            return result
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            # Return default values if prediction fails
            return {
                'fraud_score': 0.0,
                'fraud_flags': []
            }
    
    def save(self, filepath: str = None) -> str:
        """
        Save the trained model to disk
        
        Args:
            filepath: Path to save the model to. If None, a default path is used.
            
        Returns:
            filepath: Path where the model was saved
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() before save().")
        
        if filepath is None:
            os.makedirs(MODEL_DIR, exist_ok=True)
            filepath = os.path.join(MODEL_DIR, f"{self.model_id}.pkl")
            
        # Create a dictionary with all model components
        model_data = {
            'model': self.model,
            'pipeline': self.pipeline,
            'model_id': self.model_id,
            'feature_names': self.feature_names,
            'original_feature_names': self.original_feature_names,
            'feature_name_mapping': self.feature_name_mapping,
            'training_date': self.training_date,
            'model_metrics': self.model_metrics,
            'threshold': self.threshold,
            'fraud_tags': self.fraud_tags
        }
        
        # Save to disk
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"Model saved to {filepath}")
        
        return filepath
    
    def load(self, filepath: str) -> None:
        """
        Load a trained model from disk
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            None
        """
        try:
            if filepath.endswith('.joblib'):
                import joblib
                model_data = joblib.load(filepath)
            else:
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                
            if isinstance(model_data, dict):
                self.model = model_data.get('model', self.model)
                self.pipeline = model_data.get('pipeline', self.pipeline)
                self.model_id = model_data.get('model_id', self.model_id)
                self.feature_names = model_data.get('feature_names', self.feature_names)
                self.original_feature_names = model_data.get('original_feature_names', self.original_feature_names)
                self.feature_name_mapping = model_data.get('feature_name_mapping', self.feature_name_mapping)
                self.training_date = model_data.get('training_date', self.training_date)
                self.model_metrics = model_data.get('model_metrics', self.model_metrics)
                self.threshold = model_data.get('threshold', self.threshold)
                self.fraud_tags = model_data.get('fraud_tags', self.fraud_tags)
            else:
                self.model = model_data
                
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model from {filepath}: {str(e)}")
            # Initialize a default model if loading fails
            self.model = LogisticRegression(max_iter=1000)  # Use a simpler model
            logger.warning("Using default model due to loading error")

    def get_accuracy(self) -> float:
        """Get model accuracy"""
        return self.model_metrics.get('accuracy', 0.0)

    def get_precision(self) -> float:
        """Get model precision"""
        return self.model_metrics.get('precision', 0.0)

    def get_recall(self) -> float:
        """Get model recall"""
        return self.model_metrics.get('recall', 0.0)

    def get_f1_score(self) -> float:
        """Get model F1 score"""
        return self.model_metrics.get('f1_score', 0.0)

    def get_training_data_size(self) -> int:
        """Get size of training data"""
        return self.model_metrics.get('training_data_size', 0)

    def get_performance_metrics(self) -> Dict:
        """Get additional performance metrics"""
        return {
            'feature_importance': self.model.coef_[0].tolist() if self.model else [],
            'threshold': self.threshold,
            'model_id': self.model_id,
            'training_date': self.training_date.isoformat() if self.training_date else None
        }

    def detect_synthetic_id(self, application_data: dict) -> dict:
        """
        Detect potential synthetic ID fraud using multiple signals
        
        Args:
            application_data: Loan application data with identity information
            
        Returns:
            Dictionary containing synthetic ID risk score and signals
        """
        try:
            # Extract identity signals
            identity_signals = self._extract_identity_signals(application_data)
            
            # Check for velocity patterns
            velocity_risk = self._check_velocity_patterns(application_data)
            
            # Check for identity consistency
            consistency_risk = self._check_identity_consistency(identity_signals)
            
            # Check for synthetic patterns
            synthetic_patterns = self._check_synthetic_patterns(identity_signals)
            
            # Calculate final synthetic ID risk score
            risk_score = self._calculate_synthetic_id_risk(
                velocity_risk,
                consistency_risk,
                synthetic_patterns
            )
            
            return {
                "synthetic_id_risk_score": risk_score,
                "risk_level": self._get_risk_level(risk_score),
                "signals": {
                    "velocity_risk": velocity_risk,
                    "consistency_risk": consistency_risk,
                    "synthetic_patterns": synthetic_patterns
                },
                "recommendation": self._get_synthetic_id_recommendation(risk_score)
            }
        except Exception as e:
            logger.error(f"Error detecting synthetic ID: {str(e)}")
            raise

    def validate_income(self, application_data: dict) -> dict:
        """
        Validate income claims using multiple data sources
        
        Args:
            application_data: Loan application data with income information
            
        Returns:
            Dictionary containing income validation results
        """
        try:
            # Extract income information
            stated_income = application_data.get("stated_income", {})
            bank_transactions = application_data.get("bank_transactions", [])
            employment_data = application_data.get("employment_data", {})
            
            # Validate using bank transactions
            bank_validation = self._validate_income_from_bank_transactions(
                stated_income,
                bank_transactions
            )
            
            # Validate using employment data
            employment_validation = self._validate_income_from_employment(
                stated_income,
                employment_data
            )
            
            # Check for income consistency
            consistency_check = self._check_income_consistency(
                bank_validation,
                employment_validation
            )
            
            # Calculate final validation score
            validation_score = self._calculate_income_validation_score(
                bank_validation,
                employment_validation,
                consistency_check
            )
            
            return {
                "validation_score": validation_score,
                "validation_status": self._get_validation_status(validation_score),
                "bank_validation": bank_validation,
                "employment_validation": employment_validation,
                "consistency_check": consistency_check,
                "recommendation": self._get_income_validation_recommendation(validation_score)
            }
        except Exception as e:
            logger.error(f"Error validating income: {str(e)}")
            raise

    def _extract_identity_signals(self, application_data: dict) -> dict:
        """Extract identity-related signals from application data"""
        signals = {
            "email": self._analyze_email(application_data.get("email", "")),
            "phone": self._analyze_phone(application_data.get("phone", "")),
            "address": self._analyze_address(application_data.get("address", {})),
            "device": self._analyze_device(application_data.get("device_metadata", {})),
            "ip": self._analyze_ip(application_data.get("ip_address", ""))
        }
        return signals

    def _check_velocity_patterns(self, application_data: dict) -> dict:
        """Check for suspicious velocity patterns in application data"""
        try:
            # Get historical applications
            historical_apps = self._get_historical_applications(application_data)
            
            # Check application velocity
            app_velocity = self._calculate_application_velocity(historical_apps)
            
            # Check IP velocity
            ip_velocity = self._calculate_ip_velocity(historical_apps)
            
            # Check device velocity
            device_velocity = self._calculate_device_velocity(historical_apps)
            
            return {
                "application_velocity": app_velocity,
                "ip_velocity": ip_velocity,
                "device_velocity": device_velocity,
                "risk_score": self._calculate_velocity_risk_score(
                    app_velocity,
                    ip_velocity,
                    device_velocity
                )
            }
        except Exception as e:
            logger.error(f"Error checking velocity patterns: {str(e)}")
            raise

    def _validate_income_from_bank_transactions(self, stated_income: dict, transactions: list) -> dict:
        """Validate income using bank transaction data"""
        try:
            # Calculate average monthly income
            monthly_income = self._calculate_monthly_income(transactions)
            
            # Check income stability
            stability_score = self._calculate_income_stability(transactions)
            
            # Compare with stated income
            income_comparison = self._compare_with_stated_income(
                monthly_income,
                stated_income
            )
            
            return {
                "calculated_monthly_income": monthly_income,
                "income_stability": stability_score,
                "income_comparison": income_comparison,
                "validation_score": self._calculate_bank_validation_score(
                    monthly_income,
                    stability_score,
                    income_comparison
                )
            }
        except Exception as e:
            logger.error(f"Error validating income from bank transactions: {str(e)}")
            raise

    def _validate_income_from_employment(self, stated_income: dict, employment_data: dict) -> dict:
        """Validate income using employment data"""
        try:
            # Verify employment status
            employment_status = self._verify_employment_status(employment_data)
            
            # Calculate expected income
            expected_income = self._calculate_expected_income(employment_data)
            
            # Compare with stated income
            income_comparison = self._compare_with_stated_income(
                expected_income,
                stated_income
            )
            
            return {
                "employment_status": employment_status,
                "expected_income": expected_income,
                "income_comparison": income_comparison,
                "validation_score": self._calculate_employment_validation_score(
                    employment_status,
                    expected_income,
                    income_comparison
                )
            }
        except Exception as e:
            logger.error(f"Error validating income from employment: {str(e)}")
            raise

    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Args:
            X: Feature DataFrame or NumPy array
            
        Returns:
            probabilities: Array of shape (n_samples, n_classes) containing the class probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() before predict_proba().")
        
        # Preprocess the input
        X_processed = self.pipeline.transform(X)
        
        # Get prediction probabilities
        return self.model.predict_proba(X_processed)