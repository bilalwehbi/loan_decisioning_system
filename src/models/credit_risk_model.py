# src/models/credit_risk_model.py
import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union
import uuid
from datetime import datetime

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.data.preprocessing import create_credit_risk_pipeline, preprocess_loan_application
from src.configs.config import MODEL_CONFIG, MODEL_DIR
from src.models.explainability import ExplainabilityGenerator

logger = logging.getLogger(__name__)

class CreditRiskModel:
    """
    Credit Risk Scoring model to predict probability of default and assign credit scores
    """
    
    def __init__(self):
        """Initialize the model with default values"""
        self.model = None
        self.pipeline = create_credit_risk_pipeline()
        self.model_config = MODEL_CONFIG['credit_risk']
        self.thresholds = self.model_config['threshold']
        
        # For tracking model metadata
        self.model_id = f"credit_risk_model_{uuid.uuid4().hex[:8]}"
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
        
        # Initialize a default model
        self.model = lgb.LGBMClassifier(**self.model_config['params'])
        
    def _convert_to_credit_score(self, default_prob: float) -> int:
        """
        Convert default probability to a credit score from 0-1000
        Lower default probability = higher score
        
        Args:
            default_prob: Probability of default between 0 and 1
            
        Returns:
            credit_score: Integer score between 0 and 1000
        """
        # Invert the probability (higher is better)
        inverted_prob = 1 - default_prob
        
        # Scale between 0 and 1000
        credit_score = int(inverted_prob * 1000)
        
        return credit_score
    
    def _get_decision_segment(self, credit_score: int) -> str:
        """
        Determine decision segment based on credit score
        
        Args:
            credit_score: Integer score between 0 and 1000
            
        Returns:
            segment: Decision segment (Low Risk, Review, or Decline)
        """
        if credit_score >= self.thresholds['low_risk']:
            return "Low Risk"
        elif credit_score >= self.thresholds['review']:
            return "Review"
        else:
            return "Decline"
    
    def _create_feature_name_mapping(self, X: pd.DataFrame, pipeline):
        """
        Create mapping between transformed feature indices and original feature names
        
        Args:
            X: Original feature DataFrame
            pipeline: Fitted preprocessing pipeline
            
        Returns:
            None (updates self.feature_name_mapping)
        """
        # Store original feature names
        self.original_feature_names = X.columns.tolist()
        
        # Define feature categories from the pipeline
        # These should match the categories in create_credit_risk_pipeline()
        numeric_features = [
            'credit_score', 'num_accounts', 'num_active_accounts', 'credit_utilization',
            'num_delinquencies_30d', 'num_delinquencies_60d', 'num_delinquencies_90d', 
            'num_collections', 'total_debt', 'inquiries_last_6mo', 'longest_credit_length_months',
            'avg_monthly_deposits', 'avg_monthly_withdrawals', 'monthly_income',
            'income_stability_score', 'num_income_sources', 'avg_daily_balance',
            'num_nsf_transactions', 'num_overdrafts',
            'time_spent_on_application', 'num_previous_applications',
            'loan_amount', 'loan_term_months', 'applicant_income', 'applicant_employment_length_years'
        ]
        
        categorical_features = [
            'loan_purpose', 'email_domain'
        ]
        
        # Filter to features that exist in the input data
        numeric_features = [f for f in numeric_features if f in X.columns]
        categorical_features = [f for f in categorical_features if f in X.columns]
        
        # Create feature name mapping
        feature_mapping = {}
        
        # Add numeric features (these stay mostly the same, might be transformed)
        for i, feature in enumerate(numeric_features):
            feature_mapping[i] = f"numeric_{feature}"
            
        # Add engineered features from CustomFeatureGenerator
        engineered_features = [
            'dti_ratio', 'payment_to_income', 'income_verification_ratio',
            'high_utilization', 'has_delinquencies', 'total_banking_negatives',
            'loan_to_income', 'credit_score_category'
        ]
        
        offset = len(numeric_features)
        for i, feature in enumerate(engineered_features):
            feature_mapping[offset + i] = f"engineered_{feature}"
            
        # Add categorical features (these expand with one-hot encoding)
        # Need to determine how many columns each categorical feature expands to
        # This is complex to determine precisely without fitting the pipeline
        # For simplicity, we'll use an approximation
        
        offset = len(numeric_features) + len(engineered_features)
        for feature in categorical_features:
            if feature == 'loan_purpose':
                # Approximate number of categories for loan_purpose
                loan_purposes = ["Debt consolidation", "Home improvement", "Medical expenses", 
                                "Education", "Major purchase", "Vehicle", "Other"]
                for i, purpose in enumerate(loan_purposes):
                    feature_mapping[offset + i] = f"categorical_{feature}_{purpose}"
                offset += len(loan_purposes)
            elif feature == 'email_domain':
                # Approximate number of categories for email_domain
                email_domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com", "other"]
                for i, domain in enumerate(email_domains):
                    feature_mapping[offset + i] = f"categorical_{feature}_{domain}"
                offset += len(email_domains)
        
        self.feature_name_mapping = feature_mapping
        logger.info(f"Created feature mapping with {len(feature_mapping)} features")
        
    def _get_mapped_feature_name(self, index: int) -> str:
        """Get mapped feature name for a given index"""
        if index in self.feature_name_mapping:
            return self.feature_name_mapping[index]
        else:
            return f"feature_{index}"
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Train the credit risk model
        
        Args:
            X_train: Training feature DataFrame
            y_train: Training target Series (1 = default, 0 = repaid)
            X_test: Test feature DataFrame
            y_test: Test target Series
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        logger.info(f"Training credit risk model {self.model_id}")
        
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
            eval_set=[(X_test_processed, y_test)],
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
        Generate credit risk prediction for a loan application
        
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
            default_prob = self.model.predict_proba(X_processed)[0, 1]
            
            # Convert to credit score
            credit_score = self._convert_to_credit_score(default_prob)
            
            # Get decision segment
            decision = self._get_decision_segment(credit_score)

            # Generate top_factors using SHAP values
            explain_gen = ExplainabilityGenerator()
            explanations = explain_gen.explain_credit_risk_prediction(
                self.model, self.pipeline, df, self.feature_names
            )
            # Extract top_factors (top 5 by contribution)
            top_factors = [
                {e['feature']: e['contribution']} for e in explanations[:5]
            ] if explanations else []
            
            # Return prediction result
            result = {
                'credit_score': credit_score,
                'probability_of_default': float(default_prob),
                'decision': decision,
                'top_factors': top_factors
            }
            
            return result
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            # Return default values if prediction fails
            return {
                'credit_score': 500,  # Middle of the range
                'probability_of_default': 0.5,
                'decision': 'Review',
                'top_factors': []
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
            'thresholds': self.thresholds
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
                self.thresholds = model_data.get('thresholds', self.thresholds)
            else:
                self.model = model_data
                
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model from {filepath}: {str(e)}")
            # Initialize a default model if loading fails
            self.model = lgb.LGBMClassifier(**self.model_config['params'])
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
            'feature_importance': self.model.feature_importances_.tolist() if self.model else [],
            'thresholds': self.thresholds,
            'model_id': self.model_id,
            'training_date': self.training_date.isoformat() if self.training_date else None
        }