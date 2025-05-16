import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import logging
import shap

logger = logging.getLogger(__name__)

class RiskScorer:
    def __init__(self):
        self.model = None
        self.pipeline = None
        self.feature_names = None
        self.explainer = None
        
    def _prepare_features(self, data):
        """Prepare features for model training or prediction."""
        # Define numeric and categorical features
        numeric_features = [
            'loan_amount', 'loan_term', 'credit_score', 'delinquencies',
            'inquiries_last_6m', 'tradelines', 'utilization', 'average_balance',
            'income', 'employment_length', 'avg_monthly_income',
            'income_stability_score', 'spending_pattern_score', 'transaction_count'
        ]
        
        categorical_features = ['purpose', 'has_co_applicant']
        
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
        """Train the risk scoring model."""
        # Prepare features
        preprocessor, feature_names = self._prepare_features(train_data)
        self.feature_names = feature_names
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ))
        ])
        
        # Prepare data
        X_train = train_data.drop(['default', 'fraud'], axis=1)
        y_train = train_data['default']
        X_test = test_data.drop(['default', 'fraud'], axis=1)
        y_test = test_data['default']
        
        # Train model
        self.pipeline.fit(X_train, y_train)
        self.model = self.pipeline.named_steps['classifier']
        
        # Create SHAP explainer
        X_train_processed = self.pipeline.named_steps['preprocessor'].transform(X_train)
        self.explainer = shap.TreeExplainer(self.model)
        
        # Calculate metrics
        train_score = self.pipeline.score(X_train, y_train)
        test_score = self.pipeline.score(X_test, y_test)
        
        metrics = {
            'train_accuracy': train_score,
            'test_accuracy': test_score
        }
        
        return metrics
    
    def predict(self, application_data):
        """Make risk prediction for a loan application."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Convert to DataFrame if dict
        if isinstance(application_data, dict):
            df = pd.DataFrame([application_data])
        else:
            df = application_data
        
        # Get prediction
        default_prob = self.pipeline.predict_proba(df)[0, 1]
        
        # Convert to risk score (300-850)
        risk_score = int(300 + (1 - default_prob) * 550)
        
        # Determine risk segment
        if risk_score >= 750:
            risk_segment = "Low Risk"
        elif risk_score >= 600:
            risk_segment = "Review"
        else:
            risk_segment = "High Risk"
        
        return risk_score, default_prob, risk_segment
    
    def explain_prediction(self, application_data):
        """Generate SHAP-based explanation for a prediction."""
        if self.explainer is None:
            return {
                'top_factors': {},
                'explanations': {"message": "Model explainer not available"}
            }
        
        # Convert to DataFrame if dict
        if isinstance(application_data, dict):
            df = pd.DataFrame([application_data])
        else:
            df = application_data
        
        # Get SHAP values
        X_processed = self.pipeline.named_steps['preprocessor'].transform(df)
        shap_values = self.explainer.shap_values(X_processed)
        
        # Get feature names after preprocessing
        feature_names = self.pipeline.named_steps['preprocessor'].get_feature_names_out()
        
        # Get top factors
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(shap_values[0])
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        # Generate explanations
        top_factors = {}
        for _, row in feature_importance.head(5).iterrows():
            feature = row['feature']
            importance = row['importance']
            top_factors[feature] = float(importance)
        
        explanations = {
            'top_factors': top_factors,
            'explanations': {
                'message': "Risk assessment based on credit profile and application data"
            }
        }
        
        return explanations
    
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
        
        # Recreate SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        logger.info(f"Model loaded from {path}") 