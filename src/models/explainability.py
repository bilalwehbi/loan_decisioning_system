# src/models/explainability.py
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union
import shap
from datetime import datetime
import json
from src.models.feature_mapping import get_display_name

logger = logging.getLogger(__name__)

class ExplainabilityGenerator:
    """
    Generates explanations for model predictions using SHAP values
    and provides compliance-ready documentation
    """
    
    def __init__(self):
        self.explanation_store = {}
        self.feature_descriptions = {
            # Credit features
            "credit_score": "FICO credit score (300-850)",
            "delinquencies": "Number of delinquent accounts",
            "inquiries_last_6m": "Credit inquiries in last 6 months",
            "tradelines": "Number of active credit accounts",
            "utilization": "Credit utilization ratio",
            "payment_history_score": "Payment history quality score",
            "credit_age_months": "Average age of credit accounts",
            "credit_mix_score": "Credit mix diversity score",
            
            # Banking features
            "avg_monthly_income": "Average monthly income",
            "income_stability_score": "Income stability measure",
            "spending_pattern_score": "Spending behavior score",
            "transaction_count": "Monthly transaction volume",
            "avg_account_balance": "Average account balance",
            "overdraft_frequency": "Overdraft occurrences",
            "savings_rate": "Monthly savings rate",
            
            # Behavioral features
            "device_trust_score": "Device trustworthiness score",
            "location_risk_score": "Location risk assessment",
            "digital_footprint_score": "Digital presence score",
            "application_completion_time": "Time to complete application",
            
            # Employment features
            "employment_length_months": "Length of employment",
            "employment_verification_score": "Employment verification confidence",
            "income_verification_score": "Income verification confidence"
        }
    
    def explain_credit_risk_prediction(self, model: Any, pipeline: Any, 
                                     application_df: pd.DataFrame,
                                     feature_names: List[str]) -> List[Dict[str, Any]]:
        """
        Generate SHAP-based explanations for credit risk prediction
        
        Args:
            model: Trained credit risk model
            pipeline: Preprocessing pipeline
            application_df: Application data
            feature_names: List of feature names
            
        Returns:
            List of explanation dictionaries
        """
        try:
            # Generate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(pipeline.transform(application_df))
            
            # Get feature importance
            feature_importance = np.abs(shap_values).mean(0)
            
            # Create explanations
            explanations = []
            for i, (feature, importance) in enumerate(zip(feature_names, feature_importance)):
                if importance > 0.01:  # Only include significant features
                    explanations.append({
                        "feature": feature,
                        "description": self.feature_descriptions.get(feature, "Unknown feature"),
                        "contribution": float(importance),
                        "direction": "positive" if shap_values[0][i] > 0 else "negative",
                        "value": float(application_df[feature].iloc[0]),
                        "impact": self._get_impact_description(feature, application_df[feature].iloc[0])
                    })
            
            # Sort by contribution
            explanations.sort(key=lambda x: x["contribution"], reverse=True)
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error generating credit risk explanations: {str(e)}")
            return []
    
    def explain_fraud_prediction(self, model: Any, pipeline: Any,
                               application_df: pd.DataFrame,
                               feature_names: List[str]) -> List[Dict[str, Any]]:
        """
        Generate SHAP-based explanations for fraud prediction
        
        Args:
            model: Trained fraud detection model
            pipeline: Preprocessing pipeline
            application_df: Application data
            feature_names: List of feature names
            
        Returns:
            List of explanation dictionaries
        """
        try:
            # Generate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(pipeline.transform(application_df))
            
            # Get feature importance
            feature_importance = np.abs(shap_values).mean(0)
            
            # Create explanations
            explanations = []
            for i, (feature, importance) in enumerate(zip(feature_names, feature_importance)):
                if importance > 0.01:  # Only include significant features
                    explanations.append({
                        "feature": feature,
                        "description": self.feature_descriptions.get(feature, "Unknown feature"),
                        "contribution": float(importance),
                        "direction": "positive" if shap_values[0][i] > 0 else "negative",
                        "value": float(application_df[feature].iloc[0]),
                        "risk_level": self._get_risk_level(feature, application_df[feature].iloc[0])
                    })
            
            # Sort by contribution
            explanations.sort(key=lambda x: x["contribution"], reverse=True)
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error generating fraud explanations: {str(e)}")
            return []
    
    def store_explanation_data(self, application_id: str, explanations: List[Dict[str, Any]]):
        """
        Store explanation data for compliance and audit purposes
        
        Args:
            application_id: Unique application identifier
            explanations: List of explanation dictionaries
        """
        try:
            explanation_record = {
                "application_id": application_id,
                "timestamp": datetime.utcnow().isoformat(),
                "explanations": explanations
            }
            
            # Store in memory
            self.explanation_store[application_id] = explanation_record
            
            # Store to disk
            os.makedirs("data/explanations", exist_ok=True)
            with open(f"data/explanations/{application_id}.json", "w") as f:
                json.dump(explanation_record, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error storing explanation data: {str(e)}")
    
    def get_explanation_data(self, application_id: str) -> Dict[str, Any]:
        """
        Retrieve stored explanation data
        
        Args:
            application_id: Unique application identifier
            
        Returns:
            Explanation data dictionary
        """
        try:
            # Try memory first
            if application_id in self.explanation_store:
                return self.explanation_store[application_id]
            
            # Try disk
            explanation_path = f"data/explanations/{application_id}.json"
            if os.path.exists(explanation_path):
                with open(explanation_path, "r") as f:
                    return json.load(f)
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving explanation data: {str(e)}")
            return None
    
    def _get_impact_description(self, feature: str, value: float) -> str:
        """Generate human-readable impact description for a feature value."""
        if feature == "credit_score":
            if value >= 750:
                return "Excellent credit score"
            elif value >= 700:
                return "Good credit score"
            elif value >= 650:
                return "Fair credit score"
            else:
                return "Poor credit score"
        elif feature == "utilization":
            if value <= 0.3:
                return "Low credit utilization"
            elif value <= 0.5:
                return "Moderate credit utilization"
            else:
                return "High credit utilization"
        elif feature == "income_stability_score":
            if value >= 0.8:
                return "Very stable income"
            elif value >= 0.6:
                return "Stable income"
            else:
                return "Unstable income"
        else:
            return "Feature impact"
    
    def _get_risk_level(self, feature: str, value: float) -> str:
        """Generate risk level description for a feature value."""
        if feature in ["device_trust_score", "employment_verification_score", "income_verification_score"]:
            if value >= 0.8:
                return "Low Risk"
            elif value >= 0.5:
                return "Medium Risk"
            else:
                return "High Risk"
        elif feature == "location_risk_score":
            if value <= 0.3:
                return "Low Risk"
            elif value <= 0.6:
                return "Medium Risk"
            else:
                return "High Risk"
        else:
            return "Unknown Risk Level"