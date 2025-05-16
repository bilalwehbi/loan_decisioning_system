from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import json
import uuid
from fastapi import HTTPException
from .models import LoanApplication

logger = logging.getLogger(__name__)

class LoanDecisioningService:
    def __init__(self, risk_model_path: str, fraud_model_path: str):
        """
        Initialize the loan decisioning service with risk and fraud models.
        
        Args:
            risk_model_path: Path to the risk assessment model
            fraud_model_path: Path to the fraud detection model
        """
        self.risk_model_path = risk_model_path
        self.fraud_model_path = fraud_model_path
        self.risk_model = self._load_model(risk_model_path)
        self.fraud_model = self._load_model(fraud_model_path)
        self.metrics_history = self._load_metrics_history()
        self.application_history = self._load_application_history()
        
        if self.risk_model is None or self.fraud_model is None:
            raise ValueError("Failed to load one or more models. Please check the model files and paths.")
        
    def _load_model(self, model_path: str) -> Any:
        """Load a model from disk."""
        try:
            model_data = joblib.load(model_path)
            # If it's a dict with a 'model' key, extract the model
            if isinstance(model_data, dict) and 'model' in model_data:
                return model_data['model']
            return model_data
        except FileNotFoundError:
            logger.warning(f"Model file not found: {model_path}. The model will need to be trained.")
            return None
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            return None
    
    def _load_metrics_history(self) -> List[Dict]:
        """Load model metrics history from disk."""
        metrics_path = "data/metrics_history.json"
        try:
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading metrics history: {str(e)}")
            return []
    
    def _load_application_history(self) -> List[Dict]:
        """Load application history from disk."""
        history_path = "data/application_history.json"
        try:
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading application history: {str(e)}")
            return []
    
    def _save_metrics_history(self):
        """Save model metrics history to disk."""
        metrics_path = "data/metrics_history.json"
        try:
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics_history, f)
        except Exception as e:
            logger.error(f"Error saving metrics history: {str(e)}")
    
    def _save_application_history(self):
        """Save application history to disk."""
        history_path = "data/application_history.json"
        try:
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            with open(history_path, 'w') as f:
                json.dump(self.application_history, f)
        except Exception as e:
            logger.error(f"Error saving application history: {str(e)}")
    
    def process_application(self, application: LoanApplication) -> Dict[str, Any]:
        """Process a loan application and return assessment results."""
        try:
            # Generate application ID if not provided
            if not application.application_id:
                application.application_id = str(uuid.uuid4())
            
            # Get risk assessment
            risk_score, risk_explanation = self.assess_risk(application)
            risk_assessment = {
                "score": risk_score,
                "probability_default": 1 - (risk_score / 1000),  # Convert score to probability
                "risk_segment": self._get_risk_segment(risk_score),
                "top_factors": self._get_risk_factors(application),
                "explanation": risk_explanation
            }
            
            # Get fraud assessment
            fraud_score, fraud_flags = self.assess_fraud(application)
            fraud_assessment = {
                "score": fraud_score,
                "flags": fraud_flags,
                "explanation": self._get_fraud_explanation(fraud_flags)
            }
            
            # Make final decision
            decision = self._make_decision(risk_score, fraud_score, fraud_flags)
            
            # Prepare response
            response = {
                "application_id": application.application_id,
                "risk_assessment": risk_assessment,
                "fraud_assessment": fraud_assessment,
                "final_decision": decision,
                "risk_score": risk_score,
                "fraud_score": fraud_score,
                "fraud_flags": fraud_flags,
                "explanations": [risk_explanation] + [f"Fraud flag: {flag}" for flag in fraud_flags]
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing application: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing application: {str(e)}"
            )

    def _get_risk_segment(self, risk_score: float) -> str:
        """Get risk segment based on risk score."""
        if risk_score >= 800:
            return "Low Risk"
        elif risk_score >= 650:
            return "Medium Risk"
        elif risk_score >= 500:
            return "High Risk"
        else:
            return "Very High Risk"

    def _get_risk_factors(self, application: LoanApplication) -> List[str]:
        """Get top risk factors for the application."""
        factors = []
        if application.credit_data.credit_score < 650:
            factors.append("Low credit score")
        if application.applicant_income < 50000:
            factors.append("Low income")
        if application.applicant_employment_length_years < 2:
            factors.append("Short employment history")
        return factors[:3]  # Return top 3 factors

    def _get_fraud_explanation(self, fraud_flags: List[str]) -> str:
        """Generate explanation for fraud flags."""
        if not fraud_flags:
            return "No fraud indicators detected"
        return f"Fraud indicators detected: {', '.join(fraud_flags)}"

    def _make_decision(self, risk_score: float, fraud_score: float, fraud_flags: List[str]) -> Dict[str, Any]:
        """Make final decision based on risk and fraud assessments."""
        if fraud_score > 0.7 or len(fraud_flags) > 2:
            return {
                "approved": False,
                "requires_review": False,
                "reasons": ["High fraud risk"],
                "recommendations": ["Application rejected due to fraud concerns"]
            }
        
        if risk_score < 500:
            return {
                "approved": False,
                "requires_review": False,
                "reasons": ["High risk score"],
                "recommendations": ["Application rejected due to high risk"]
            }
        
        if risk_score < 650 or fraud_score > 0.3:
            return {
                "approved": True,
                "requires_review": True,
                "reasons": ["Moderate risk", "Potential fraud indicators"],
                "recommendations": ["Manual review recommended"]
            }
        
        return {
            "approved": True,
            "requires_review": False,
            "reasons": ["Low risk", "No fraud indicators"],
            "recommendations": ["Application approved"]
        }
    
    def _extract_risk_features(self, application_data: Dict) -> List:
        """Extract features for risk assessment, matching model training order."""
        credit_data = application_data["credit_data"]
        banking_data = application_data["banking_data"]
        behavioral_data = application_data["behavioral_data"]
        employment_data = application_data["employment_data"]
        loan_amount = application_data["loan_amount"]
        loan_term = application_data["loan_term"]
        loan_purpose = application_data.get("loan_purpose", "Other")

        # Map loan_purpose to a numeric value
        purpose_map = {
            "home_improvement": 1,
            "debt_consolidation": 2,
            "business": 3,
            "education": 4,
            "medical_expenses": 5,
            "major_purchase": 6
        }
        numeric_purpose = purpose_map.get(loan_purpose, 0)

        return [
            # Credit features
            credit_data["credit_score"],
            credit_data["delinquencies"],
            credit_data["inquiries_last_6m"],
            credit_data["tradelines"],
            credit_data["utilization"],
            credit_data["payment_history_score"],
            credit_data["credit_age_months"],
            credit_data["credit_mix_score"],
            # Banking features
            banking_data["avg_monthly_income"],
            banking_data["income_stability_score"],
            banking_data["spending_pattern_score"],
            banking_data["transaction_count"],
            banking_data["avg_account_balance"],
            banking_data["overdraft_frequency"],
            banking_data["savings_rate"],
            # Behavioral features
            behavioral_data["application_completion_time"],
            behavioral_data["device_trust_score"],
            behavioral_data["location_risk_score"],
            behavioral_data["digital_footprint_score"],
            # Employment features
            employment_data["employment_length_months"],
            employment_data["employment_verification_score"],
            employment_data["income_verification_score"],
            # Loan features
            loan_amount,
            loan_term,
            numeric_purpose  # Numeric encoding for loan_purpose
        ]
    
    def _extract_fraud_features(self, application_data: Dict) -> List[float]:
        """Extract features for fraud detection."""
        app_data = application_data["application_data"]
        behavioral_data = application_data["behavioral_data"]
        employment_data = application_data["employment_data"]
        
        # Convert timestamp to hour of day and day of week
        timestamp_str = app_data.get("application_timestamp")
        if not isinstance(timestamp_str, str):
            # Default to current time if timestamp is not a string
            timestamp = datetime.utcnow()
        else:
            timestamp = datetime.fromisoformat(timestamp_str)
        hour_of_day = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Extract device and network features
        device_fingerprint = app_data["device_fingerprint"]
        browser_fingerprint = app_data["browser_fingerprint"]
        network_info = app_data["network_info"]
        
        # Ensure all 25 features are included with default values if missing
        return [
            # Temporal features
            hour_of_day,
            day_of_week,
            
            # Device features
            behavioral_data["device_trust_score"],
            len(app_data["device_id"]),
            len(app_data["email_domain"]),
            len(app_data["phone_number"]),
            
            # Behavioral features
            behavioral_data["application_completion_time"],
            behavioral_data["location_risk_score"],
            
            # Employment verification
            employment_data["employment_verification_score"],
            employment_data["income_verification_score"],
            
            # Network features
            network_info.get("proxy_score", 0),
            network_info.get("vpn_score", 0),
            network_info.get("tor_score", 0),
            
            # Additional features to match 25 expected features
            0.0,  # Feature 14
            0.0,  # Feature 15
            0.0,  # Feature 16
            0.0,  # Feature 17
            0.0,  # Feature 18
            0.0,  # Feature 19
            0.0,  # Feature 20
            0.0,  # Feature 21
            0.0,  # Feature 22
            0.0,  # Feature 23
            0.0,  # Feature 24
            0.0   # Feature 25
        ]
    
    def _generate_risk_assessment(self, risk_score: float, application_data: Dict) -> Dict:
        """Generate risk assessment based on model output."""
        credit_data = application_data["credit_data"]
        banking_data = application_data["banking_data"]
        behavioral_data = application_data["behavioral_data"]
        employment_data = application_data["employment_data"]
        
        # Calculate probability of default
        prob_default = risk_score
        
        # Convert probability to risk score (300-850)
        risk_score = int(300 + (1 - prob_default) * 550)
        
        # Determine risk segment with enhanced logic
        if (prob_default < 0.2 and 
            credit_data["credit_score"] >= 700 and 
            banking_data["income_stability_score"] >= 0.8 and
            employment_data["employment_verification_score"] >= 0.9):
            risk_segment = "Low Risk"
        elif (prob_default < 0.4 and 
              credit_data["credit_score"] >= 600 and 
              banking_data["income_stability_score"] >= 0.6 and
              employment_data["employment_verification_score"] >= 0.7):
            risk_segment = "Medium Risk"
        elif (prob_default < 0.6 and 
              credit_data["credit_score"] >= 500 and 
              banking_data["income_stability_score"] >= 0.4 and
              employment_data["employment_verification_score"] >= 0.5):
            risk_segment = "High Risk"
        else:
            risk_segment = "Very High Risk"
        
        # Identify top risk factors with enhanced analysis
        top_factors = [
            {"credit_score": 1 - (credit_data["credit_score"] / 850)},
            {"delinquencies": credit_data["delinquencies"] / 10},
            {"utilization": credit_data["utilization"]},
            {"income_stability": 1 - banking_data["income_stability_score"]},
            {"employment_verification": 1 - employment_data["employment_verification_score"]},
            {"device_trust": 1 - behavioral_data["device_trust_score"]}
        ]
        
        # Sort factors by impact
        top_factors.sort(key=lambda x: list(x.values())[0], reverse=True)
        
        return {
            "score": risk_score,  # Use the calculated risk score instead of credit score
            "probability_default": float(prob_default),
            "risk_segment": risk_segment,
            "top_factors": top_factors[:5],  # Return top 5 factors
            "explanation": {
                "risk_level": f"Application classified as {risk_segment}",
                "factors": "Top risk factors identified based on credit history, banking behavior, and employment verification",
                "recommendation": self._get_risk_recommendation(risk_segment, top_factors)
            }
        }
    
    def _generate_fraud_assessment(self, fraud_score: float, application_data: Dict) -> Dict:
        """Generate fraud assessment based on model output."""
        app_data = application_data["application_data"]
        behavioral_data = application_data["behavioral_data"]
        employment_data = application_data["employment_data"]
        credit_data = application_data["credit_data"]
        banking_data = application_data["banking_data"]
        
        # Determine fraud flags with enhanced detection
        fraud_flags = []
        risk_multiplier = 1.0
        
        # Credit and banking checks
        if credit_data["credit_score"] < 600:
            fraud_flags.append("Low credit score")
            risk_multiplier *= 1.3
        if credit_data["utilization"] > 0.8:
            fraud_flags.append("High credit utilization")
            risk_multiplier *= 1.4
        if credit_data["inquiries_last_6m"] > 5:
            fraud_flags.append("Multiple recent credit inquiries")
            risk_multiplier *= 1.3
        if banking_data["income_stability_score"] < 0.4:
            fraud_flags.append("Unstable income")
            risk_multiplier *= 1.4
        
        # Device and location checks
        if behavioral_data["device_trust_score"] < 0.5:
            fraud_flags.append("Suspicious device pattern")
            risk_multiplier *= 1.3
        if behavioral_data["location_risk_score"] > 0.7:
            fraud_flags.append("High-risk location")
            risk_multiplier *= 1.4
        if behavioral_data["digital_footprint_score"] < 0.3:
            fraud_flags.append("Limited digital footprint")
            risk_multiplier *= 1.3
            
        # Email and phone checks
        if app_data["email_domain"] in ["tempmail.com", "throwaway.com", "temp-mail.org"]:
            fraud_flags.append("Suspicious email domain")
            risk_multiplier *= 1.5
        if len(app_data["phone_number"]) < 10:
            fraud_flags.append("Invalid phone number")
            risk_multiplier *= 1.3
            
        # Employment verification
        if employment_data["employment_verification_score"] < 0.5:
            fraud_flags.append("Employment verification failed")
            risk_multiplier *= 1.4
        if employment_data["income_verification_score"] < 0.5:
            fraud_flags.append("Income verification failed")
            risk_multiplier *= 1.4
            
        # Network checks
        network_info = app_data["network_info"]
        if network_info.get("proxy_detected", False):
            fraud_flags.append("Proxy detected")
            risk_multiplier *= 1.5
        if network_info.get("vpn_detected", False):
            fraud_flags.append("VPN detected")
            risk_multiplier *= 1.5
            
        # Behavioral checks
        if behavioral_data["application_completion_time"] < 60:  # Less than 1 minute
            fraud_flags.append("Suspiciously fast application completion")
            risk_multiplier *= 1.3
        
        # Apply risk multiplier to fraud score
        adjusted_fraud_score = min(1.0, fraud_score * risk_multiplier)
        
        # Ensure minimum fraud score for suspicious applications
        if len(fraud_flags) >= 3:
            adjusted_fraud_score = max(adjusted_fraud_score, 0.7)
        elif len(fraud_flags) >= 2:
            adjusted_fraud_score = max(adjusted_fraud_score, 0.5)
        
        # Additional checks for very suspicious applications
        if (network_info.get("proxy_detected", False) or 
            network_info.get("vpn_detected", False) or
            app_data["email_domain"] in ["tempmail.com", "throwaway.com", "temp-mail.org"]):
            adjusted_fraud_score = max(adjusted_fraud_score, 0.8)
        
        return {
            "score": float(adjusted_fraud_score),
            "flags": fraud_flags,
            "explanation": {
                "fraud_risk": f"Fraud risk score: {adjusted_fraud_score:.2f}",
                "flags": "Fraud flags identified based on application patterns, device data, and verification results",
                "recommendation": self._get_fraud_recommendation(adjusted_fraud_score, fraud_flags)
            }
        }
    
    def _get_risk_recommendation(self, risk_segment: str, top_factors: List[Dict[str, float]]) -> str:
        """Generate risk-based recommendation."""
        if risk_segment == "Low Risk":
            return "Application appears low risk. Consider automatic approval."
        elif risk_segment == "Medium Risk":
            return "Application shows moderate risk. Consider approval with standard terms."
        elif risk_segment == "High Risk":
            return "Application shows high risk. Consider manual review or higher interest rate."
        else:
            return "Application shows very high risk. Consider decline or extensive review."
    
    def _get_fraud_recommendation(self, fraud_score: float, fraud_flags: List[str]) -> str:
        """Generate fraud-based recommendation."""
        if fraud_score > 0.7 or len(fraud_flags) >= 3:
            return "High fraud risk detected. Recommend decline and flag for investigation."
        elif fraud_score > 0.4 or len(fraud_flags) >= 2:
            return "Moderate fraud risk detected. Recommend manual review with enhanced verification."
        else:
            return "Low fraud risk detected. Proceed with standard verification."
    
    def _make_final_decision(self, risk_assessment: Dict, fraud_assessment: Dict) -> Dict:
        """Make final decision based on risk and fraud assessments."""
        risk_score = risk_assessment["score"]  # Now using the actual risk score (300-850)
        risk_segment = risk_assessment["risk_segment"]
        prob_default = risk_assessment["probability_default"]
        fraud_score = fraud_assessment["score"]
        fraud_flags = fraud_assessment["flags"]
        
        # Define risk thresholds
        RISK_THRESHOLDS = {
            "prob_default_high": 0.25,  # 25% probability of default
            "prob_default_medium": 0.15,  # 15% probability of default
            "risk_score_low": 600,  # FICO-like score threshold
            "risk_score_medium": 700,  # FICO-like score threshold
            "fraud_score_high": 0.7,  # 70% fraud probability
            "fraud_score_medium": 0.4,  # 40% fraud probability
            "min_fraud_flags_high": 3,
            "min_fraud_flags_medium": 2
        }
        
        reasons = []
        recommendations = []
        approved = False
        requires_review = False
        
        # Fraud-based decision logic
        if fraud_score > RISK_THRESHOLDS["fraud_score_high"] or len(fraud_flags) >= RISK_THRESHOLDS["min_fraud_flags_high"]:
            approved = False
            requires_review = False
            reasons.append("High fraud risk detected")
            recommendations.append("Application cannot be processed due to fraud concerns")
        elif fraud_score > RISK_THRESHOLDS["fraud_score_medium"] or len(fraud_flags) >= RISK_THRESHOLDS["min_fraud_flags_medium"]:
            approved = False
            requires_review = True
            reasons.append("Moderate fraud risk requires enhanced verification")
            recommendations.append("Additional identity verification required")
        else:
            # Risk-based decision logic
            if risk_segment == "Very High Risk":
                approved = False
                requires_review = True
                reasons.append("Very high risk application requires manual review")
                recommendations.extend([
                    "Consider reapplying after improving credit score",
                    "Reduce existing debt and credit utilization",
                    "Maintain consistent payment history for 6+ months"
                ])
            elif risk_segment == "High Risk":
                approved = False
                requires_review = True
                reasons.append("High risk application requires manual review")
                recommendations.extend([
                    "Consider reapplying after improving credit score",
                    "Reduce credit utilization below 30%",
                    "Maintain consistent payment history"
                ])
            elif risk_segment == "Medium Risk":
                approved = True
                requires_review = True
                reasons.append("Medium risk application requires standard review")
                recommendations.append("Application will be reviewed with standard terms")
            else:  # Low Risk
                approved = True
                requires_review = False
                reasons.append("Low risk application meets automatic approval criteria")
                recommendations.append("Application approved with standard terms")
        
        # Additional risk-based checks
        if prob_default > RISK_THRESHOLDS["prob_default_high"]:
            approved = False
            requires_review = True
            reasons.append(f"High probability of default ({prob_default:.1%}) requires review")
            recommendations.append("Consider reducing requested loan amount")
        elif prob_default > RISK_THRESHOLDS["prob_default_medium"]:
            requires_review = True
            reasons.append(f"Moderate probability of default ({prob_default:.1%}) requires review")
            recommendations.append("Consider providing additional income documentation")
        
        if risk_score < RISK_THRESHOLDS["risk_score_low"]:
            approved = False
            requires_review = True
            reasons.append(f"Low risk score ({risk_score}) requires review")
            recommendations.append("Consider reapplying after improving credit score")
        elif risk_score < RISK_THRESHOLDS["risk_score_medium"]:
            requires_review = True
            reasons.append(f"Moderate risk score ({risk_score}) requires review")
            recommendations.append("Consider providing additional credit references")
        
        # Ensure consistency between risk segment and decision
        if risk_segment in ["High Risk", "Very High Risk"] and approved:
            approved = False
            requires_review = True
            reasons.append(f"{risk_segment} classification requires review")
            recommendations.append("Application requires manual underwriting review")
        
        # Add specific thresholds to the response for transparency
        return {
            "approved": approved,
            "requires_review": requires_review,
            "reasons": reasons,
            "recommendations": recommendations,
            "risk_segment": risk_segment,
            "risk_score": risk_score,
            "probability_default": prob_default,
            "fraud_score": fraud_score,
            "fraud_flags": fraud_flags,
            "thresholds": {
                "risk_score": {
                    "low": RISK_THRESHOLDS["risk_score_low"],
                    "medium": RISK_THRESHOLDS["risk_score_medium"]
                },
                "probability_default": {
                    "medium": RISK_THRESHOLDS["prob_default_medium"],
                    "high": RISK_THRESHOLDS["prob_default_high"]
                },
                "fraud_score": {
                    "medium": RISK_THRESHOLDS["fraud_score_medium"],
                    "high": RISK_THRESHOLDS["fraud_score_high"]
                }
            }
        }
    
    def _record_application(self, application_data: Dict, risk_assessment: Dict, 
                          fraud_assessment: Dict, final_decision: Dict):
        """Record application in history."""
        application_record = {
            "application_id": f"APP-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.utcnow().isoformat(),
            "loan_amount": application_data["loan_amount"],
            "loan_term": application_data["loan_term"],
            "risk_score": risk_assessment["score"],
            "fraud_score": fraud_assessment["score"],
            "decision": "Approved" if final_decision["approved"] else "Rejected",
            "review_required": final_decision["requires_review"],
            "review_status": "Pending" if final_decision["requires_review"] else None
        }
        
        self.application_history.append(application_record)
        self._save_application_history()
    
    def get_model_metadata(self) -> Dict:
        """Get metadata about loaded models."""
        return {
            "risk_model": {
                "path": self.risk_model_path,
                "last_updated": datetime.fromtimestamp(
                    os.path.getmtime(self.risk_model_path)
                ).isoformat()
            },
            "fraud_model": {
                "path": self.fraud_model_path,
                "last_updated": datetime.fromtimestamp(
                    os.path.getmtime(self.fraud_model_path)
                ).isoformat()
            }
        }
    
    def has_recent_training(self, model_type: str) -> bool:
        """Check if model has been trained recently."""
        if not self.metrics_history:
            return False
        
        recent_metrics = [
            m for m in self.metrics_history
            if m["model_name"] == model_type
            and (datetime.utcnow() - datetime.fromisoformat(m["last_updated"])) < timedelta(days=1)
        ]
        
        return len(recent_metrics) > 0
    
    def retrain_model(self, model_type: str) -> Dict:
        """
        Retrain the specified model with latest data.
        
        Args:
            model_type: Type of model to retrain (risk/fraud)
            
        Returns:
            Dictionary containing training metrics
        """
        try:
            # Load latest training data
            data_path = f"data/{model_type}_training_data.csv"
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Training data not found at {data_path}")
            
            data = pd.read_csv(data_path)
            X = data.drop("target", axis=1)
            y = data["target"]
            
            # Retrain model
            if self.risk_model is None and model_type == "risk":
                from xgboost import XGBClassifier
                self.risk_model = XGBClassifier()
            if self.fraud_model is None and model_type == "fraud":
                from xgboost import XGBClassifier
                self.fraud_model = XGBClassifier()
            
            if model_type == "risk":
                self.risk_model.fit(X, y)
                model = self.risk_model
                model_path = self.risk_model_path
            else:
                self.fraud_model.fit(X, y)
                model = self.fraud_model
                model_path = self.fraud_model_path
            
            # Calculate metrics
            y_pred = model.predict(X)
            metrics = {
                "model_name": str(model_type),
                "accuracy": float(accuracy_score(y, y_pred)),
                "precision": float(precision_score(y, y_pred)),
                "recall": float(recall_score(y, y_pred)),
                "f1_score": float(f1_score(y, y_pred)),
                "last_updated": datetime.utcnow().isoformat(),
                "training_data_size": int(len(data)),
                "performance_metrics": {
                    "feature_importance": {str(col): float(val) for col, val in zip(X.columns, model.feature_importances_)}
                }
            }
            
            # Save model and metrics
            joblib.dump(model, model_path)
            self.metrics_history.append(metrics)
            self._save_metrics_history()
            
            return metrics
        except Exception as e:
            logger.error(f"Error retraining {model_type} model: {str(e)}")
            raise
    
    def get_model_metrics(self, model_type: Optional[str] = None, 
                         days: int = 7) -> List[Dict]:
        """
        Get model performance metrics.
        
        Args:
            model_type: Filter by model type (risk/fraud)
            days: Number of days of metrics to return
            
        Returns:
            List of model metrics
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            metrics = [
                m for m in self.metrics_history
                if (model_type is None or m["model_name"] == model_type)
                and datetime.fromisoformat(m["last_updated"]) >= cutoff_date
            ]
            
            return sorted(metrics, key=lambda x: x["last_updated"], reverse=True)
        except Exception as e:
            logger.error(f"Error retrieving model metrics: {str(e)}")
            raise
    
    def get_application_history(self, start_date: datetime, end_date: datetime,
                              status: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """
        Get application history and audit trail.
        
        Args:
            start_date: Start date for history
            end_date: End date for history
            status: Filter by application status
            limit: Maximum number of records to return
            
        Returns:
            List of application records
        """
        try:
            # Ensure start_date and end_date are timezone-aware (UTC)
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=timezone.utc)
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)
            
            history = [
                h for h in self.application_history
                if start_date <= datetime.fromisoformat(h["timestamp"]).replace(tzinfo=timezone.utc) <= end_date
                and (status is None or h["decision"] == status)
            ]
            
            return sorted(history, key=lambda x: x["timestamp"], reverse=True)[:limit]
        except Exception as e:
            logger.error(f"Error retrieving application history: {str(e)}")
            raise
    
    def _get_risk_segment(self, risk_score: float) -> str:
        """Get risk segment based on risk score."""
        if risk_score < 0.2:
            return "Low Risk"
        elif risk_score < 0.4:
            return "Medium Risk"
        elif risk_score < 0.6:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def _get_top_risk_factors(self, application_data: Dict) -> List[str]:
        """Get top risk factors from application data."""
        factors = []
        credit_data = application_data.get("credit_data", {})
        banking_data = application_data.get("banking_data", {})
        
        if credit_data.get("credit_score", 0) < 600:
            factors.append("Low credit score")
        if credit_data.get("credit_utilization", 0) > 0.7:
            factors.append("High credit utilization")
        if banking_data.get("income_stability_score", 0) < 0.5:
            factors.append("Unstable income")
        if application_data.get("applicant_employment_length_years", 0) < 1:
            factors.append("Short employment history")
        
        return factors
    
    def _get_risk_explanations(self, application_data: Dict) -> List[str]:
        """Get detailed risk explanations."""
        explanations = []
        credit_data = application_data.get("credit_data", {})
        banking_data = application_data.get("banking_data", {})
        
        if credit_data.get("credit_score", 0) < 600:
            explanations.append(f"Credit score of {credit_data.get('credit_score')} is below recommended threshold")
        if credit_data.get("credit_utilization", 0) > 0.7:
            explanations.append(f"Credit utilization of {credit_data.get('credit_utilization')*100}% is high")
        if banking_data.get("income_stability_score", 0) < 0.5:
            explanations.append("Income stability is below recommended threshold")
        if application_data.get("applicant_employment_length_years", 0) < 1:
            explanations.append("Employment history is less than 1 year")
        
        return explanations
    
    def _get_fraud_flags(self, application_data: Dict, fraud_score: float) -> List[str]:
        """Get fraud flags based on application data and fraud score."""
        flags = []
        app_data = application_data.get("application_data", {})
        banking_data = application_data.get("banking_data", {})
        
        if fraud_score > 0.7:
            flags.append("High fraud risk score")
        if app_data.get("email_domain") in ["temp-mail.org", "mailinator.com"]:
            flags.append("Suspicious email domain")
        if app_data.get("time_spent_on_application", 0) < 300:
            flags.append("Suspiciously short application time")
        if app_data.get("num_previous_applications", 0) > 3:
            flags.append("Multiple recent applications")
        if banking_data.get("monthly_income", 0) > 10000:
            flags.append("Unusually high income")
        
        return flags
    
    def _get_fraud_explanations(self, application_data: Dict) -> List[str]:
        """Get detailed fraud explanations."""
        explanations = []
        app_data = application_data.get("application_data", {})
        banking_data = application_data.get("banking_data", {})
        
        if app_data.get("email_domain") in ["temp-mail.org", "mailinator.com"]:
            explanations.append(f"Suspicious email domain: {app_data.get('email_domain')}")
        if app_data.get("time_spent_on_application", 0) < 300:
            explanations.append(f"Application completed in {app_data.get('time_spent_on_application')} seconds")
        if app_data.get("num_previous_applications", 0) > 3:
            explanations.append(f"Multiple applications in last 30 days: {app_data.get('num_previous_applications')}")
        if banking_data.get("monthly_income", 0) > 10000:
            explanations.append(f"Unusually high monthly income: ${banking_data.get('monthly_income')}")
        
        return explanations
    
    def _get_decision_reasons(self, risk_score: float, fraud_score: float) -> List[str]:
        """Get decision reasons based on risk and fraud scores."""
        reasons = []
        
        if risk_score >= 0.7:
            reasons.append("High risk score indicates significant probability of default")
        elif risk_score >= 0.5:
            reasons.append("Moderate risk score requires additional review")
        
        if fraud_score >= 0.7:
            reasons.append("High fraud risk detected")
        elif fraud_score >= 0.5:
            reasons.append("Suspicious patterns detected requiring review")
        
        return reasons
    
    def _get_decision_recommendations(self, risk_score: float, fraud_score: float) -> List[str]:
        """Get decision recommendations based on risk and fraud scores."""
        recommendations = []
        
        if risk_score >= 0.7:
            recommendations.append("Consider requesting additional collateral")
            recommendations.append("Review income verification documents")
        elif risk_score >= 0.5:
            recommendations.append("Request additional income documentation")
            recommendations.append("Consider shorter loan term")
        
        if fraud_score >= 0.7:
            recommendations.append("Perform enhanced identity verification")
            recommendations.append("Review all submitted documents")
        elif fraud_score >= 0.5:
            recommendations.append("Verify employment details")
            recommendations.append("Check for duplicate applications")
        
        return recommendations 