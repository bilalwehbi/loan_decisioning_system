# src/models/loan_decisioning_service.py (updated)
import os
import logging
import time
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import joblib

from src.models.credit_risk_model import CreditRiskModel
from src.models.fraud_detection_model import FraudDetectionModel
from src.models.explainability import ExplainabilityGenerator
from src.data.preprocessing import preprocess_loan_application
from src.configs.config import MODEL_DIR

logger = logging.getLogger(__name__)

class LoanDecisioningService:
    """
    Service that combines credit risk and fraud detection models
    to provide a unified loan decision with explanations
    """
    
    def __init__(self):
        """Initialize the service with empty models"""
        # Initialize models first
        self.risk_model = CreditRiskModel()
        self.fraud_model = FraudDetectionModel()
        self.explainability_generator = ExplainabilityGenerator()
        self.metrics_history = []
        
        # Load models immediately during initialization
        self.load_models()
        
    def load_models(self):
        """Load the latest trained models"""
        logger.info(f"Loading models from directory: {MODEL_DIR}")
        logger.info(f"Absolute model directory path: {os.path.abspath(MODEL_DIR)}")
        
        if not os.path.exists(MODEL_DIR):
            raise Exception(f"Model directory does not exist: {MODEL_DIR}")
        
        logger.info(f"Directory contents: {os.listdir(MODEL_DIR)}")
        
        try:
            # Load credit risk model
            credit_model_files = [f for f in os.listdir(MODEL_DIR) 
                                if (f.startswith('credit_risk_model_') and f.endswith('.pkl')) or
                                   (f == 'risk_model.joblib')]
            logger.info(f"Found credit risk model files: {credit_model_files}")
            
            if not credit_model_files:
                raise Exception("No credit risk model files found")
            
            # Prefer .joblib file if it exists, otherwise use latest .pkl
            credit_model_file = next((f for f in credit_model_files if f.endswith('.joblib')), max(credit_model_files))
            credit_model_path = os.path.join(MODEL_DIR, credit_model_file)
            logger.info(f"Loading credit risk model from: {credit_model_path}")
            
            try:
                self.risk_model.load(credit_model_path)
                logger.info(f"Successfully loaded credit risk model: {self.risk_model.model_id}")
            except Exception as e:
                logger.error(f"Error loading credit risk model: {str(e)}")
                raise Exception(f"Failed to load credit risk model: {str(e)}")
            
            # Load fraud detection model
            fraud_model_files = [f for f in os.listdir(MODEL_DIR) 
                                if (f.startswith('fraud_detection_model_') and f.endswith('.pkl')) or
                                   (f == 'fraud_model.joblib')]
            logger.info(f"Found fraud detection model files: {fraud_model_files}")
            
            if not fraud_model_files:
                raise Exception("No fraud detection model files found")
            
            # Prefer .joblib file if it exists, otherwise use latest .pkl
            fraud_model_file = next((f for f in fraud_model_files if f.endswith('.joblib')), max(fraud_model_files))
            fraud_model_path = os.path.join(MODEL_DIR, fraud_model_file)
            logger.info(f"Loading fraud detection model from: {fraud_model_path}")
            
            try:
                self.fraud_model.load(fraud_model_path)
                logger.info(f"Successfully loaded fraud detection model: {self.fraud_model.model_id}")
            except Exception as e:
                logger.error(f"Error loading fraud detection model: {str(e)}")
                raise Exception(f"Failed to load fraud detection model: {str(e)}")
            
            # Load metrics history if available
            metrics_path = os.path.join(MODEL_DIR, 'metrics_history.json')
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'r') as f:
                        self.metrics_history = json.load(f)
                    logger.info(f"Successfully loaded metrics history with {len(self.metrics_history)} entries")
                except Exception as e:
                    logger.error(f"Error loading metrics history: {str(e)}")
                    self.metrics_history = []
            else:
                logger.info("No metrics history file found, initializing empty history")
                self.metrics_history = []
                
        except Exception as e:
            logger.error(f"Error in model loading process: {str(e)}")
            raise Exception(f"Failed to initialize loan decisioning service: {str(e)}")
    
    def _flatten_loan_application(self, loan_application: Dict) -> pd.DataFrame:
        """
        Convert nested loan application dict to flat DataFrame
        for model processing and explainability
        
        Args:
            loan_application: Nested loan application dictionary
            
        Returns:
            flat_df: Flattened DataFrame with all features
        """
        flat_dict = {}
        
        # Add loan-level data
        for key, value in loan_application.items():
            if not isinstance(value, dict):
                flat_dict[key] = value
        
        # Add nested data
        for section in ['credit_data', 'banking_data', 'application_data']:
            if section in loan_application and isinstance(loan_application[section], dict):
                for key, value in loan_application[section].items():
                    flat_dict[key] = value
        
        # Convert to DataFrame
        return pd.DataFrame([flat_dict])
    
    def make_decision(self, loan_application: Dict) -> Dict:
        """
        Process a loan application and return a comprehensive decision with explanations
        
        Args:
            loan_application: Dictionary containing loan application data
            
        Returns:
            decision: Dictionary with comprehensive decision results and explanations
        """
        try:
            start_time = time.time()
            
            # Generate a unique application ID
            application_id = str(uuid.uuid4())
            
            # Flatten the application for explanation
            application_df = self._flatten_loan_application(loan_application)
            
            # Make credit risk prediction
            if not self.risk_model or not hasattr(self.risk_model, 'model'):
                raise Exception("Credit risk model not loaded")
            
            credit_risk_result = self.risk_model.predict(loan_application)
            
            # Make fraud detection prediction
            if not self.fraud_model or not hasattr(self.fraud_model, 'model'):
                raise Exception("Fraud detection model not loaded")
            
            fraud_detection_result = self.fraud_model.predict(loan_application)
            
            # Generate explanations
            credit_explanations = self.explainability_generator.explain_credit_risk_prediction(
                self.risk_model.model,
                self.risk_model.pipeline,
                application_df,
                self.risk_model.feature_names
            )
            
            fraud_explanations = self.explainability_generator.explain_fraud_prediction(
                self.fraud_model.model,
                self.fraud_model.pipeline,
                application_df,
                self.fraud_model.feature_names
            )
            
            # Combine results
            decision = credit_risk_result['decision']
            
            if fraud_detection_result['fraud_score'] >= 0.7:
                if decision == "Low Risk":
                    decision = "Review"
                else:
                    decision = "Decline"
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare response
            response = {
                "application_id": application_id,
                "timestamp": datetime.now().isoformat(),
                "processing_time_ms": round(processing_time * 1000, 2),
                "risk_score": credit_risk_result['credit_score'],
                "probability_of_default": credit_risk_result['probability_of_default'],
                "decision": decision,
                "fraud_score": fraud_detection_result['fraud_score'],
                "explanations": credit_explanations + fraud_explanations,
                "processing_time": processing_time,
                "credit_risk": {
                    "score": credit_risk_result['credit_score'],
                    "probability_of_default": credit_risk_result['probability_of_default'],
                    "decision": decision,
                    "top_factors": credit_risk_result['top_factors'],
                    "explanations": credit_explanations
                },
                "fraud_detection": {
                    "score": fraud_detection_result['fraud_score'],
                    "flags": fraud_detection_result['fraud_flags'],
                    "explanations": fraud_explanations
                },
                "final_decision": decision
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error making loan decision: {str(e)}")
            raise Exception(f"Failed to process loan application: {str(e)}")

    def get_model_metrics(self, model_type: Optional[str] = None, days: int = 7) -> List[Dict]:
        """
        Get model performance metrics.
        
        Args:
            model_type: Filter by model type (risk/fraud)
            days: Number of days of metrics to return
            
        Returns:
            List of model metrics
        """
        try:
            if not self.risk_model or not self.fraud_model:
                raise ValueError("Models not loaded")
            
            # Get current metrics
            current_metrics = []
            
            # Credit risk model metrics
            if model_type is None or model_type == "risk":
                credit_metrics = {
                    "model_name": "credit_risk",
                    "accuracy": self.risk_model.get_accuracy() or 0.0,
                    "precision": self.risk_model.get_precision() or 0.0,
                    "recall": self.risk_model.get_recall() or 0.0,
                    "f1_score": self.risk_model.get_f1_score() or 0.0,
                    "last_updated": datetime.now().isoformat(),
                    "training_data_size": self.risk_model.get_training_data_size() or 0,
                    "performance_metrics": self.risk_model.get_performance_metrics() or {}
                }
                
                # Add historical metrics if available
                if self.metrics_history:
                    credit_history = [m for m in self.metrics_history 
                                    if m.get('model_name') == 'credit_risk']
                    if credit_history:
                        credit_metrics['historical_metrics'] = credit_history[-days:]
                
                current_metrics.append(credit_metrics)
            
            # Fraud detection model metrics
            if model_type is None or model_type == "fraud":
                fraud_metrics = {
                    "model_name": "fraud_detection",
                    "accuracy": self.fraud_model.get_accuracy() or 0.0,
                    "precision": self.fraud_model.get_precision() or 0.0,
                    "recall": self.fraud_model.get_recall() or 0.0,
                    "f1_score": self.fraud_model.get_f1_score() or 0.0,
                    "last_updated": datetime.now().isoformat(),
                    "training_data_size": self.fraud_model.get_training_data_size() or 0,
                    "performance_metrics": self.fraud_model.get_performance_metrics() or {}
                }
                
                # Add historical metrics if available
                if self.metrics_history:
                    fraud_history = [m for m in self.metrics_history 
                                   if m.get('model_name') == 'fraud_detection']
                    if fraud_history:
                        fraud_metrics['historical_metrics'] = fraud_history[-days:]
                
                current_metrics.append(fraud_metrics)
            
            return current_metrics
            
        except Exception as e:
            logger.error(f"Error getting model metrics: {str(e)}")
            raise Exception(f"Failed to get model metrics: {str(e)}")

    def _load_model(self, model_path: str) -> Any:
        """Load a model from disk."""
        try:
            logger.info(f"Attempting to load model from {model_path}")
            model_data = joblib.load(model_path)
            # If it's a dict with a 'model' key, extract the model
            if isinstance(model_data, dict) and 'model' in model_data:
                logger.info(f"Model loaded successfully from {model_path}")
                return model_data['model']
            logger.info(f"Model loaded successfully from {model_path}")
            return model_data
        except FileNotFoundError:
            logger.warning(f"Model file not found: {model_path}. The model will need to be trained.")
            return None
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            return None

    def update_thresholds(self, thresholds: dict) -> dict:
        """
        Update custom scoring thresholds for risk segmentation
        
        Args:
            thresholds: Dictionary containing new threshold values
            
        Returns:
            Updated threshold configuration
        """
        try:
            # Validate threshold values
            self._validate_thresholds(thresholds)
            
            # Update thresholds in configuration
            self.config.update_thresholds(thresholds)
            
            return self.config.get_thresholds()
        except Exception as e:
            logger.error(f"Error updating thresholds: {str(e)}")
            raise

    def make_enhanced_decision(self, application_data: dict) -> dict:
        """
        Make enhanced loan decision with alternative data and detailed risk segmentation
        
        Args:
            application_data: Loan application data with optional alternative data sources
            
        Returns:
            Enhanced loan decision with detailed risk factors
        """
        try:
            # Process alternative data sources
            alt_data_features = self._process_alternative_data(application_data)
            
            # Get base credit risk score
            base_score = self.risk_model.predict(application_data)
            
            # Get behavioral risk score
            behavioral_score = self._calculate_behavioral_score(application_data, alt_data_features)
            
            # Get fraud risk score
            fraud_score = self.fraud_model.predict(application_data)
            
            # Combine scores and get risk segmentation
            final_score = self._combine_scores(base_score, behavioral_score, fraud_score)
            risk_segment = self._get_risk_segment(final_score)
            
            # Generate detailed explanations
            explanations = self._generate_detailed_explanations(
                application_data,
                base_score,
                behavioral_score,
                fraud_score
            )
            
            return {
                "application_id": application_data.get("application_id"),
                "final_score": final_score,
                "risk_segment": risk_segment,
                "base_credit_score": base_score,
                "behavioral_score": behavioral_score,
                "fraud_score": fraud_score,
                "explanations": explanations,
                "alternative_data_used": list(alt_data_features.keys()),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error making enhanced decision: {str(e)}")
            raise

    def _process_alternative_data(self, application_data: dict) -> dict:
        """Process alternative data sources to extract features"""
        features = {}
        
        # Process bank transaction data
        if "bank_transactions" in application_data:
            features.update(self._process_bank_transactions(application_data["bank_transactions"]))
        
        # Process device metadata
        if "device_metadata" in application_data:
            features.update(self._process_device_metadata(application_data["device_metadata"]))
        
        # Process application behavior
        if "application_behavior" in application_data:
            features.update(self._process_application_behavior(application_data["application_behavior"]))
        
        return features

    def _calculate_behavioral_score(self, application_data: dict, alt_data_features: dict) -> float:
        """Calculate behavioral risk score using alternative data features"""
        try:
            # Combine application data with alternative data features
            features = {**application_data, **alt_data_features}
            
            # Use behavioral model to predict score
            return self.behavioral_model.predict(features)
        except Exception as e:
            logger.error(f"Error calculating behavioral score: {str(e)}")
            raise

    def _get_risk_segment(self, final_score: float) -> str:
        """Get risk segment based on final score and thresholds"""
        thresholds = self.config.get_thresholds()
        
        if final_score >= thresholds["high_risk_threshold"]:
            return "High Risk"
        elif final_score >= thresholds["medium_risk_threshold"]:
            return "Medium Risk"
        else:
            return "Low Risk"

    def _generate_detailed_explanations(
        self,
        application_data: dict,
        base_score: float,
        behavioral_score: float,
        fraud_score: float
    ) -> dict:
        """Generate detailed explanations for each risk component"""
        explanations = {
            "credit_risk": self._explain_credit_risk(application_data, base_score),
            "behavioral_risk": self._explain_behavioral_risk(application_data, behavioral_score),
            "fraud_risk": self._explain_fraud_risk(application_data, fraud_score)
        }
        
        return explanations

    def get_performance_metrics(self, start_date: datetime, end_date: datetime, metric_type: str) -> dict:
        """Get detailed performance metrics for monitoring"""
        try:
            metrics = {
                "model_performance": self._get_model_performance_metrics(start_date, end_date),
                "system_performance": self._get_system_performance_metrics(start_date, end_date),
                "business_metrics": self._get_business_metrics(start_date, end_date)
            }
            
            if metric_type != "all":
                return metrics.get(metric_type, {})
            
            return metrics
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            raise

    def generate_compliance_report(self, start_date: datetime, end_date: datetime, report_type: str) -> dict:
        """Generate compliance report for audit purposes"""
        try:
            report = {
                "decision_summary": self._get_decision_summary(start_date, end_date),
                "model_performance": self._get_model_performance_metrics(start_date, end_date),
                "risk_distribution": self._get_risk_distribution(start_date, end_date),
                "audit_trail": self._get_audit_trail(start_date, end_date)
            }
            
            if report_type == "summary":
                return {
                    "decision_summary": report["decision_summary"],
                    "risk_distribution": report["risk_distribution"]
                }
            
            return report
        except Exception as e:
            logger.error(f"Error generating compliance report: {str(e)}")
            raise