# src/models/model_feedback_loop.py
import os
import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import uuid

from src.models.credit_risk_model import CreditRiskModel
from src.models.fraud_detection_model import FraudDetectionModel
from src.configs.config import DATA_DIR, MODEL_DIR, PERFORMANCE_TARGETS

logger = logging.getLogger(__name__)

class ModelFeedbackLoop:
    """
    Handles model performance monitoring and retraining
    based on new outcome data
    """
    
    def __init__(self):
        self.outcome_data_path = os.path.join(DATA_DIR, "feedback_outcomes.csv")
        self.predictions_path = os.path.join(DATA_DIR, "predictions_log.csv")
        self.performance_log_path = os.path.join(DATA_DIR, "model_performance_log.json")
        
        # Ensure necessary files exist
        self._initialize_data_files()
        
        # Initialize performance tracking
        self.performance_data = {
            'risk_model': [],
            'fraud_model': []
        }
        
        # Load historical performance data if available
        self._load_performance_history()
        
        # Initialize retraining thresholds
        self.retraining_thresholds = {
            'risk_model': {
                'accuracy_drop': 0.05,  # 5% drop in accuracy
                'f1_drop': 0.05,        # 5% drop in F1 score
                'min_samples': 1000     # Minimum samples for retraining
            },
            'fraud_model': {
                'accuracy_drop': 0.05,
                'f1_drop': 0.05,
                'min_samples': 1000
            }
        }
    
    def _initialize_data_files(self):
        """Initialize data files if they don't exist"""
        # Create feedback outcomes file
        if not os.path.exists(self.outcome_data_path):
            # Create with headers
            pd.DataFrame(columns=[
                'application_id', 'prediction_date', 'outcome_date', 
                'predicted_default', 'actual_default', 'predicted_fraud', 'actual_fraud'
            ]).to_csv(self.outcome_data_path, index=False)
            logger.info(f"Created feedback outcomes file: {self.outcome_data_path}")
        
        # Create predictions log file
        if not os.path.exists(self.predictions_path):
            # Create with headers
            pd.DataFrame(columns=[
                'application_id', 'timestamp', 'credit_score', 'default_probability',
                'decision', 'fraud_score', 'fraud_flags', 'processing_time'
            ]).to_csv(self.predictions_path, index=False)
            logger.info(f"Created predictions log file: {self.predictions_path}")
        
        # Create performance log file
        if not os.path.exists(self.performance_log_path):
            # Create initial performance log
            performance_data = {
                'risk_model': [],
                'fraud_model': []
            }
            with open(self.performance_log_path, 'w') as f:
                json.dump(performance_data, f, indent=2)
            logger.info(f"Created model performance log: {self.performance_log_path}")
    
    def log_prediction(self, prediction_data: Dict) -> None:
        """
        Log a prediction to the predictions file
        
        Args:
            prediction_data: Dictionary containing prediction info
        """
        try:
            # Create row for predictions log
            prediction_row = {
                'application_id': prediction_data.get('application_id'),
                'timestamp': datetime.now().isoformat(),
                'credit_score': prediction_data.get('credit_score'),
                'default_probability': prediction_data.get('probability_of_default'),
                'decision': prediction_data.get('decision'),
                'fraud_score': prediction_data.get('fraud_score'),
                'fraud_flags': "|".join(prediction_data.get('fraud_flags', [])),
                'processing_time': prediction_data.get('processing_time')
            }
            
            # Append to CSV
            predictions_df = pd.read_csv(self.predictions_path)
            predictions_df = pd.concat([predictions_df, pd.DataFrame([prediction_row])], ignore_index=True)
            predictions_df.to_csv(self.predictions_path, index=False)
            
            logger.info(f"Logged prediction for application ID: {prediction_data.get('application_id')}")
        except Exception as e:
            logger.error(f"Error logging prediction: {str(e)}")
    
    def record_outcome(self, outcome_data: Dict) -> None:
        """
        Record actual loan outcome for a previous prediction
        
        Args:
            outcome_data: Dictionary with actual outcome info
            {
                'application_id': 'unique_id',
                'actual_default': 1 or 0,
                'actual_fraud': 1 or 0,
                'outcome_date': '2023-01-01' (optional)
            }
        """
        try:
            # Load existing outcomes
            outcomes_df = pd.read_csv(self.outcome_data_path)
            
            # Load predictions to get the prediction data
            predictions_df = pd.read_csv(self.predictions_path)
            prediction = predictions_df[predictions_df['application_id'] == outcome_data['application_id']]
            
            if len(prediction) == 0:
                logger.warning(f"No prediction found for application ID: {outcome_data['application_id']}")
                return
            
            # Create outcome row
            outcome_row = {
                'application_id': outcome_data['application_id'],
                'prediction_date': prediction['timestamp'].values[0],
                'outcome_date': outcome_data.get('outcome_date', datetime.now().isoformat()),
                'predicted_default': prediction['default_probability'].values[0],
                'actual_default': outcome_data['actual_default'],
                'predicted_fraud': prediction['fraud_score'].values[0],
                'actual_fraud': outcome_data['actual_fraud']
            }
            
            # Check if this application_id already has an outcome
            existing_outcome = outcomes_df[outcomes_df['application_id'] == outcome_data['application_id']]
            if len(existing_outcome) > 0:
                # Update existing outcome
                for col in outcome_row:
                    outcomes_df.loc[outcomes_df['application_id'] == outcome_data['application_id'], col] = outcome_row[col]
            else:
                # Append new outcome
                outcomes_df = pd.concat([outcomes_df, pd.DataFrame([outcome_row])], ignore_index=True)
            
            # Save updated outcomes
            outcomes_df.to_csv(self.outcome_data_path, index=False)
            
            logger.info(f"Recorded outcome for application ID: {outcome_data['application_id']}")
        except Exception as e:
            logger.error(f"Error recording outcome: {str(e)}")
    
    def evaluate_model_performance(self) -> Dict:
        """
        Evaluate current model performance based on recorded outcomes
        
        Returns:
            performance: Dictionary of performance metrics
        """
        try:
            # Load outcomes data
            outcomes_df = pd.read_csv(self.outcome_data_path)
            
            # Filter to outcomes with actual results
            valid_outcomes = outcomes_df.dropna(subset=['actual_default', 'actual_fraud'])
            
            if len(valid_outcomes) < 10:  # Lower threshold for demonstration
                logger.warning(f"Not enough outcome data for reliable evaluation (found {len(valid_outcomes)}, need at least 10)")
                return {"status": "insufficient_data", "count": len(valid_outcomes)}
            
            # Calculate credit risk model performance
            credit_metrics = self._calculate_credit_metrics(valid_outcomes)
            
            # Calculate fraud detection model performance
            fraud_metrics = self._calculate_fraud_metrics(valid_outcomes)
            
            # Log performance
            self._log_performance_metrics(credit_metrics, fraud_metrics)
            
            # Return combined metrics
            return {
                "status": "success",
                "count": len(valid_outcomes),
                "credit_risk_metrics": credit_metrics,
                "fraud_detection_metrics": fraud_metrics
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model performance: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _calculate_credit_metrics(self, outcomes_df: pd.DataFrame) -> Dict:
        """Calculate performance metrics for credit risk model"""
        # Convert probabilities to binary predictions using 0.5 threshold
        outcomes_df['predicted_default_binary'] = (outcomes_df['predicted_default'] >= 0.5).astype(int)
        
        # Calculate metrics
        true_positives = sum((outcomes_df['predicted_default_binary'] == 1) & (outcomes_df['actual_default'] == 1))
        false_positives = sum((outcomes_df['predicted_default_binary'] == 1) & (outcomes_df['actual_default'] == 0))
        true_negatives = sum((outcomes_df['predicted_default_binary'] == 0) & (outcomes_df['actual_default'] == 0))
        false_negatives = sum((outcomes_df['predicted_default_binary'] == 0) & (outcomes_df['actual_default'] == 1))
        
        # Calculate precision, recall, and F1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate false positive rate
        fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
        
        # Calculate AUC (approximate)
        # For a proper AUC, we'd need to compute the ROC curve
        # This is a simplified version for feedback purposes
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(outcomes_df['actual_default'], outcomes_df['predicted_default'])
        except:
            auc = 0.5  # Default if calculation fails
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'false_positive_rate': fpr,
            'auc': auc,
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'true_negatives': int(true_negatives),
            'false_negatives': int(false_negatives),
            'evaluation_date': datetime.now().isoformat(),
            'sample_size': len(outcomes_df)
        }
        
        return metrics
    
    def _calculate_fraud_metrics(self, outcomes_df: pd.DataFrame) -> Dict:
        """Calculate performance metrics for fraud detection model"""
        # Convert probabilities to binary predictions using the model threshold
        # For fraud, we typically use a lower threshold like 0.7
        fraud_threshold = 0.7  # This should come from the model config
        outcomes_df['predicted_fraud_binary'] = (outcomes_df['predicted_fraud'] >= fraud_threshold).astype(int)
        
        # Calculate metrics
        true_positives = sum((outcomes_df['predicted_fraud_binary'] == 1) & (outcomes_df['actual_fraud'] == 1))
        false_positives = sum((outcomes_df['predicted_fraud_binary'] == 1) & (outcomes_df['actual_fraud'] == 0))
        true_negatives = sum((outcomes_df['predicted_fraud_binary'] == 0) & (outcomes_df['actual_fraud'] == 0))
        false_negatives = sum((outcomes_df['predicted_fraud_binary'] == 0) & (outcomes_df['actual_fraud'] == 1))
        
        # Calculate precision, recall, and F1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate false positive rate
        fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
        
        # Calculate AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(outcomes_df['actual_fraud'], outcomes_df['predicted_fraud'])
        except:
            auc = 0.5
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'false_positive_rate': fpr,
            'auc': auc,
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'true_negatives': int(true_negatives),
            'false_negatives': int(false_negatives),
            'evaluation_date': datetime.now().isoformat(),
            'sample_size': len(outcomes_df)
        }
        
        return metrics
    
    def _log_performance_metrics(self, credit_metrics: Dict, fraud_metrics: Dict) -> None:
        """Log performance metrics to the performance log file"""
        try:
            # Load existing performance log
            with open(self.performance_log_path, 'r') as f:
                performance_data = json.load(f)
            
            # Append new metrics
            performance_data['risk_model'].append(credit_metrics)
            performance_data['fraud_model'].append(fraud_metrics)
            
            # Save updated performance log
            with open(self.performance_log_path, 'w') as f:
                json.dump(performance_data, f, indent=2)
            
            logger.info(f"Logged model performance metrics")
        except Exception as e:
            logger.error(f"Error logging performance metrics: {str(e)}")
    
    def should_retrain_models(self) -> Dict:
        """
        Determine if models should be retrained based on performance
        
        Returns:
            retraining_decision: Dictionary with retraining recommendations
        """
        try:
            # Load performance log
            with open(self.performance_log_path, 'r') as f:
                performance_data = json.load(f)
            
            # Check if we have enough performance data
            if (len(performance_data['risk_model']) < 2 or 
                len(performance_data['fraud_model']) < 2):
                return {
                    "retrain_risk_model": False,
                    "retrain_fraud_model": False,
                    "reason": "Insufficient performance history"
                }
            
            # Get latest and previous metrics
            latest_credit = performance_data['risk_model'][-1]
            prev_credit = performance_data['risk_model'][-2]
            
            latest_fraud = performance_data['fraud_model'][-1]
            prev_fraud = performance_data['fraud_model'][-2]
            
            # Check for significant performance degradation
            credit_degradation = prev_credit['auc'] - latest_credit['auc']
            fraud_degradation = prev_fraud['auc'] - latest_fraud['auc']
            
            # Check if performance is below targets
            credit_below_target = (
                latest_credit['precision'] < PERFORMANCE_TARGETS['precision'] or
                latest_credit['false_positive_rate'] > PERFORMANCE_TARGETS['false_positive_rate']
            )
            
            fraud_below_target = (
                latest_fraud['precision'] < PERFORMANCE_TARGETS['precision'] or
                latest_fraud['false_positive_rate'] > PERFORMANCE_TARGETS['false_positive_rate']
            )
            
            # Make retraining decisions
            retrain_credit = credit_degradation > 0.05 or credit_below_target
            retrain_fraud = fraud_degradation > 0.05 or fraud_below_target
            
            return {
                "retrain_risk_model": retrain_credit,
                "retrain_fraud_model": retrain_fraud,
                "credit_degradation": credit_degradation,
                "fraud_degradation": fraud_degradation,
                "credit_below_target": credit_below_target,
                "fraud_below_target": fraud_below_target
            }
            
        except Exception as e:
            logger.error(f"Error determining retraining need: {str(e)}")
            return {
                "retrain_risk_model": False,
                "retrain_fraud_model": False,
                "reason": f"Error: {str(e)}"
            }
    
    def retrain_models(self) -> Dict:
        """
        Retrain models using the latest outcome data
        
        Returns:
            result: Dictionary with retraining results
        """
        try:
            # Load all available data
            # In a real system, this would include historical data as well
            merged_data = self._prepare_training_data()
            
            if len(merged_data) < 10:  # Lower threshold for demonstration
                return {
                    "status": "insufficient_data",
                    "message": f"Not enough data for retraining (found {len(merged_data)}, need at least 10)"
                }
            
            # Retrain credit risk model
            credit_result = self._retrain_risk_model(merged_data)
            
            # Retrain fraud detection model
            fraud_result = self._retrain_fraud_model(merged_data)
            
            return {
                "status": "success",
                "credit_model_result": credit_result,
                "fraud_model_result": fraud_result,
                "training_data_size": len(merged_data)
            }
            
        except Exception as e:
            logger.error(f"Error retraining models: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _prepare_training_data(self) -> pd.DataFrame:
        """Prepare data for model retraining"""
        # Load outcomes data
        outcomes_df = pd.read_csv(self.outcome_data_path)
        
        # Load original dataset (in a real system, this would be from a database)
        merged_df = pd.read_csv(os.path.join(DATA_DIR, "merged_dataset.csv"), low_memory=False)
        
        # In a real system, we would merge the outcomes with the original features
        # For this demo, we'll just use the original dataset with a subset of records
        sample_size = min(5000, len(merged_df))
        merged_df = merged_df.sample(sample_size)
        
        # Ensure target variables exist
        if 'default' not in merged_df.columns:
            merged_df['default'] = (merged_df['loan_status'] == 'defaulted').astype(int)
        
        if 'fraud' not in merged_df.columns and 'fraud_flag' in merged_df.columns:
            merged_df['fraud'] = merged_df['fraud_flag'].astype(int)
        
        return merged_df
    
    def _retrain_risk_model(self, data: pd.DataFrame) -> Dict:
        """Retrain the credit risk model."""
        try:
            # Load training data
            from src.models.train_credit_risk_model import load_data
            train_data = load_data()
            if train_data is None:
                return {'success': False, 'error': 'Failed to load training data'}
            
            # Merge with new data
            merged_data = pd.concat([train_data, data], ignore_index=True)
            
            # Train new model
            model = CreditRiskModel()
            model.train(merged_data)
            
            # Save model
            model_id = f"credit_risk_model_{uuid.uuid4().hex[:8]}"
            model_path = os.path.join(MODEL_DIR, f"{model_id}.pkl")
            model.save(model_path)
            
            # Update service
            self.risk_model = model
            
            return {'success': True, 'model_id': model_id}
            
        except Exception as e:
            logger.error(f"Error retraining credit risk model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _retrain_fraud_model(self, data: pd.DataFrame) -> Dict:
        """Retrain the fraud detection model"""
        # In a real implementation, we would call the actual training function
        # Here we'll simulate retraining for simplicity
        logger.info("Simulating fraud detection model retraining")
        
        # Return simulated results
        return {
            "model_id": f"fraud_detection_model_{uuid.uuid4().hex[:8]}",
            "auc": 0.89,
            "precision": 0.94,
            "false_positive_rate": 0.03,
            "training_date": datetime.now().isoformat()
        }
    
    def get_model_metrics(self, days: int = 7) -> List[Dict]:
        """
        Get model performance metrics for the specified time period.
        
        Args:
            days: Number of days to look back for metrics
            
        Returns:
            List[Dict]: List of model metrics
        """
        try:
            # Load performance data
            if not os.path.exists(self.performance_log_path):
                # Return default metrics if no performance data exists
                return [
                    {
                        'model_type': 'credit_risk',
                        'accuracy': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1_score': 0.0,
                        'timestamp': datetime.now().isoformat()
                    },
                    {
                        'model_type': 'fraud_detection',
                        'accuracy': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1_score': 0.0,
                        'timestamp': datetime.now().isoformat()
                    }
                ]
                
            with open(self.performance_log_path, 'r') as f:
                performance_data = [json.loads(line) for line in f]
            
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Filter metrics for the specified time period
            recent_metrics = [
                entry for entry in performance_data
                if datetime.fromisoformat(entry['timestamp']) >= cutoff_date
            ]
            
            # Group metrics by model type
            credit_metrics = [m for m in recent_metrics if m['model_type'] == 'credit_risk']
            fraud_metrics = [m for m in recent_metrics if m['model_type'] == 'fraud_detection']
            
            # Calculate averages for each model
            metrics = []
            
            if credit_metrics:
                credit_avg = {
                    'model_type': 'credit_risk',
                    'accuracy': sum(m['accuracy'] for m in credit_metrics) / len(credit_metrics),
                    'precision': sum(m['precision'] for m in credit_metrics) / len(credit_metrics),
                    'recall': sum(m['recall'] for m in credit_metrics) / len(credit_metrics),
                    'f1_score': sum(m['f1_score'] for m in credit_metrics) / len(credit_metrics),
                    'timestamp': datetime.now().isoformat()
                }
                metrics.append(credit_avg)
            else:
                # Add default credit risk metrics if no data
                metrics.append({
                    'model_type': 'credit_risk',
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'timestamp': datetime.now().isoformat()
                })
            
            if fraud_metrics:
                fraud_avg = {
                    'model_type': 'fraud_detection',
                    'accuracy': sum(m['accuracy'] for m in fraud_metrics) / len(fraud_metrics),
                    'precision': sum(m['precision'] for m in fraud_metrics) / len(fraud_metrics),
                    'recall': sum(m['recall'] for m in fraud_metrics) / len(fraud_metrics),
                    'f1_score': sum(m['f1_score'] for m in fraud_metrics) / len(fraud_metrics),
                    'timestamp': datetime.now().isoformat()
                }
                metrics.append(fraud_avg)
            else:
                # Add default fraud detection metrics if no data
                metrics.append({
                    'model_type': 'fraud_detection',
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'timestamp': datetime.now().isoformat()
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting model metrics: {str(e)}")
            # Return default metrics on error
            return [
                {
                    'model_type': 'credit_risk',
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'model_type': 'fraud_detection',
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
            ]
    
    def _load_performance_history(self):
        """Load historical performance data from the performance log file."""
        try:
            if os.path.exists(self.performance_log_path):
                with open(self.performance_log_path, 'r') as f:
                    self.performance_data = json.load(f)
            else:
                self.performance_data = {
                    'risk_model': [],
                    'fraud_model': []
                }
        except Exception as e:
            logger.error(f"Error loading performance history: {str(e)}")
            self.performance_data = {
                'risk_model': [],
                'fraud_model': []
            }