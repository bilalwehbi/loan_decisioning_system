import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import os
from scipy import stats

logger = logging.getLogger(__name__)

class ABTestingManager:
    """
    Manages A/B testing of different model versions and decisioning strategies
    """
    
    def __init__(self):
        self.experiments = {}
        self.results = {}
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        self.load_experiments()
    
    def load_experiments(self):
        """Load existing experiments from disk."""
        try:
            if os.path.exists("data/experiments.json"):
                with open("data/experiments.json", "r") as f:
                    self.experiments = json.load(f)
        except Exception as e:
            logger.error(f"Error loading experiments: {str(e)}")
    
    def save_experiments(self):
        """Save experiments to disk."""
        try:
            os.makedirs("data", exist_ok=True)
            with open("data/experiments.json", "w") as f:
                json.dump(self.experiments, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving experiments: {str(e)}")
    
    def create_experiment(self, experiment_id: str, description: str,
                         variants: List[Dict[str, Any]], duration_days: int = 14) -> Dict:
        """
        Create a new A/B test experiment
        
        Args:
            experiment_id: Unique experiment identifier
            description: Experiment description
            variants: List of model variants to test
            duration_days: Experiment duration in days
            
        Returns:
            Experiment configuration
        """
        try:
            # Validate variants
            if len(variants) < 2:
                raise ValueError("At least 2 variants required for A/B testing")
            
            # Create experiment
            experiment = {
                "experiment_id": experiment_id,
                "description": description,
                "variants": variants,
                "start_date": datetime.utcnow().isoformat(),
                "end_date": (datetime.utcnow() + timedelta(days=duration_days)).isoformat(),
                "status": "active",
                "results": {
                    "total_applications": 0,
                    "variant_results": {v["variant_id"]: {
                        "applications": 0,
                        "approvals": 0,
                        "defaults": 0,
                        "fraud_detections": 0,
                        "avg_processing_time": 0.0
                    } for v in variants}
                }
            }
            
            # Store experiment
            self.experiments[experiment_id] = experiment
            self.save_experiments()
            
            return experiment
            
        except Exception as e:
            logger.error(f"Error creating experiment: {str(e)}")
            raise
    
    def assign_variant(self, experiment_id: str, application_id: str) -> str:
        """
        Assign a variant to an application
        
        Args:
            experiment_id: Experiment identifier
            application_id: Application identifier
            
        Returns:
            Assigned variant ID
        """
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.experiments[experiment_id]
            if experiment["status"] != "active":
                raise ValueError(f"Experiment {experiment_id} is not active")
            
            # Simple random assignment
            variant = np.random.choice(experiment["variants"])
            return variant["variant_id"]
            
        except Exception as e:
            logger.error(f"Error assigning variant: {str(e)}")
            raise
    
    def record_result(self, experiment_id: str, application_id: str,
                     variant_id: str, result: Dict[str, Any]):
        """
        Record an application result for an experiment
        
        Args:
            experiment_id: Experiment identifier
            application_id: Application identifier
            variant_id: Assigned variant ID
            result: Application result data
        """
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.experiments[experiment_id]
            if variant_id not in experiment["results"]["variant_results"]:
                raise ValueError(f"Variant {variant_id} not found in experiment")
            
            # Update results
            variant_results = experiment["results"]["variant_results"][variant_id]
            variant_results["applications"] += 1
            if result["approved"]:
                variant_results["approvals"] += 1
            if result.get("defaulted", False):
                variant_results["defaults"] += 1
            if result.get("fraud_detected", False):
                variant_results["fraud_detections"] += 1
            
            # Update average processing time
            current_avg = variant_results["avg_processing_time"]
            new_time = result.get("processing_time", 0)
            variant_results["avg_processing_time"] = (
                (current_avg * (variant_results["applications"] - 1) + new_time) /
                variant_results["applications"]
            )
            
            experiment["results"]["total_applications"] += 1
            
            # Save updated results
            self.save_experiments()
            
        except Exception as e:
            logger.error(f"Error recording result: {str(e)}")
            raise
    
    def get_experiment_results(self, experiment_id: str) -> Dict:
        """
        Get results for an experiment
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Experiment results
        """
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.experiments[experiment_id]
            results = experiment["results"]
            
            # Calculate statistical significance
            if results["total_applications"] > 0:
                for variant_id, variant_results in results["variant_results"].items():
                    if variant_results["applications"] > 0:
                        # Calculate approval rate
                        approval_rate = variant_results["approvals"] / variant_results["applications"]
                        variant_results["approval_rate"] = approval_rate
                        
                        # Calculate default rate
                        if variant_results["approvals"] > 0:
                            default_rate = variant_results["defaults"] / variant_results["approvals"]
                            variant_results["default_rate"] = default_rate
                        
                        # Calculate fraud detection rate
                        fraud_rate = variant_results["fraud_detections"] / variant_results["applications"]
                        variant_results["fraud_rate"] = fraud_rate
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting experiment results: {str(e)}")
            raise
    
    def analyze_experiment(self, experiment_id: str) -> Dict:
        """
        Perform statistical analysis of experiment results
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Analysis results
        """
        try:
            results = self.get_experiment_results(experiment_id)
            analysis = {
                "experiment_id": experiment_id,
                "total_applications": results["total_applications"],
                "variant_analysis": {},
                "statistical_tests": {}
            }
            
            # Analyze each variant
            for variant_id, variant_results in results["variant_results"].items():
                if variant_results["applications"] > 0:
                    analysis["variant_analysis"][variant_id] = {
                        "approval_rate": variant_results.get("approval_rate", 0),
                        "default_rate": variant_results.get("default_rate", 0),
                        "fraud_rate": variant_results.get("fraud_rate", 0),
                        "avg_processing_time": variant_results["avg_processing_time"]
                    }
            
            # Perform statistical tests if enough data
            if results["total_applications"] >= 100:
                # Compare approval rates
                approval_rates = []
                for variant_id, variant_results in results["variant_results"].items():
                    if variant_results["applications"] > 0:
                        approval_rates.append({
                            "variant_id": variant_id,
                            "rate": variant_results.get("approval_rate", 0),
                            "n": variant_results["applications"]
                        })
                
                if len(approval_rates) >= 2:
                    # Perform chi-square test
                    observed = [r["rate"] * r["n"] for r in approval_rates]
                    expected = [sum(observed) / len(observed)] * len(observed)
                    chi2, p_value = stats.chisquare(observed, expected)
                    
                    analysis["statistical_tests"]["approval_rates"] = {
                        "test": "chi-square",
                        "chi2": float(chi2),
                        "p_value": float(p_value),
                        "significant": p_value < 0.05
                    }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing experiment: {str(e)}")
            raise
    
    def end_experiment(self, experiment_id: str) -> Dict:
        """
        End an experiment and get final results
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Final experiment results and analysis
        """
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.experiments[experiment_id]
            experiment["status"] = "completed"
            experiment["end_date"] = datetime.utcnow().isoformat()
            
            # Get final results and analysis
            results = self.get_experiment_results(experiment_id)
            analysis = self.analyze_experiment(experiment_id)
            
            # Save updated experiment
            self.save_experiments()
            
            return {
                "experiment": experiment,
                "results": results,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Error ending experiment: {str(e)}")
            raise 