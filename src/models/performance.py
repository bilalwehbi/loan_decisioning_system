import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from .training import ModelTrainer

class ModelPerformanceAnalyzer:
    def __init__(self, risk_model_path: str, fraud_model_path: str):
        """Initialize the performance analyzer with model paths."""
        self.risk_trainer = ModelTrainer.load_model(risk_model_path)
        self.fraud_trainer = ModelTrainer.load_model(fraud_model_path)
    
    def get_current_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get current performance metrics for both models."""
        return {
            "risk_model": {
                "metrics": self.risk_trainer.metrics,
                "last_training_date": self.risk_trainer.last_training_date,
                "feature_importance": self.risk_trainer.feature_importance
            },
            "fraud_model": {
                "metrics": self.fraud_trainer.metrics,
                "last_training_date": self.fraud_trainer.last_training_date,
                "feature_importance": self.fraud_trainer.feature_importance
            }
        }
    
    def get_performance_history(self) -> Dict[str, pd.DataFrame]:
        """Get performance history as DataFrames for easier analysis."""
        risk_history = pd.DataFrame(self.risk_trainer.training_history)
        fraud_history = pd.DataFrame(self.fraud_trainer.training_history)
        
        return {
            "risk_model": risk_history,
            "fraud_model": fraud_history
        }
    
    def plot_performance_trends(self, metric: str = "accuracy", save_path: str = None):
        """Plot performance trends over time for both models."""
        risk_history = pd.DataFrame(self.risk_trainer.training_history)
        fraud_history = pd.DataFrame(self.fraud_trainer.training_history)
        
        plt.figure(figsize=(12, 6))
        
        # Plot risk model performance
        plt.plot(risk_history['training_date'], risk_history[metric], 
                label='Risk Model', marker='o')
        
        # Plot fraud model performance
        plt.plot(fraud_history['training_date'], fraud_history[metric], 
                label='Fraud Model', marker='s')
        
        plt.title(f'{metric.capitalize()} Over Time')
        plt.xlabel('Training Date')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_feature_importance(self, model_type: str = "risk", top_n: int = 10, save_path: str = None):
        """Plot feature importance for the specified model."""
        trainer = self.risk_trainer if model_type == "risk" else self.fraud_trainer
        
        # Get feature importance and sort
        importance = pd.DataFrame({
            'feature': list(trainer.feature_importance.keys()),
            'importance': list(trainer.feature_importance.values())
        })
        importance = importance.sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=importance)
        plt.title(f'Top {top_n} Features - {model_type.capitalize()} Model')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def get_performance_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get a comprehensive performance summary for both models."""
        risk_history = pd.DataFrame(self.risk_trainer.training_history)
        fraud_history = pd.DataFrame(self.fraud_trainer.training_history)
        
        def calculate_summary(history: pd.DataFrame) -> Dict[str, Any]:
            return {
                "current_performance": {
                    "accuracy": history['accuracy'].iloc[-1],
                    "precision": history['precision'].iloc[-1],
                    "recall": history['recall'].iloc[-1],
                    "f1_score": history['f1_score'].iloc[-1],
                    "roc_auc": history['roc_auc'].iloc[-1]
                },
                "performance_trend": {
                    "accuracy_change": history['accuracy'].iloc[-1] - history['accuracy'].iloc[0],
                    "precision_change": history['precision'].iloc[-1] - history['precision'].iloc[0],
                    "recall_change": history['recall'].iloc[-1] - history['recall'].iloc[0],
                    "f1_score_change": history['f1_score'].iloc[-1] - history['f1_score'].iloc[0],
                    "roc_auc_change": history['roc_auc'].iloc[-1] - history['roc_auc'].iloc[0]
                },
                "training_stats": {
                    "total_updates": len(history),
                    "last_update": history['training_date'].iloc[-1],
                    "total_samples": history['n_samples'].sum()
                }
            }
        
        return {
            "risk_model": calculate_summary(risk_history),
            "fraud_model": calculate_summary(fraud_history)
        }
    
    def print_performance_report(self):
        """Print a detailed performance report for both models."""
        summary = self.get_performance_summary()
        
        print("\n=== Model Performance Report ===\n")
        
        for model_type, model_summary in summary.items():
            print(f"\n{model_type.upper()} MODEL")
            print("-" * 50)
            
            print("\nCurrent Performance:")
            for metric, value in model_summary["current_performance"].items():
                print(f"{metric:12}: {value:.4f}")
            
            print("\nPerformance Trends:")
            for metric, change in model_summary["performance_trend"].items():
                print(f"{metric:12}: {change:+.4f}")
            
            print("\nTraining Statistics:")
            for stat, value in model_summary["training_stats"].items():
                print(f"{stat:12}: {value}")
            
            print("\nTop 5 Important Features:")
            trainer = self.risk_trainer if model_type == "risk_model" else self.fraud_trainer
            importance = pd.DataFrame({
                'feature': list(trainer.feature_importance.keys()),
                'importance': list(trainer.feature_importance.values())
            })
            importance = importance.sort_values('importance', ascending=False).head()
            for _, row in importance.iterrows():
                print(f"{row['feature']:12}: {row['importance']:.4f}")
            
            print("\n" + "=" * 50) 