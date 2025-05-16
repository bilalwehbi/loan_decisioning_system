import os
from src.models.performance import ModelPerformanceAnalyzer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Create output directory for reports
    os.makedirs("reports", exist_ok=True)
    
    # Initialize the analyzer
    analyzer = ModelPerformanceAnalyzer(
        risk_model_path="models/risk_model.joblib",
        fraud_model_path="models/fraud_model.joblib"
    )
    
    try:
        # 1. Print detailed performance report
        logger.info("Generating performance report...")
        analyzer.print_performance_report()
        
        # 2. Generate performance trend plots
        logger.info("Generating performance trend plots...")
        metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        for metric in metrics:
            analyzer.plot_performance_trends(
                metric=metric,
                save_path=f"reports/{metric}_trends.png"
            )
        
        # 3. Generate feature importance plots
        logger.info("Generating feature importance plots...")
        analyzer.plot_feature_importance(
            model_type="risk",
            top_n=10,
            save_path="reports/risk_feature_importance.png"
        )
        analyzer.plot_feature_importance(
            model_type="fraud",
            top_n=10,
            save_path="reports/fraud_feature_importance.png"
        )
        
        # 4. Save performance summary to file
        logger.info("Saving performance summary...")
        summary = analyzer.get_performance_summary()
        with open("reports/performance_summary.txt", "w") as f:
            for model_type, model_summary in summary.items():
                f.write(f"\n{model_type.upper()} MODEL\n")
                f.write("-" * 50 + "\n")
                
                f.write("\nCurrent Performance:\n")
                for metric, value in model_summary["current_performance"].items():
                    f.write(f"{metric:12}: {value:.4f}\n")
                
                f.write("\nPerformance Trends:\n")
                for metric, change in model_summary["performance_trend"].items():
                    f.write(f"{metric:12}: {change:+.4f}\n")
                
                f.write("\nTraining Statistics:\n")
                for stat, value in model_summary["training_stats"].items():
                    f.write(f"{stat:12}: {value}\n")
        
        logger.info("Analysis complete! Check the 'reports' directory for results.")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 