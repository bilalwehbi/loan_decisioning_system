# src/api/loan_api.py
import os
import sys
import logging
import time
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import uuid
from datetime import datetime

# Add the project root to the Python path
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

from src.utils.schemas import LoanRequest, LoanDecision, ExplanationItem
from src.models.loan_decisioning_service import LoanDecisioningService
from src.models.model_feedback_loop import ModelFeedbackLoop
from src.configs.config import API_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api.log')
    ]
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Loan Decisioning API",
    description="API for AI-powered loan decisioning with credit risk scoring and fraud detection",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service
decisioning_service = LoanDecisioningService()
feedback_loop = ModelFeedbackLoop()

# Define outcome feedback model
class OutcomeFeedback(BaseModel):
    application_id: str = Field(..., description="Unique identifier for the loan application")
    actual_default: int = Field(..., description="Actual default outcome (1=default, 0=repaid)")
    actual_fraud: int = Field(..., description="Actual fraud outcome (1=fraud, 0=legitimate)")
    outcome_date: Optional[str] = Field(None, description="Date when the outcome was observed")

# Startup event
@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Starting Loan Decisioning API")
        decisioning_service.load_models()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise Exception(f"Failed to initialize loan decisioning service: {str(e)}")

# Health check endpoint
@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if models are loaded
        if not decisioning_service.risk_model or not decisioning_service.fraud_model:
            raise Exception("Models not loaded")
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Service not healthy: {str(e)}"
        )

# Predict endpoint
@app.post("/api/v1/assess", response_model=LoanDecision)
async def assess_loan(loan_request: LoanRequest, background_tasks: BackgroundTasks):
    """
    Assess a loan application
    
    Args:
        loan_request: Loan application data
        
    Returns:
        LoanDecision: Comprehensive decision with credit score, fraud checks, and explanations
    """
    try:
        # Validate required fields
        # (Removed manual required_fields check; Pydantic model validation is sufficient)
        
        # Make prediction
        decision = decisioning_service.make_decision(loan_request.dict())
        
        # Log prediction asynchronously (doesn't block response)
        background_tasks.add_task(feedback_loop.log_prediction, decision)
        
        return decision
    except ValueError as e:
        logger.error(f"Invalid loan application data: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Application history endpoint
@app.get("/api/v1/applications/history")
async def get_application_history(
    start_date: datetime,
    end_date: datetime,
    limit: int = 10
):
    """Get application history within a date range"""
    try:
        if start_date > end_date:
            raise HTTPException(status_code=400, detail="Start date must be before end date")
        
        history = decisioning_service.get_application_history(start_date, end_date, limit=limit)
        return history
    except ValueError as e:
        logger.error(f"Invalid date range: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting application history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Model metrics endpoint
@app.get("/api/v1/models/metrics")
async def get_model_metrics(days: int = 7):
    """Get model performance metrics"""
    try:
        metrics = feedback_loop.get_model_metrics(days)
        return [{"model_name": m["model_type"], **m} for m in metrics]
    except Exception as e:
        logger.error(f"Error getting model metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Employment validation endpoint
@app.post("/api/v1/validate/employment")
async def validate_employment(employment_data: dict):
    """Validate employment information"""
    try:
        result = decisioning_service.validate_employment(employment_data)
        return result
    except ValueError as e:
        logger.error(f"Invalid employment data: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error validating employment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Experiment endpoints
@app.post("/api/v1/experiments")
async def create_experiment(experiment_data: dict):
    """Create a new A/B test experiment"""
    try:
        result = feedback_loop.create_experiment(experiment_data)
        return result
    except ValueError as e:
        logger.error(f"Invalid experiment data: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating experiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/experiments/{experiment_id}")
async def get_experiment_results(experiment_id: str):
    """Get results for a specific experiment"""
    try:
        results = feedback_loop.get_experiment_results(experiment_id)
        return results
    except ValueError as e:
        logger.error(f"Invalid experiment ID: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting experiment results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/experiments/{experiment_id}/end")
async def end_experiment(experiment_id: str):
    """End an experiment"""
    try:
        result = feedback_loop.end_experiment(experiment_id)
        return result
    except ValueError as e:
        logger.error(f"Invalid experiment ID: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error ending experiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Model retraining endpoint
@app.post("/api/v1/models/retrain")
async def retrain_models(model_type: str = None, force: bool = False):
    """Trigger model retraining"""
    try:
        # Use decisioning_service for retraining
        if hasattr(decisioning_service, 'retrain_models'):
            result = decisioning_service.retrain_models(model_type, force)
        else:
            result = decisioning_service.retrain_model(model_type)
        return result
    except ValueError as e:
        logger.error(f"Invalid model type: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error retraining models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Document validation endpoints
@app.post("/api/v1/validate/paystub")
async def validate_paystub(application_id: str, paystub_image: bytes):
    """Validate paystub document"""
    try:
        result = decisioning_service.validate_paystub(application_id, paystub_image)
        return result
    except ValueError as e:
        logger.error(f"Invalid paystub data: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error validating paystub: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/validate/bank-transactions")
async def validate_bank_transactions(transactions: dict):
    """Validate bank transactions"""
    try:
        result = decisioning_service.validate_bank_transactions(transactions)
        return result
    except ValueError as e:
        logger.error(f"Invalid transaction data: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error validating bank transactions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Fraud check endpoints
@app.post("/fraud/check")
async def fraud_check(data: dict):
    """Perform fraud check with optional historical data"""
    try:
        result = decisioning_service.check_fraud(data)
        return result
    except ValueError as e:
        logger.error(f"Invalid fraud check data: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error performing fraud check: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Custom scoring thresholds endpoint
@app.post("/api/v1/config/thresholds")
async def update_scoring_thresholds(thresholds: dict):
    """Update custom scoring thresholds for risk segmentation"""
    try:
        result = decisioning_service.update_thresholds(thresholds)
        return {"status": "success", "message": "Thresholds updated successfully", "new_thresholds": result}
    except ValueError as e:
        logger.error(f"Invalid threshold data: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating thresholds: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced risk segmentation endpoint
@app.post("/api/v1/assess/enhanced")
async def assess_loan_enhanced(loan_request: LoanRequest, background_tasks: BackgroundTasks):
    """
    Enhanced loan assessment with detailed risk segmentation and alternative data
    
    Args:
        loan_request: Loan application data with optional alternative data sources
        
    Returns:
        Enhanced loan decision with detailed risk factors and segmentation
    """
    try:
        # Make enhanced prediction with alternative data
        decision = decisioning_service.make_enhanced_decision(loan_request.dict())
        
        # Log prediction asynchronously
        background_tasks.add_task(feedback_loop.log_prediction, decision)
        
        return decision
    except ValueError as e:
        logger.error(f"Invalid loan application data: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error making enhanced prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Performance monitoring endpoint
@app.get("/api/v1/monitoring/performance")
async def get_performance_metrics(
    start_date: datetime,
    end_date: datetime,
    metric_type: str = "all"
):
    """Get detailed performance metrics for monitoring"""
    try:
        metrics = decisioning_service.get_performance_metrics(start_date, end_date, metric_type)
        return metrics
    except ValueError as e:
        logger.error(f"Invalid date range or metric type: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Compliance reporting endpoint
@app.get("/api/v1/compliance/report")
async def get_compliance_report(
    start_date: datetime,
    end_date: datetime,
    report_type: str = "full"
):
    """Generate compliance report for audit purposes"""
    try:
        report = decisioning_service.generate_compliance_report(start_date, end_date, report_type)
        return report
    except ValueError as e:
        logger.error(f"Invalid date range or report type: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating compliance report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Main function to run the API server
def main():
    uvicorn.run(
        "loan_api:app",
        host=API_CONFIG.get("host", "0.0.0.0"),
        port=API_CONFIG.get("port", 8000),
        reload=True
    )

if __name__ == "__main__":
    main()