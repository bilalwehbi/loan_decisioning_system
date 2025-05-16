from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime
from src.services.fraud_detector import FraudDetector

router = APIRouter(prefix="/fraud", tags=["fraud-detection"])

# Initialize fraud detector
fraud_detector = FraudDetector()

class ApplicationData(BaseModel):
    application_id: str = Field(..., description="Unique identifier for the application")
    name: str = Field(..., description="Applicant's full name")
    email: str = Field(..., description="Applicant's email address")
    phone: str = Field(..., description="Applicant's phone number")
    address: str = Field(..., description="Applicant's address")
    ip_address: str = Field(..., description="IP address of the application")
    device_id: str = Field(..., description="Unique device identifier")
    device_fingerprint: Dict = Field(..., description="Device fingerprint data")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(), description="Application timestamp")

class HistoricalData(BaseModel):
    applications: List[ApplicationData] = Field(..., description="List of historical applications")

class FraudCheckResponse(BaseModel):
    is_fraudulent: bool = Field(..., description="Whether the application is flagged as fraudulent")
    risk_level: float = Field(..., description="Overall risk level (0-1)")
    risk_scores: Dict = Field(..., description="Individual risk scores for different checks")
    patterns_detected: List[Dict] = Field(..., description="Detected fraud patterns")
    timestamp: datetime = Field(..., description="Timestamp of the fraud check")

@router.post("/check", response_model=FraudCheckResponse)
async def check_fraud(
    application: ApplicationData,
    historical_data: Optional[HistoricalData] = None
) -> FraudCheckResponse:
    """
    Perform comprehensive fraud check on an application.
    
    Args:
        application: Current application data
        historical_data: Optional historical application data for velocity checks
        
    Returns:
        FraudCheckResponse containing fraud detection results
    """
    try:
        # Convert Pydantic models to dicts
        app_data = application.model_dump()
        hist_data = historical_data.applications if historical_data else []
        
        # Run all fraud checks
        synthetic_result = await fraud_detector.detect_synthetic_id(app_data)
        device_result = await fraud_detector.analyze_device_risk(app_data)
        velocity_result = await fraud_detector.check_velocity(app_data, hist_data)
        collusion_result = await fraud_detector.detect_collusion(hist_data)
        
        # Combine results
        risk_scores = {
            'synthetic_id': synthetic_result.get('risk_level', 0.0),
            'device_risk': device_result.get('risk_level', 0.0),
            'velocity': velocity_result.get('risk_level', 0.0),
            'collusion': collusion_result.get('risk_level', 0.0)
        }
        
        # Calculate overall risk level (weighted average)
        weights = {
            'synthetic_id': 0.4,
            'device_risk': 0.3,
            'velocity': 0.2,
            'collusion': 0.1
        }
        
        overall_risk = sum(
            risk_scores[check] * weights[check]
            for check in risk_scores
        )
        
        # Collect all detected patterns
        patterns = []
        if synthetic_result.get('patterns_detected'):
            patterns.extend(synthetic_result['patterns_detected'])
        if device_result.get('patterns_detected'):
            patterns.extend(device_result['patterns_detected'])
        if velocity_result.get('threshold_violations'):
            patterns.extend(velocity_result['threshold_violations'])
        if collusion_result.get('patterns_detected'):
            patterns.extend(collusion_result['patterns_detected'])
        
        return FraudCheckResponse(
            is_fraudulent=overall_risk >= 0.7,
            risk_level=overall_risk,
            risk_scores=risk_scores,
            patterns_detected=patterns,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error performing fraud check: {str(e)}"
        )

@router.get("/health")
async def health_check() -> Dict:
    """
    Health check endpoint for the fraud detection service.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    } 