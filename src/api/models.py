from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

class CreditData(BaseModel):
    credit_score: int = Field(..., ge=300, le=850, description="FICO credit score (300-850)")
    num_accounts: int = Field(..., ge=0, description="Number of credit accounts")
    num_active_accounts: int = Field(..., ge=0, description="Number of active credit accounts")
    credit_utilization: float = Field(..., ge=0, le=1, description="Credit utilization ratio (0-1)")
    num_delinquencies_30d: int = Field(..., ge=0, description="Number of 30-day delinquencies")
    num_delinquencies_60d: int = Field(..., ge=0, description="Number of 60-day delinquencies")
    num_delinquencies_90d: int = Field(..., ge=0, description="Number of 90-day delinquencies")
    num_collections: int = Field(..., ge=0, description="Number of accounts in collections")
    total_debt: float = Field(..., ge=0, description="Total outstanding debt")
    inquiries_last_6mo: int = Field(..., ge=0, description="Credit inquiries in last 6 months")
    longest_credit_length_months: int = Field(..., ge=0, description="Months since oldest account opened")

class BankingData(BaseModel):
    avg_monthly_deposits: float = Field(..., gt=0, description="Average monthly deposits")
    avg_monthly_withdrawals: float = Field(..., gt=0, description="Average monthly withdrawals")
    monthly_income: float = Field(..., gt=0, description="Stated monthly income")
    income_stability_score: float = Field(..., ge=0, le=1, description="Income stability (0-1)")
    num_income_sources: int = Field(..., ge=0, description="Number of income sources")
    avg_daily_balance: float = Field(..., ge=0, description="Average daily balance")
    num_nsf_transactions: int = Field(..., ge=0, description="Number of NSF transactions")
    num_overdrafts: int = Field(..., ge=0, description="Number of overdrafts")

class BehavioralData(BaseModel):
    application_completion_time: float = Field(..., gt=0, description="Time taken to complete application in seconds")
    form_fill_pattern: Dict[str, float] = Field(..., description="Timing patterns of form filling")
    device_trust_score: float = Field(..., ge=0, le=1, description="Device trust score (0-1)")
    location_risk_score: float = Field(..., ge=0, le=1, description="Location risk score (0-1)")
    digital_footprint_score: float = Field(..., ge=0, le=1, description="Digital footprint score (0-1)")
    social_media_presence: Dict[str, float] = Field(..., description="Social media presence metrics")

class EmploymentData(BaseModel):
    employer_name: str = Field(..., description="Current employer name")
    employment_length_months: int = Field(..., ge=0, description="Length of employment in months")
    employment_verification_score: float = Field(..., ge=0, le=1, description="Employment verification score (0-1)")
    income_verification_score: float = Field(..., ge=0, le=1, description="Income verification score (0-1)")
    paystub_analysis: Dict[str, float] = Field(..., description="Paystub analysis results")
    employment_history: List[Dict[str, Any]] = Field(..., description="Employment history details")

class ApplicationData(BaseModel):
    ip_address: str = Field(..., description="IP address")
    device_id: str = Field(..., min_length=1, description="Device identifier")
    email_domain: str = Field(..., description="Email domain")
    application_timestamp: str = Field(..., description="Timestamp of application")
    time_spent_on_application: int = Field(..., description="Seconds spent completing application")
    num_previous_applications: int = Field(..., description="Number of previous applications")

class LoanApplication(BaseModel):
    loan_amount: float = Field(..., gt=0, description="Requested loan amount")
    loan_term_months: int = Field(..., ge=12, le=84, description="Loan term in months (12-84)")
    loan_purpose: str = Field(..., description="Purpose of the loan")
    credit_data: CreditData
    banking_data: BankingData
    application_data: ApplicationData
    applicant_income: float = Field(..., gt=0, description="Stated annual income")
    applicant_employment_length_years: float = Field(..., gt=0, description="Years at current employer")

class RiskScore(BaseModel):
    score: int = Field(..., ge=300, le=850, description="Risk score (300-850)")
    probability_default: float = Field(..., ge=0, le=1, description="Probability of default")
    risk_segment: str = Field(..., description="Risk segment classification")
    top_factors: List[Dict[str, float]] = Field(..., description="Top risk factors")
    fraud_risk_score: float = Field(..., ge=0, le=1, description="Fraud risk score")
    fraud_flags: List[str] = Field(..., description="Fraud detection flags")
    explanation: Dict[str, str] = Field(..., description="Explanation of the decision")

class ModelMetrics(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    last_updated: datetime
    training_data_size: int
    performance_metrics: Dict[str, Dict[str, float]]

class ApplicationHistory(BaseModel):
    application_id: str
    timestamp: datetime
    loan_amount: float
    loan_term: int
    risk_score: int
    fraud_score: float
    decision: str
    review_required: bool
    review_status: Optional[str]

class RetrainRequest(BaseModel):
    model_type: str
    force: bool = False

class PaystubValidationRequest(BaseModel):
    paystub_image: bytes = Field(..., description="Paystub image in bytes")
    application_id: str = Field(..., description="Unique application identifier")

class BankTransactionValidationRequest(BaseModel):
    transactions: List[Dict] = Field(..., description="List of bank transactions")
    application_id: str = Field(..., description="Unique application identifier")

class EmploymentValidationRequest(BaseModel):
    employer: str = Field(..., description="Employer name")
    start_date: str = Field(..., description="Employment start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="Employment end date (YYYY-MM-DD)")
    job_title: str = Field(..., description="Job title")
    application_id: str = Field(..., description="Unique application identifier") 