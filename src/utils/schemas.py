# src/utils/schemas.py
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field

class CreditFileData(BaseModel):
    """Credit bureau data structure"""
    credit_score: Optional[int] = Field(None, description="Traditional credit score (300-850)")
    num_accounts: int = Field(..., description="Number of credit accounts")
    num_active_accounts: int = Field(..., description="Number of active credit accounts")
    credit_utilization: float = Field(..., description="Credit utilization ratio (0-1)")
    num_delinquencies_30d: int = Field(..., description="Number of 30-day delinquencies")
    num_delinquencies_60d: int = Field(..., description="Number of 60-day delinquencies")
    num_delinquencies_90d: int = Field(..., description="Number of 90-day delinquencies")
    num_collections: int = Field(..., description="Number of accounts in collections")
    total_debt: float = Field(..., description="Total outstanding debt")
    inquiries_last_6mo: int = Field(..., description="Credit inquiries in last 6 months")
    longest_credit_length_months: int = Field(..., description="Months since oldest account opened")
    
class BankingData(BaseModel):
    """Banking transaction data structure"""
    avg_monthly_deposits: float = Field(..., description="Average monthly deposits")
    avg_monthly_withdrawals: float = Field(..., description="Average monthly withdrawals")
    monthly_income: float = Field(..., description="Stated monthly income")
    income_stability_score: float = Field(..., description="Income stability (0-1)")
    num_income_sources: int = Field(..., description="Number of income sources")
    avg_daily_balance: float = Field(..., description="Average daily balance")
    num_nsf_transactions: int = Field(..., description="Number of NSF transactions")
    num_overdrafts: int = Field(..., description="Number of overdrafts")
    
class ApplicationData(BaseModel):
    """Application metadata structure"""
    ip_address: str = Field(..., description="IP address")
    device_id: str = Field(..., description="Device identifier")
    email_domain: str = Field(..., description="Email domain")
    application_timestamp: str = Field(..., description="Timestamp of application")
    time_spent_on_application: int = Field(..., description="Seconds spent completing application")
    num_previous_applications: int = Field(..., description="Number of previous applications")
    
class LoanRequest(BaseModel):
    """Loan application request structure"""
    loan_amount: float = Field(..., description="Requested loan amount")
    loan_term_months: int = Field(..., description="Requested loan term in months")
    loan_purpose: str = Field(..., description="Purpose of the loan")
    credit_data: CreditFileData
    banking_data: BankingData
    application_data: ApplicationData
    applicant_income: float = Field(..., description="Stated annual income")
    applicant_employment_length_years: float = Field(..., description="Years at current employer")
    
class ExplanationItem(BaseModel):
    """Single explanation factor"""
    feature: str = Field(..., description="Feature name")
    contribution: float = Field(..., description="SHAP contribution value")
    description: str = Field(..., description="Human-readable explanation")
    
class LoanDecision(BaseModel):
    """Loan decision response structure"""
    application_id: str = Field(..., description="Unique application identifier")
    risk_score: int = Field(..., description="Model risk score (0-1000)")
    probability_of_default: float = Field(..., description="Predicted default probability")
    decision: str = Field(..., description="Decision: Low Risk, Review, or Decline")
    fraud_score: float = Field(..., description="Fraud risk score (0-1)")
    fraud_flags: List[str] = Field(default=[], description="Fraud warning flags")
    explanations: List[ExplanationItem] = Field(..., description="SHAP-based explanations")
    processing_time: float = Field(..., description="API processing time in seconds")