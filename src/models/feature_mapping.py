# src/models/feature_mapping.py
import os
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Define display-friendly feature names for visualization
FEATURE_DISPLAY_NAMES = {
    # Credit features
    "numeric_credit_score": "Credit Score",
    "numeric_num_accounts": "# of Accounts",
    "numeric_num_active_accounts": "# of Active Accounts",
    "numeric_credit_utilization": "Credit Utilization",
    "numeric_num_delinquencies_30d": "30d Delinquencies",
    "numeric_num_delinquencies_60d": "60d Delinquencies",
    "numeric_num_delinquencies_90d": "90d Delinquencies",
    "numeric_num_collections": "Collections",
    "numeric_total_debt": "Total Debt",
    "numeric_inquiries_last_6mo": "Recent Inquiries",
    "numeric_longest_credit_length_months": "Credit History Length",

    # Banking features
    "numeric_avg_monthly_deposits": "Monthly Deposits",
    "numeric_avg_monthly_withdrawals": "Monthly Withdrawals",
    "numeric_monthly_income": "Stated Income",
    "numeric_income_stability_score": "Income Stability",
    "numeric_num_income_sources": "Income Sources",
    "numeric_avg_daily_balance": "Avg Balance",
    "numeric_num_nsf_transactions": "NSF Count",
    "numeric_num_overdrafts": "Overdrafts",

    # Application features
    "numeric_time_spent_on_application": "App Completion Time",
    "numeric_num_previous_applications": "Prior Applications",
    "numeric_loan_amount": "Loan Amount",
    "numeric_applicant_income": "Annual Income",
    "numeric_applicant_employment_length_years": "Employment Length",

    # Engineered features
    "engineered_dti_ratio": "DTI Ratio",
    "engineered_payment_to_income": "Payment/Income",
    "engineered_income_verification_ratio": "Income Verification",
    "engineered_high_utilization": "High Utilization",
    "engineered_has_delinquencies": "Has Delinquencies",
    "engineered_total_banking_negatives": "Banking Negatives",
    "engineered_loan_to_income": "Loan/Income Ratio",
    "engineered_credit_score_category": "Credit Category",

    # Fraud features
    "fraud_income_verification_ratio": "Income Verification",
    "fraud_income_mismatch_flag": "Income Mismatch",
    "fraud_high_velocity_flag": "Multiple Applications",
    "fraud_application_speed_concern": "Suspiciously Fast App",
    "fraud_free_email_domain": "Free Email",
    "fraud_private_ip_flag": "Private IP",
    "fraud_generic_device_flag": "Generic Device ID",
    "fraud_withdrawal_deposit_ratio": "Withdrawal/Deposit",
    "fraud_unusual_cash_flow": "Unusual Cash Flow",
    "fraud_banking_reliability_concern": "Banking Issues",
    "fraud_affordability_concern": "Affordability Concern",
    
    # Fallback for generic features
    "feature_0": "Feature 0",
    "feature_1": "Feature 1",
    "feature_2": "Feature 2",
    "feature_3": "Feature 3",
    "feature_4": "Feature 4",
    "feature_5": "Feature 5",
    "feature_6": "Feature 6",
    "feature_7": "Feature 7",
    "feature_8": "Feature 8",
    "feature_9": "Feature 9",
}

def get_display_name(feature_name: str) -> str:
    """
    Get a display-friendly name for a feature
    
    Args:
        feature_name: Original feature name
        
    Returns:
        display_name: User-friendly display name
    """
    # Check direct match
    if feature_name in FEATURE_DISPLAY_NAMES:
        return FEATURE_DISPLAY_NAMES[feature_name]
    
    # Check for partial matches (for categorical features with values)
    for prefix, display_name in FEATURE_DISPLAY_NAMES.items():
        if feature_name.startswith(prefix):
            # Extract value portion if present (e.g., categorical_loan_purpose_Other)
            if "_" in feature_name[len(prefix):]:
                value = feature_name[len(prefix):].split("_", 1)[1]
                return f"{display_name}: {value}"
            return display_name
    
    # Fallback to original name
    return feature_name

def create_feature_index_mapping(feature_names: List[str]) -> Dict[int, str]:
    """
    Create mapping from feature indices to display names
    
    Args:
        feature_names: List of original feature names
        
    Returns:
        mapping: Dictionary mapping indices to display names
    """
    mapping = {}
    for i, name in enumerate(feature_names):
        mapping[i] = get_display_name(name)
        
    return mapping

def save_mapping(mapping: Dict, filepath: str) -> None:
    """Save feature mapping to disk"""
    try:
        with open(filepath, 'w') as f:
            json.dump(mapping, f, indent=2)
        logger.info(f"Feature mapping saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving feature mapping: {str(e)}")

def load_mapping(filepath: str) -> Optional[Dict]:
    """Load feature mapping from disk"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                mapping = json.load(f)
            logger.info(f"Feature mapping loaded from {filepath}")
            return mapping
        else:
            logger.warning(f"Feature mapping file not found: {filepath}")
            return None
    except Exception as e:
        logger.error(f"Error loading feature mapping: {str(e)}")
        return None