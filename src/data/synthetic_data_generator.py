# src/data/synthetic_data_generator.py
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import uuid
from typing import Dict, List, Tuple

def generate_credit_data(num_samples: int, applicant_ids=None) -> pd.DataFrame:
    """Generate synthetic credit bureau data"""

    # Use provided applicant_ids or generate new ones
    if applicant_ids is None:
        applicant_ids = [str(uuid.uuid4()) for _ in range(num_samples)]

    data = {
        
        "applicant_id": applicant_ids,
        "credit_score": np.random.randint(300, 851, num_samples),
        "num_accounts": np.random.randint(0, 20, num_samples),
        "num_active_accounts": [],
        "credit_utilization": np.random.beta(2, 5, num_samples),
        "num_delinquencies_30d": np.random.randint(0, 5, num_samples),
        "num_delinquencies_60d": np.random.randint(0, 3, num_samples),
        "num_delinquencies_90d": np.random.randint(0, 2, num_samples),
        "num_collections": np.random.randint(0, 3, num_samples),
        "total_debt": np.random.exponential(10000, num_samples),
        "inquiries_last_6mo": np.random.randint(0, 10, num_samples),
        "longest_credit_length_months": np.random.randint(0, 300, num_samples),
    }
    
    # Generate active accounts as a subset of total accounts
    for i in range(num_samples):
        data["num_active_accounts"].append(min(
            data["num_accounts"][i], 
            np.random.randint(0, data["num_accounts"][i] + 1)
        ))
    
    return pd.DataFrame(data)

def generate_banking_data(num_samples: int, applicant_ids=None) -> pd.DataFrame:
    """Generate synthetic banking transaction data"""
    
    # Use provided applicant_ids or generate new ones
    if applicant_ids is None:
        applicant_ids = [str(uuid.uuid4()) for _ in range(num_samples)]

    # First generate incomes with a realistic distribution
    monthly_incomes = np.random.lognormal(8, 0.6, num_samples)
    
    data = {
        "applicant_id": applicant_ids,
        "monthly_income": monthly_incomes,
        "avg_monthly_deposits": [],
        "avg_monthly_withdrawals": [],
        "income_stability_score": np.random.beta(5, 2, num_samples),
        "num_income_sources": np.random.randint(1, 4, num_samples),
        "avg_daily_balance": [],
        "num_nsf_transactions": np.random.randint(0, 5, num_samples),
        "num_overdrafts": np.random.randint(0, 6, num_samples),
    }
    
    # Make deposits and withdrawals related to income
    for i in range(num_samples):
        income = data["monthly_income"][i]
        # Deposits slightly higher than income (some may have multiple sources)
        data["avg_monthly_deposits"].append(income * np.random.uniform(0.8, 1.2))
        # Withdrawals typically less than or equal to deposits
        data["avg_monthly_withdrawals"].append(
            data["avg_monthly_deposits"][i] * np.random.uniform(0.7, 1.1)
        )
        # Average daily balance
        data["avg_daily_balance"].append(
            income * np.random.uniform(0.1, 1.0)
        )
    
    return pd.DataFrame(data)

def generate_application_data(num_samples: int, applicant_ids=None) -> pd.DataFrame:
    """Generate synthetic application metadata"""
        # Use provided applicant_ids or generate new ones
    if applicant_ids is None:
        applicant_ids = [str(uuid.uuid4()) for _ in range(num_samples)]
    
    now = datetime.now()
    
    ip_prefixes = ["192.168.", "10.0.", "172.16.", "98.124.", "74.125."]
    email_domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com", "icloud.com"]
    
    data = {
        "applicant_id": applicant_ids,
        "ip_address": [
            f"{random.choice(ip_prefixes)}{np.random.randint(0, 256)}.{np.random.randint(0, 256)}"
            for _ in range(num_samples)
        ],
        "device_id": [f"device-{uuid.uuid4().hex[:12]}" for _ in range(num_samples)],
        "email_domain": [random.choice(email_domains) for _ in range(num_samples)],
        "application_timestamp": [
            (now - timedelta(days=np.random.randint(0, 365))).isoformat()
            for _ in range(num_samples)
        ],
        "time_spent_on_application": np.random.randint(60, 1800, num_samples),
        "num_previous_applications": np.random.randint(0, 5, num_samples),
    }
    
    return pd.DataFrame(data)

def generate_loan_data(num_samples: int, applicant_ids=None) -> pd.DataFrame:
    """Generate synthetic loan application data"""
    
    # Use provided applicant_ids or generate new ones
    if applicant_ids is None:
        applicant_ids = [str(uuid.uuid4()) for _ in range(num_samples)]
    
    loan_purposes = ["Debt consolidation", "Home improvement", "Medical expenses", 
                    "Education", "Major purchase", "Vehicle", "Other"]
    
    data = {
        "applicant_id": applicant_ids,
        "loan_amount": np.random.uniform(500, 10000, num_samples),
        "loan_term_months": np.random.choice([6, 12, 24, 36, 48, 60], num_samples),
        "loan_purpose": [random.choice(loan_purposes) for _ in range(num_samples)],
        "applicant_income": np.random.lognormal(11, 0.7, num_samples),  # Annual income
        "applicant_employment_length_years": np.random.uniform(0, 35, num_samples),
    }
    
    return pd.DataFrame(data)

def generate_outcome_data(
    df_credit: pd.DataFrame, 
    df_banking: pd.DataFrame, 
    df_application: pd.DataFrame,
    df_loan: pd.DataFrame
) -> pd.DataFrame:
    """Generate synthetic loan outcome data based on other features"""
    
    # First merge all dataframes
    df_merged = df_loan.copy()
    
    # Create mapping dictionaries for faster lookups
    credit_dict = df_credit.set_index("applicant_id").to_dict("index")
    banking_dict = df_banking.set_index("applicant_id").to_dict("index")
    app_dict = df_application.set_index("applicant_id").to_dict("index")
    
    data = {
        "applicant_id": df_merged["applicant_id"],
        "loan_status": [],
        "days_to_default": [],
        "fraud_flag": [],
    }
    
    for applicant_id in data["applicant_id"]:
        # Get data for this applicant
        credit = credit_dict.get(applicant_id, {})
        banking = banking_dict.get(applicant_id, {})
        app = app_dict.get(applicant_id, {})
        
        # Calculate default probability based on features
        default_prob = 0.05  # Base default rate
        
        # Credit factors
        if credit:
            # Higher credit score reduces default probability
            default_prob -= (credit.get("credit_score", 600) - 500) / 1000
            # Delinquencies increase default probability
            default_prob += 0.02 * credit.get("num_delinquencies_30d", 0)
            default_prob += 0.03 * credit.get("num_delinquencies_60d", 0)
            default_prob += 0.05 * credit.get("num_delinquencies_90d", 0)
            # Collections increase default probability
            default_prob += 0.05 * credit.get("num_collections", 0)
            # High utilization increases default probability
            default_prob += 0.10 * credit.get("credit_utilization", 0.5)
        
        # Banking factors
        if banking:
            # Income stability decreases default probability
            default_prob -= 0.10 * banking.get("income_stability_score", 0.5)
            # NSF and overdrafts increase default probability
            default_prob += 0.01 * banking.get("num_nsf_transactions", 0)
            default_prob += 0.01 * banking.get("num_overdrafts", 0)
        
        # Application factors - suspicious applications increase probability
        if app:
            if app.get("time_spent_on_application", 300) < 120:
                default_prob += 0.05
        
        # Ensure probability is between 0 and 1
        # Ensure probability is between 0.01 and 0.8 (cap maximum default probability)
        default_prob = max(0.01, min(0.40, default_prob))
        
        # Determine loan status based on probability
        if np.random.random() < default_prob:
            status = "defaulted"
            # Days to default (typically 30-180 days)
            days = np.random.randint(30, 181)
        else:
            status = "repaid"
            days = 0
        
        data["loan_status"].append(status)
        data["days_to_default"].append(days)
        
        # Determine fraud
        fraud_prob = 0.01  # Base fraud rate
        
        # Application factors that might indicate fraud
        if app:
            time_spent = app.get("time_spent_on_application", 300)
            prev_apps = app.get("num_previous_applications", 0)
            
            # Very quick applications or many previous applications 
            # may indicate fraud
            if time_spent < 100:
                fraud_prob += 0.10
            if prev_apps > 3:
                fraud_prob += 0.05
        
        # Banking factors that might indicate fraud
        if banking and credit:
            stated_income = banking.get("monthly_income", 3000)
            deposits = banking.get("avg_monthly_deposits", 3000)
            
            # Large discrepancy between stated income and deposits
            if stated_income > deposits * 2:
                fraud_prob += 0.15
        

        # Cap fraud probability
        fraud_prob = min(0.30, fraud_prob)        
        # Determine if this is a fraudulent application
        data["fraud_flag"].append(1 if np.random.random() < fraud_prob else 0)
    
    return pd.DataFrame(data)

def generate_full_dataset(num_samples: int, output_dir: str) -> None:
    """Generate and save complete synthetic dataset"""
    
    print(f"Generating synthetic dataset with {num_samples} samples...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate applicant_ids first to ensure consistency across datasets
    applicant_ids = [str(uuid.uuid4()) for _ in range(num_samples)]
    
    # Generate individual datasets with the same applicant_ids
    df_credit = generate_credit_data(num_samples, applicant_ids)
    df_banking = generate_banking_data(num_samples, applicant_ids)
    df_application = generate_application_data(num_samples, applicant_ids)
    df_loan = generate_loan_data(num_samples, applicant_ids)
    
    # Generate outcomes dataset
    df_outcome = generate_outcome_data(df_credit, df_banking, df_application, df_loan)
    
    # Save datasets to CSV files
    df_credit.to_csv(os.path.join(output_dir, "credit_data.csv"), index=False)
    df_banking.to_csv(os.path.join(output_dir, "banking_data.csv"), index=False)
    df_application.to_csv(os.path.join(output_dir, "application_data.csv"), index=False)
    df_loan.to_csv(os.path.join(output_dir, "loan_data.csv"), index=False)
    df_outcome.to_csv(os.path.join(output_dir, "outcome_data.csv"), index=False)
    
    # Create a merged dataset for training
    print("Creating merged dataset for model training...")
    
    # Create copies to avoid SettingWithCopyWarning
    df_credit_copy = df_credit.copy()
    df_banking_copy = df_banking.copy()
    df_application_copy = df_application.copy()
    df_loan_copy = df_loan.copy()
    df_outcome_copy = df_outcome.copy()
    
    # Set applicant_id as index for easier merging
    df_credit_copy.set_index("applicant_id", inplace=True)
    df_banking_copy.set_index("applicant_id", inplace=True)
    df_application_copy.set_index("applicant_id", inplace=True)
    df_loan_copy.set_index("applicant_id", inplace=True)
    df_outcome_copy.set_index("applicant_id", inplace=True)
    
    # Merge all dataframes
    df_merged = pd.concat([df_loan_copy, df_credit_copy, df_banking_copy, df_application_copy, df_outcome_copy], axis=1)
    
    # Reset index to make applicant_id a column again
    df_merged.reset_index(inplace=True)
    
    # Save merged dataset
    df_merged.to_csv(os.path.join(output_dir, "merged_dataset.csv"), index=False)
    
    print(f"Datasets saved to {output_dir}")
    print(f"Total samples: {len(df_merged)}")
    print(f"Default rate: {df_merged['loan_status'].value_counts(normalize=True)['defaulted']:.2%}")
    print(f"Fraud rate: {df_merged['fraud_flag'].value_counts(normalize=True)[1]:.2%}")

if __name__ == "__main__":
    from src.configs.config import DATA_DIR
    # Generate 50,000 synthetic loan applications
    generate_full_dataset(50000, DATA_DIR)