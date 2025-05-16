import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

os.makedirs("data", exist_ok=True)

def generate_realistic_credit_data(n_samples=200000):
    """Generate realistic credit risk data with correlations between features"""
    np.random.seed(42)
    
    # Base features with realistic distributions
    credit_score = np.random.normal(700, 50, n_samples).clip(300, 850)
    monthly_income = np.random.lognormal(8.5, 0.5, n_samples).clip(1000, 20000)
    
    # Correlated features
    utilization = np.random.beta(2, 5, n_samples)  # Most people have low utilization
    delinquencies = np.random.poisson(utilization * 5)  # Higher utilization -> more delinquencies
    inquiries = np.random.poisson(utilization * 3)  # Higher utilization -> more inquiries
    
    # Income stability (correlated with income)
    income_stability = 1 - (np.random.normal(0, 0.2, n_samples) + (monthly_income - monthly_income.mean()) / monthly_income.std() * 0.1).clip(-1, 1)
    
    # Spending pattern (correlated with income and utilization)
    spending_pattern = (income_stability * 0.7 + (1 - utilization) * 0.3 + np.random.normal(0, 0.1, n_samples)).clip(0, 1)
    
    # Calculate default probability based on features
    default_prob = (
        (1 - (credit_score - 300) / 550) * 0.3 +  # Lower credit score -> higher default
        utilization * 0.2 +  # Higher utilization -> higher default
        (delinquencies / 10) * 0.2 +  # More delinquencies -> higher default
        (1 - income_stability) * 0.2 +  # Lower income stability -> higher default
        (1 - spending_pattern) * 0.1  # Poor spending pattern -> higher default
    ).clip(0, 1)
    
    # Generate default target
    default = (default_prob > np.random.random(n_samples)).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        "credit_score": credit_score,
        "monthly_income": monthly_income,
        "utilization": utilization,
        "delinquencies": delinquencies,
        "inquiries_last_6m": inquiries,
        "income_stability_score": income_stability,
        "spending_pattern_score": spending_pattern,
        "is_default": default
    })
    
    return data

def generate_realistic_fraud_data(n_samples=200000):
    """Generate realistic fraud detection data with patterns"""
    np.random.seed(42)
    
    # Time-based features
    hour_of_day = np.random.randint(0, 24, n_samples)
    day_of_week = np.random.randint(0, 7, n_samples)
    
    # Device and contact features
    device_id_length = np.random.randint(5, 20, n_samples)
    email_domain_length = np.random.randint(5, 20, n_samples)
    phone_number_length = np.random.randint(10, 15, n_samples)
    
    # Calculate fraud probability based on patterns
    fraud_prob = (
        # Higher fraud probability during odd hours
        ((hour_of_day < 6) | (hour_of_day > 20)).astype(float) * 0.2 +
        # Higher fraud probability on weekends
        (day_of_week >= 5).astype(float) * 0.1 +
        # Suspicious device ID lengths
        ((device_id_length < 8) | (device_id_length > 15)).astype(float) * 0.2 +
        # Suspicious email domains
        ((email_domain_length < 8) | (email_domain_length > 15)).astype(float) * 0.2 +
        # Suspicious phone numbers
        (phone_number_length != 10).astype(float) * 0.3
    ).clip(0, 1)
    
    # Generate fraud target
    fraud = (fraud_prob > np.random.random(n_samples)).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "device_id_length": device_id_length,
        "email_domain_length": email_domain_length,
        "phone_number_length": phone_number_length,
        "is_fraud": fraud
    })
    
    return data

def generate_credit_risk_data(n_samples):
    data = {
        'credit_score': np.random.normal(650, 100, n_samples),
        'monthly_income': np.random.normal(5000, 2000, n_samples),
        'employment_length': np.random.normal(5, 3, n_samples),
        'loan_amount': np.random.normal(5000, 2000, n_samples),
        'loan_term': np.random.normal(24, 12, n_samples),
        'num_delinquencies': np.random.normal(1, 2, n_samples),
        'num_inquiries': np.random.normal(2, 1, n_samples),
        'utilization_rate': np.random.normal(0.3, 0.2, n_samples),
        'income_stability_score': np.random.normal(0.7, 0.2, n_samples),
        'spending_pattern_score': np.random.normal(0.6, 0.2, n_samples),
        'device_risk_score': np.random.normal(0.5, 0.2, n_samples),
        'ip_risk_score': np.random.normal(0.5, 0.2, n_samples),
        'time_spent_on_application': np.random.normal(10, 5, n_samples),
        'num_previous_applications': np.random.normal(1, 1, n_samples),
        'default': np.random.binomial(1, 0.2, n_samples)
    }
    return pd.DataFrame(data)

def generate_fraud_detection_data(n_samples):
    """Generate comprehensive fraud detection data with additional features"""
    np.random.seed(42)
    
    # Time-based features
    hour_of_day = np.random.randint(0, 24, n_samples)
    day_of_week = np.random.randint(0, 7, n_samples)
    
    # Device and contact features
    device_id_length = np.random.randint(10, 20, n_samples)
    email_domain_length = np.random.randint(5, 15, n_samples)
    phone_number_length = np.random.randint(10, 15, n_samples)
    
    # Credit-related features
    credit_score = np.random.normal(650, 100, n_samples).clip(300, 850)
    monthly_income = np.random.lognormal(8.5, 0.5, n_samples).clip(1000, 20000)
    employment_length = np.random.normal(5, 3, n_samples).clip(0, 40)
    loan_amount = np.random.normal(5000, 2000, n_samples).clip(1000, 10000)
    loan_term = np.random.normal(24, 12, n_samples).clip(12, 60)
    num_delinquencies = np.random.poisson(2, n_samples)
    num_inquiries = np.random.poisson(3, n_samples)
    utilization_rate = np.random.beta(2, 5, n_samples)
    
    # Behavioral features
    income_stability_score = np.random.normal(0.7, 0.2, n_samples).clip(0, 1)
    spending_pattern_score = np.random.normal(0.6, 0.2, n_samples).clip(0, 1)
    avg_monthly_deposits = np.random.lognormal(7, 0.5, n_samples)
    avg_monthly_withdrawals = np.random.lognormal(6.5, 0.5, n_samples)
    num_income_sources = np.random.poisson(2, n_samples)
    avg_daily_balance = np.random.lognormal(6, 0.5, n_samples)
    
    # Application behavior features
    time_spent_on_application = np.random.normal(10, 5, n_samples).clip(1, 30)
    num_previous_applications = np.random.poisson(2, n_samples)
    application_completion_rate = np.random.normal(0.8, 0.1, n_samples).clip(0, 1)
    form_fill_time = np.random.normal(5, 2, n_samples).clip(1, 15)
    
    # Risk scores
    device_risk_score = np.random.normal(0.5, 0.2, n_samples).clip(0, 1)
    ip_risk_score = np.random.normal(0.5, 0.2, n_samples).clip(0, 1)
    location_risk_score = np.random.normal(0.5, 0.2, n_samples).clip(0, 1)
    behavioral_risk_score = np.random.normal(0.5, 0.2, n_samples).clip(0, 1)
    
    # Calculate fraud probability based on patterns
    fraud_prob = (
        # Time-based patterns
        ((hour_of_day < 6) | (hour_of_day > 20)).astype(float) * 0.1 +
        (day_of_week >= 5).astype(float) * 0.05 +
        
        # Device and contact patterns
        ((device_id_length < 8) | (device_id_length > 15)).astype(float) * 0.1 +
        ((email_domain_length < 8) | (email_domain_length > 15)).astype(float) * 0.1 +
        (phone_number_length != 10).astype(float) * 0.1 +
        
        # Credit-related patterns
        (credit_score < 600).astype(float) * 0.1 +
        (monthly_income < 2000).astype(float) * 0.1 +
        (employment_length < 1).astype(float) * 0.1 +
        (num_delinquencies > 3).astype(float) * 0.1 +
        (num_inquiries > 5).astype(float) * 0.1 +
        (utilization_rate > 0.8).astype(float) * 0.1 +
        
        # Behavioral patterns
        (income_stability_score < 0.3).astype(float) * 0.1 +
        (spending_pattern_score < 0.3).astype(float) * 0.1 +
        (avg_monthly_deposits < 1000).astype(float) * 0.1 +
        (avg_monthly_withdrawals > 5000).astype(float) * 0.1 +
        (num_income_sources > 3).astype(float) * 0.1 +
        (avg_daily_balance < 100).astype(float) * 0.1 +
        
        # Application behavior patterns
        (time_spent_on_application < 2).astype(float) * 0.1 +
        (num_previous_applications > 5).astype(float) * 0.1 +
        (application_completion_rate < 0.5).astype(float) * 0.1 +
        (form_fill_time < 1).astype(float) * 0.1 +
        
        # Risk score patterns
        (device_risk_score > 0.8).astype(float) * 0.1 +
        (ip_risk_score > 0.8).astype(float) * 0.1 +
        (location_risk_score > 0.8).astype(float) * 0.1 +
        (behavioral_risk_score > 0.8).astype(float) * 0.1
    ).clip(0, 1)
    
    # Generate fraud target
    fraud = (fraud_prob > np.random.random(n_samples)).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        # Time-based features
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        
        # Device and contact features
        "device_id_length": device_id_length,
        "email_domain_length": email_domain_length,
        "phone_number_length": phone_number_length,
        
        # Credit-related features
        "credit_score": credit_score,
        "monthly_income": monthly_income,
        "employment_length": employment_length,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "num_delinquencies": num_delinquencies,
        "num_inquiries": num_inquiries,
        "utilization_rate": utilization_rate,
        
        # Behavioral features
        "income_stability_score": income_stability_score,
        "spending_pattern_score": spending_pattern_score,
        "avg_monthly_deposits": avg_monthly_deposits,
        "avg_monthly_withdrawals": avg_monthly_withdrawals,
        "num_income_sources": num_income_sources,
        "avg_daily_balance": avg_daily_balance,
        
        # Application behavior features
        "time_spent_on_application": time_spent_on_application,
        "num_previous_applications": num_previous_applications,
        "application_completion_rate": application_completion_rate,
        "form_fill_time": form_fill_time,
        
        # Risk scores
        "device_risk_score": device_risk_score,
        "ip_risk_score": ip_risk_score,
        "location_risk_score": location_risk_score,
        "behavioral_risk_score": behavioral_risk_score,
        
        # Target
        "is_fraud": fraud
    })
    
    return data

def main():
    # Generate credit risk data
    credit_data = generate_realistic_credit_data()
    credit_train, credit_test = train_test_split(credit_data, test_size=0.2, random_state=42)
    credit_train.to_csv("data/credit_risk_train.csv", index=False)
    credit_test.to_csv("data/credit_risk_test.csv", index=False)
    
    # Generate fraud detection data
    fraud_data = generate_fraud_detection_data(200000)
    fraud_train, fraud_test = train_test_split(fraud_data, test_size=0.2, random_state=42)
    fraud_train.to_csv("data/fraud_detection_train.csv", index=False)
    fraud_test.to_csv("data/fraud_detection_test.csv", index=False)
    
    print("Generated realistic training data with 200,000 samples:")
    print(f"Credit Risk - Train: {len(credit_train)}, Test: {len(credit_test)}")
    print(f"Fraud Detection - Train: {len(fraud_train)}, Test: {len(fraud_test)}")

if __name__ == "__main__":
    main() 