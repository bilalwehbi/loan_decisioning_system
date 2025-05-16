import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any

class LoanDataGenerator:
    def __init__(self, seed: int = 42):
        """Initialize the data generator with a random seed."""
        np.random.seed(seed)
        
    def generate_credit_data(self) -> Dict[str, Any]:
        """Generate realistic credit data."""
        # Credit score follows a normal distribution with mean 700 and std 100
        credit_score = int(np.clip(np.random.normal(700, 100), 300, 850))
        
        # Delinquencies follow a Poisson distribution
        delinquencies = np.random.poisson(0.5)
        
        # Inquiries follow a Poisson distribution
        inquiries_last_6m = np.random.poisson(2)
        
        # Tradelines follow a normal distribution
        tradelines = int(np.clip(np.random.normal(8, 3), 0, 20))
        
        # Utilization follows a beta distribution
        utilization = np.random.beta(2, 5)
        
        # Payment history score based on credit score and delinquencies
        base_score = credit_score / 850
        delinquency_penalty = delinquencies * 0.1
        payment_history_score = max(0, min(1, base_score - delinquency_penalty))
        
        # Credit age follows a normal distribution
        credit_age_months = int(np.clip(np.random.normal(84, 24), 0, 300))
        
        # Credit mix score based on tradelines and credit age
        credit_mix_score = min(1, (tradelines / 10) * (credit_age_months / 120))
        
        return {
            "credit_score": credit_score,
            "delinquencies": delinquencies,
            "inquiries_last_6m": inquiries_last_6m,
            "tradelines": tradelines,
            "utilization": float(utilization),
            "payment_history_score": float(payment_history_score),
            "credit_age_months": credit_age_months,
            "credit_mix_score": float(credit_mix_score)
        }
    
    def generate_banking_data(self) -> Dict[str, Any]:
        """Generate realistic banking data."""
        # Income follows a log-normal distribution
        avg_monthly_income = np.random.lognormal(8, 0.5)
        
        # Income stability based on income and random factors
        income_stability_score = np.random.beta(5, 2)
        
        # Spending pattern score based on income and random factors
        spending_pattern_score = np.random.beta(4, 3)
        
        # Transaction count follows a Poisson distribution
        transaction_count = np.random.poisson(45)
        
        # Account balance based on income
        avg_account_balance = avg_monthly_income * np.random.uniform(1, 3)
        
        # Overdraft frequency follows a Poisson distribution
        overdraft_frequency = np.random.poisson(0.3)
        
        # Savings rate follows a beta distribution
        savings_rate = np.random.beta(3, 7)
        
        # Recurring payments based on income
        rent = avg_monthly_income * 0.3
        utilities = avg_monthly_income * 0.05
        recurring_payments = [
            {"rent": float(rent)},
            {"utilities": float(utilities)}
        ]
        
        return {
            "avg_monthly_income": float(avg_monthly_income),
            "income_stability_score": float(income_stability_score),
            "spending_pattern_score": float(spending_pattern_score),
            "transaction_count": transaction_count,
            "avg_account_balance": float(avg_account_balance),
            "overdraft_frequency": overdraft_frequency,
            "savings_rate": float(savings_rate),
            "recurring_payments": recurring_payments
        }
    
    def generate_behavioral_data(self) -> Dict[str, Any]:
        """Generate realistic behavioral data."""
        # Application completion time follows a normal distribution
        application_completion_time = np.random.normal(600, 120)
        
        # Form fill pattern based on completion time
        section1_time = application_completion_time * 0.3
        section2_time = application_completion_time * 0.7
        form_fill_pattern = {
            "section1": float(section1_time),
            "section2": float(section2_time)
        }
        
        # Device trust score follows a beta distribution
        device_trust_score = np.random.beta(8, 2)
        
        # Location risk score follows a beta distribution
        location_risk_score = np.random.beta(2, 8)
        
        # Digital footprint score based on device trust and random factors
        digital_footprint_score = np.random.beta(7, 3)
        
        # Social media presence scores
        social_media_presence = {
            "linkedin": float(np.random.beta(6, 4)),
            "facebook": float(np.random.beta(5, 5))
        }
        
        return {
            "application_completion_time": float(application_completion_time),
            "form_fill_pattern": form_fill_pattern,
            "device_trust_score": float(device_trust_score),
            "location_risk_score": float(location_risk_score),
            "digital_footprint_score": float(digital_footprint_score),
            "social_media_presence": social_media_presence
        }
    
    def generate_employment_data(self) -> Dict[str, Any]:
        """Generate realistic employment data."""
        # Employment length follows a normal distribution
        employment_length_months = int(np.clip(np.random.normal(36, 24), 0, 360))
        
        # Employment verification score based on length and random factors
        employment_verification_score = np.random.beta(8, 2)
        
        # Income verification score based on employment verification
        income_verification_score = employment_verification_score * np.random.uniform(0.9, 1.0)
        
        # Paystub analysis scores
        paystub_analysis = {
            "income_match": float(np.random.beta(9, 1)),
            "regularity": float(np.random.beta(8, 2))
        }
        
        # Employment history
        employment_history = []
        if employment_length_months > 24:
            prev_employer = {
                "employer": f"Previous Corp {np.random.randint(1, 100)}",
                "duration_months": int(np.random.uniform(12, 24)),
                "position": np.random.choice(["Software Engineer", "Data Analyst", "Product Manager"])
            }
            employment_history.append(prev_employer)
        
        return {
            "employer_name": f"Tech Corp {np.random.randint(1, 100)}",
            "employment_length_months": employment_length_months,
            "employment_verification_score": float(employment_verification_score),
            "income_verification_score": float(income_verification_score),
            "paystub_analysis": paystub_analysis,
            "employment_history": employment_history
        }
    
    def generate_application_data(self) -> Dict[str, Any]:
        """Generate realistic application metadata."""
        # Generate a random device ID
        device_id = f"device_{np.random.randint(1000, 9999)}"
        
        # Generate a random IP address
        ip_address = f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
        
        # Generate a random email domain
        email_domain = np.random.choice(["gmail.com", "yahoo.com", "hotmail.com", "company.com"])
        
        # Generate a random phone number
        phone_number = f"1{np.random.randint(200, 999)}{np.random.randint(200, 999)}{np.random.randint(1000, 9999)}"
        
        # Generate application timestamp
        application_timestamp = datetime.utcnow() - timedelta(days=np.random.randint(0, 30))
        
        # Generate device fingerprint
        device_fingerprint = {
            "os": np.random.choice(["Windows", "MacOS", "Linux", "iOS", "Android"]),
            "browser": np.random.choice(["Chrome", "Firefox", "Safari", "Edge"])
        }
        
        # Generate browser fingerprint
        browser_fingerprint = {
            "user_agent": f"Mozilla/5.0 ({device_fingerprint['os']}) {device_fingerprint['browser']}"
        }
        
        # Generate network info
        network_info = {
            "proxy_score": float(np.random.beta(1, 9)),  # Most users don't use proxies
            "vpn_score": float(np.random.beta(1, 9))     # Most users don't use VPNs
        }
        
        # Generate location data
        location_data = {
            "country": np.random.choice(["US", "CA", "UK", "AU", "DE"]),
            "city": np.random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"])
        }
        
        return {
            "device_id": device_id,
            "ip_address": ip_address,
            "email_domain": email_domain,
            "phone_number": phone_number,
            "application_timestamp": application_timestamp.isoformat(),
            "device_fingerprint": device_fingerprint,
            "browser_fingerprint": browser_fingerprint,
            "network_info": network_info,
            "location_data": location_data
        }
    
    def generate_loan_application(self) -> Dict[str, Any]:
        """Generate a complete loan application with realistic data."""
        # Generate loan amount (follows a log-normal distribution)
        loan_amount = np.random.lognormal(9, 0.5)
        
        # Generate loan term (typically 12, 24, 36, 48, 60, or 72 months)
        loan_term = np.random.choice([12, 24, 36, 48, 60, 72])
        
        # Generate loan purpose
        loan_purpose = np.random.choice([
            "Home improvement",
            "Debt consolidation",
            "Business",
            "Education",
            "Medical expenses",
            "Major purchase"
        ])
        
        return {
            "loan_amount": float(loan_amount),
            "loan_term": loan_term,
            "loan_purpose": loan_purpose,
            "credit_data": self.generate_credit_data(),
            "banking_data": self.generate_banking_data(),
            "behavioral_data": self.generate_behavioral_data(),
            "employment_data": self.generate_employment_data(),
            "application_data": self.generate_application_data()
        }
    
    def generate_training_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate a dataset for model training."""
        applications = []
        for _ in range(n_samples):
            app = self.generate_loan_application()
            
            # Extract features for training
            features = {
                "credit_score": app["credit_data"]["credit_score"],
                "delinquencies": app["credit_data"]["delinquencies"],
                "inquiries_last_6m": app["credit_data"]["inquiries_last_6m"],
                "tradelines": app["credit_data"]["tradelines"],
                "utilization": app["credit_data"]["utilization"],
                "payment_history_score": app["credit_data"]["payment_history_score"],
                "credit_age_months": app["credit_data"]["credit_age_months"],
                "credit_mix_score": app["credit_data"]["credit_mix_score"],
                "avg_monthly_income": app["banking_data"]["avg_monthly_income"],
                "income_stability_score": app["banking_data"]["income_stability_score"],
                "spending_pattern_score": app["banking_data"]["spending_pattern_score"],
                "transaction_count": app["banking_data"]["transaction_count"],
                "avg_account_balance": app["banking_data"]["avg_account_balance"],
                "overdraft_frequency": app["banking_data"]["overdraft_frequency"],
                "savings_rate": app["banking_data"]["savings_rate"],
                "application_completion_time": app["behavioral_data"]["application_completion_time"],
                "device_trust_score": app["behavioral_data"]["device_trust_score"],
                "location_risk_score": app["behavioral_data"]["location_risk_score"],
                "digital_footprint_score": app["behavioral_data"]["digital_footprint_score"],
                "employment_length_months": app["employment_data"]["employment_length_months"],
                "employment_verification_score": app["employment_data"]["employment_verification_score"],
                "income_verification_score": app["employment_data"]["income_verification_score"],
                "loan_amount": app["loan_amount"],
                "loan_term": app["loan_term"]
            }
            
            # Generate target variables (default and fraud)
            # Default probability based on credit score, income, and loan amount
            default_prob = 1 / (1 + np.exp(
                -(-2 + 
                  0.005 * features["credit_score"] + 
                  0.0001 * features["avg_monthly_income"] - 
                  0.0001 * features["loan_amount"])
            ))
            is_default = np.random.binomial(1, default_prob)
            
            # Fraud probability based on behavioral and application data
            fraud_prob = 1 / (1 + np.exp(
                -(-3 + 
                  2 * (1 - features["device_trust_score"]) + 
                  2 * features["location_risk_score"] + 
                  2 * (1 - features["digital_footprint_score"]))
            ))
            is_fraud = np.random.binomial(1, fraud_prob)
            
            features["is_default"] = is_default
            features["is_fraud"] = is_fraud
            
            applications.append(features)
        
        return pd.DataFrame(applications) 