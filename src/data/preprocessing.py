# src/data/preprocessing.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# Additional imports
import re
from urllib.parse import urlparse

class CustomFeatureGenerator(BaseEstimator, TransformerMixin):
    """Custom transformer to generate additional features"""
    
    def __init__(self):
        self.fitted = False
    
    def fit(self, X, y=None):
        self.fitted = True
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # DTI (Debt-to-Income) ratio
        if 'total_debt' in X_copy.columns and 'applicant_income' in X_copy.columns:
            X_copy['dti_ratio'] = X_copy['total_debt'] / (X_copy['applicant_income'] + 1e-10)
            # Cap at a reasonable maximum
            X_copy['dti_ratio'] = X_copy['dti_ratio'].clip(0, 5)
        
        # Payment-to-Income ratio
        if 'loan_amount' in X_copy.columns and 'loan_term_months' in X_copy.columns and 'applicant_income' in X_copy.columns:
            # Simple approximation of monthly payment
            monthly_payment = X_copy['loan_amount'] / X_copy['loan_term_months']
            X_copy['payment_to_income'] = monthly_payment / ((X_copy['applicant_income'] / 12) + 1e-10)
            X_copy['payment_to_income'] = X_copy['payment_to_income'].clip(0, 1)
        
        # Income verification ratio (stated income vs. observed deposits)
        if 'monthly_income' in X_copy.columns and 'avg_monthly_deposits' in X_copy.columns:
            X_copy['income_verification_ratio'] = X_copy['monthly_income'] / (X_copy['avg_monthly_deposits'] + 1e-10)
            X_copy['income_verification_ratio'] = X_copy['income_verification_ratio'].clip(0, 5)
        
        # Credit utilization categories
        if 'credit_utilization' in X_copy.columns:
            X_copy['high_utilization'] = (X_copy['credit_utilization'] > 0.7).astype(int)
        
        # Delinquency flags
        delinq_cols = [col for col in X_copy.columns if 'delinquencies' in col]
        if delinq_cols:
            X_copy['has_delinquencies'] = (X_copy[delinq_cols].sum(axis=1) > 0).astype(int)
        
        # Total number of negative banking items
        banking_neg_cols = ['num_nsf_transactions', 'num_overdrafts']
        if all(col in X_copy.columns for col in banking_neg_cols):
            X_copy['total_banking_negatives'] = X_copy[banking_neg_cols].sum(axis=1)
        
        # Loan-to-Income ratio
        if 'loan_amount' in X_copy.columns and 'applicant_income' in X_copy.columns:
            X_copy['loan_to_income'] = X_copy['loan_amount'] / (X_copy['applicant_income'] + 1e-10)
            X_copy['loan_to_income'] = X_copy['loan_to_income'].clip(0, 2)
        
        # Credit score categories (for credit models)
        if 'credit_score' in X_copy.columns:
            X_copy['credit_score_category'] = pd.cut(
                X_copy['credit_score'], 
                bins=[0, 580, 670, 740, 800, 850], 
                labels=[0, 1, 2, 3, 4]
            )
            X_copy['credit_score_category'] = X_copy['credit_score_category'].cat.add_categories([-1]).fillna(-1).astype(int)
        
        return X_copy

def create_credit_risk_pipeline() -> ColumnTransformer:
    """Create preprocessing pipeline for credit risk model"""
    
    # Define column types based on our generated data
    numeric_features = [
        'credit_score',
        'monthly_income',
        'utilization',
        'delinquencies',
        'inquiries_last_6m',
        'income_stability_score',
        'spending_pattern_score'
    ]
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ],
        remainder='drop'  # Drop columns not specified
    )
    
    # Create full pipeline with feature generation
    full_pipeline = Pipeline(steps=[
        ('feature_generator', CustomFeatureGenerator()),
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline

def create_fraud_detection_pipeline() -> ColumnTransformer:
    """Create preprocessing pipeline for fraud detection model"""
    # Define column types based on our generated data
    numeric_features = [
        'hour_of_day',
        'day_of_week',
        'device_id_length',
        'email_domain_length',
        'phone_number_length',
        'income_verification_ratio',
        'num_previous_applications',
        'time_spent_on_application'
    ]
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ],
        remainder='drop'  # Drop columns not specified
    )
    # Create full pipeline with feature generation
    full_pipeline = Pipeline(steps=[
        ('feature_generator', CustomFeatureGenerator()),
        ('preprocessor', preprocessor)
    ])
    return full_pipeline

def preprocess_loan_application(application_data: Dict) -> pd.DataFrame:
    """Process a single loan application into a DataFrame for model input"""
    
    # Extract data from the application
    credit_data = application_data.get('credit_data', {})
    banking_data = application_data.get('banking_data', {})
    app_data = application_data.get('application_data', {})
    
    # Combine all data into a single dictionary
    combined_data = {
        # Loan data
        'loan_amount': application_data.get('loan_amount'),
        'loan_term_months': application_data.get('loan_term_months'),
        'loan_purpose': application_data.get('loan_purpose'),
        'applicant_income': application_data.get('applicant_income'),
        'applicant_employment_length_years': application_data.get('applicant_employment_length_years'),
        
        # Credit data
        'credit_score': credit_data.get('credit_score'),
        'num_accounts': credit_data.get('num_accounts'),
        'num_active_accounts': credit_data.get('num_active_accounts'),
        'credit_utilization': credit_data.get('credit_utilization'),
        'num_delinquencies_30d': credit_data.get('num_delinquencies_30d'),
        'num_delinquencies_60d': credit_data.get('num_delinquencies_60d'),
        'num_delinquencies_90d': credit_data.get('num_delinquencies_90d'),
        'num_collections': credit_data.get('num_collections'),
        'total_debt': credit_data.get('total_debt'),
        'inquiries_last_6mo': credit_data.get('inquiries_last_6mo'),
        'longest_credit_length_months': credit_data.get('longest_credit_length_months'),
        
        # Banking data
        'avg_monthly_deposits': banking_data.get('avg_monthly_deposits'),
        'avg_monthly_withdrawals': banking_data.get('avg_monthly_withdrawals'),
        'monthly_income': banking_data.get('monthly_income'),
        'income_stability_score': banking_data.get('income_stability_score'),
        'num_income_sources': banking_data.get('num_income_sources'),
        'avg_daily_balance': banking_data.get('avg_daily_balance'),
        'num_nsf_transactions': banking_data.get('num_nsf_transactions'),
        'num_overdrafts': banking_data.get('num_overdrafts'),
        
        # Application data
        'ip_address': app_data.get('ip_address'),
        'device_id': app_data.get('device_id'),
        'email_domain': app_data.get('email_domain'),
        'application_timestamp': app_data.get('application_timestamp'),
        'time_spent_on_application': app_data.get('time_spent_on_application'),
        'num_previous_applications': app_data.get('num_previous_applications'),
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([combined_data])
    
    return df

class FraudFeatureGenerator(BaseEstimator, TransformerMixin):
    """Custom transformer to generate fraud detection features"""
    
    def __init__(self):
        self.fitted = False
    
    def fit(self, X, y=None):
        self.fitted = True
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Income verification ratio (stated income vs. observed deposits)
        if 'monthly_income' in X_copy.columns and 'avg_monthly_deposits' in X_copy.columns:
            X_copy['income_verification_ratio'] = X_copy['monthly_income'] / (X_copy['avg_monthly_deposits'] + 1e-10)
            X_copy['income_verification_ratio'] = X_copy['income_verification_ratio'].clip(0, 5)
            
            # Flag for significant mismatch
            X_copy['income_mismatch_flag'] = (X_copy['income_verification_ratio'] > 2.0).astype(int)
        
        # Application velocity
        if 'num_previous_applications' in X_copy.columns:
            X_copy['high_velocity_flag'] = (X_copy['num_previous_applications'] >= 3).astype(int)
        
        # Application speed concerns
        if 'time_spent_on_application' in X_copy.columns:
            X_copy['application_speed_concern'] = (X_copy['time_spent_on_application'] < 120).astype(int)
        
        # Email domain risk (free vs. paid domains)
        if 'email_domain' in X_copy.columns:
            # Define list of common free email domains
            free_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com']
            X_copy['free_email_domain'] = X_copy['email_domain'].apply(
                lambda x: 1 if x in free_domains else 0
            )
        
        # IP address risk assessment
        if 'ip_address' in X_copy.columns:
            # Check for private IP ranges
            def is_private_ip(ip):
                private_patterns = [
                    r'^10\.',
                    r'^172\.(1[6-9]|2[0-9]|3[0-1])\.',
                    r'^192\.168\.'
                ]
                return any(re.match(pattern, ip) for pattern in private_patterns)
                
            X_copy['private_ip_flag'] = X_copy['ip_address'].apply(
                lambda x: 1 if is_private_ip(str(x)) else 0
            )
        
        # Device ID analysis
        if 'device_id' in X_copy.columns:
            # Flag devices with "device-" prefix (potentially generated)
            X_copy['generic_device_flag'] = X_copy['device_id'].apply(
                lambda x: 1 if str(x).startswith('device-') else 0
            )
        
        # Banking activity consistency
        if all(col in X_copy.columns for col in ['avg_monthly_deposits', 'avg_monthly_withdrawals']):
            # Calculate ratio of withdrawals to deposits
            X_copy['withdrawal_deposit_ratio'] = X_copy['avg_monthly_withdrawals'] / (X_copy['avg_monthly_deposits'] + 1e-10)
            X_copy['withdrawal_deposit_ratio'] = X_copy['withdrawal_deposit_ratio'].clip(0, 2)
            
            # Flag for unusual balance
            X_copy['unusual_cash_flow'] = (X_copy['withdrawal_deposit_ratio'] > 1.2).astype(int)
        
        # NSF and overdraft indicators
        if all(col in X_copy.columns for col in ['num_nsf_transactions', 'num_overdrafts']):
            X_copy['banking_reliability_concern'] = ((X_copy['num_nsf_transactions'] + X_copy['num_overdrafts']) > 3).astype(int)
        
        # Loan to income analysis
        if 'loan_amount' in X_copy.columns and 'applicant_income' in X_copy.columns:
            # Calculate annual payment to income
            monthly_payment = X_copy['loan_amount'] / 12  # Simplified calculation
            annual_payment = monthly_payment * 12
            X_copy['payment_to_income_ratio'] = annual_payment / (X_copy['applicant_income'] + 1e-10)
            
            # Flag for potential affordability concerns
            X_copy['affordability_concern'] = (X_copy['payment_to_income_ratio'] > 0.5).astype(int)
        
        return X_copy    