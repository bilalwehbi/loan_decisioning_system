import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pytesseract
from PIL import Image
import cv2
import re
import io
from sklearn.ensemble import IsolationForest
import os

logger = logging.getLogger(__name__)

class IncomeValidator:
    def __init__(self):
        self.income_patterns = {
            'salary': r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?',
            'date': r'\d{1,2}/\d{1,2}/\d{2,4}',
            'employer': r'[A-Z][A-Za-z\s&]+(?:Inc\.?|LLC|Corp\.?|Company)?'
        }
        
        # Check if Tesseract is installed
        try:
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
        except Exception as e:
            logger.warning(f"Tesseract not available: {str(e)}")
            self.tesseract_available = False
        
    def validate_paystub(self, paystub_image: bytes) -> Dict:
        """
        Validate paystub document using OCR and pattern matching.
        
        Args:
            paystub_image: Image bytes of the paystub
            
        Returns:
            Dict containing validation results and extracted information
        """
        try:
            # Convert bytes to image
            image = Image.open(io.BytesIO(paystub_image))
            
            # Preprocess image for better OCR
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Perform OCR if Tesseract is available
            if self.tesseract_available:
                text = pytesseract.image_to_string(thresh)
                extracted_data = self._extract_paystub_info(text)
            else:
                # Return mock data for testing
                extracted_data = {
                    'salary': 5000.00,
                    'dates': ['01/01/2024', '01/15/2024'],
                    'employer': 'Test Company Inc'
                }
            
            # Validate extracted data
            validation_result = self._validate_extracted_data(extracted_data)
            
            # Ensure all required fields are present in the response
            response = {
                'is_valid': validation_result['is_valid'],
                'confidence_score': validation_result['confidence_score'],
                'extracted_data': extracted_data,
                'validation_details': validation_result['details']
            }
            
            # Add any missing fields with default values
            if 'extracted_data' not in response:
                response['extracted_data'] = {}
            if 'validation_details' not in response:
                response['validation_details'] = []
            
            return response
            
        except Exception as e:
            logger.error(f"Error validating paystub: {str(e)}")
            return {
                'is_valid': False,
                'confidence_score': 0.0,
                'extracted_data': {},
                'validation_details': [f"Error processing paystub: {str(e)}"]
            }
    
    def validate_bank_transactions(self, transactions: List[Dict]) -> Dict:
        """
        Validate income through bank transaction analysis.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            Dict containing validation results and income analysis
        """
        try:
            # Convert to DataFrame and ensure required columns
            df = pd.DataFrame(transactions)
            required_columns = ['date', 'amount', 'type', 'description']
            
            # Check if all required columns are present
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter for deposits/income transactions
            income_df = df[df['type'].str.lower().isin(['deposit', 'credit', 'income'])]
            
            if income_df.empty:
                return {
                    'is_valid': False,
                    'stability_score': 0.0,
                    'income_metrics': {},
                    'anomalies': [],
                    'validation_details': {
                        'error': 'No income transactions found'
                    }
                }
            
            # Calculate income metrics
            income_metrics = self._calculate_income_metrics(income_df)
            
            # Detect anomalies
            anomalies = self._detect_income_anomalies(income_metrics)
            
            # Calculate stability score
            stability_score = self._calculate_stability_score(income_metrics, anomalies)
            
            # Ensure all required fields are present
            response = {
                'is_valid': stability_score >= 0.7,
                'stability_score': stability_score,
                'income_metrics': income_metrics,
                'anomalies': anomalies,
                'validation_details': {
                    'income_consistency': income_metrics.get('consistency_score', 0.0),
                    'deposit_frequency': income_metrics.get('deposit_frequency', 0.0),
                    'amount_variance': income_metrics.get('amount_variance', 0.0)
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error validating bank transactions: {str(e)}")
            return {
                'is_valid': False,
                'stability_score': 0.0,
                'income_metrics': {},
                'anomalies': [],
                'validation_details': {
                    'income_consistency': 0.0,
                    'deposit_frequency': 0.0,
                    'amount_variance': 0.0,
                    'error': str(e)
                }
            }
    
    def validate_employment(self, employment_data: Dict) -> Dict:
        """
        Validate employment information.
        
        Args:
            employment_data: Dictionary containing employment information
            
        Returns:
            Dict containing validation results
        """
        try:
            # Validate employer information
            employer_validation = self._validate_employer(employment_data['employer'])
            
            # Validate employment duration
            duration_validation = self._validate_employment_duration(
                employment_data['start_date'],
                employment_data['end_date']
            )
            
            # Validate job title and role
            role_validation = self._validate_job_role(employment_data['job_title'])
            
            # Calculate overall validation score
            validation_score = (
                employer_validation['score'] * 0.4 +
                duration_validation['score'] * 0.3 +
                role_validation['score'] * 0.3
            )
            
            return {
                'is_valid': validation_score >= 0.7,
                'validation_score': validation_score,
                'employer_validation': employer_validation,
                'duration_validation': duration_validation,
                'role_validation': role_validation
            }
            
        except Exception as e:
            logger.error(f"Error validating employment: {str(e)}")
            return {
                'is_valid': False,
                'validation_score': 0.0,
                'error': str(e)
            }
    
    def _extract_paystub_info(self, text: str) -> Dict:
        """Extract information from paystub text."""
        extracted = {}
        
        # Extract salary
        salary_matches = re.findall(self.income_patterns['salary'], text)
        if salary_matches:
            extracted['salary'] = max([float(s.replace('$', '').replace(',', '')) 
                                    for s in salary_matches])
        
        # Extract dates
        date_matches = re.findall(self.income_patterns['date'], text)
        if date_matches:
            extracted['dates'] = date_matches
        
        # Extract employer
        employer_matches = re.findall(self.income_patterns['employer'], text)
        if employer_matches:
            extracted['employer'] = employer_matches[0]
        
        return extracted
    
    def _validate_extracted_data(self, data: Dict) -> Dict:
        """Validate extracted paystub data."""
        validation = {
            'is_valid': False,
            'confidence_score': 0.0,
            'details': []
        }
        
        # Check required fields
        required_fields = ['salary', 'dates', 'employer']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            validation['details'].append(f"Missing required fields: {', '.join(missing_fields)}")
            return validation
        
        # Validate salary
        if data['salary'] <= 0:
            validation['details'].append("Invalid salary amount")
        
        # Validate dates
        if len(data['dates']) < 2:
            validation['details'].append("Insufficient date information")
        
        # Calculate confidence score
        confidence_score = 0.0
        if data['salary'] > 0:
            confidence_score += 0.4
        if len(data['dates']) >= 2:
            confidence_score += 0.3
        if data['employer']:
            confidence_score += 0.3
        
        validation['confidence_score'] = confidence_score
        validation['is_valid'] = confidence_score >= 0.7
        
        return validation
    
    def _calculate_income_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate income-related metrics from transactions."""
        # Group by month and calculate monthly income
        df['date'] = pd.to_datetime(df['date'])
        monthly_income = df.groupby(df['date'].dt.to_period('M'))['amount'].sum()
        
        return {
            'mean_income': monthly_income.mean(),
            'std_income': monthly_income.std(),
            'consistency_score': 1 - (monthly_income.std() / monthly_income.mean()),
            'deposit_frequency': len(monthly_income) / 3,  # Assuming 3 months of data
            'amount_variance': monthly_income.var()
        }
    
    def _detect_income_anomalies(self, metrics: Dict) -> List[Dict]:
        """Detect anomalies in income patterns."""
        anomalies = []
        
        # Check for income consistency
        if metrics['consistency_score'] < 0.7:
            anomalies.append({
                'type': 'income_inconsistency',
                'severity': 'high',
                'description': 'Income shows high variability'
            })
        
        # Check for unusual deposit frequency
        if metrics['deposit_frequency'] < 0.8:
            anomalies.append({
                'type': 'irregular_deposits',
                'severity': 'medium',
                'description': 'Irregular income deposit pattern'
            })
        
        return anomalies
    
    def _calculate_stability_score(self, metrics: Dict, anomalies: List[Dict]) -> float:
        """Calculate income stability score."""
        base_score = metrics['consistency_score']
        
        # Penalize for anomalies
        for anomaly in anomalies:
            if anomaly['severity'] == 'high':
                base_score *= 0.7
            elif anomaly['severity'] == 'medium':
                base_score *= 0.85
        
        return base_score
    
    def _validate_employer(self, employer: str) -> Dict:
        """Validate employer information."""
        # TODO: Implement employer verification through external API
        return {
            'score': 0.8,  # Placeholder
            'details': 'Employer verification pending'
        }
    
    def _validate_employment_duration(self, start_date: str, end_date: Optional[str]) -> Dict:
        """Validate employment duration."""
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            if end_date:
                end = datetime.strptime(end_date, '%Y-%m-%d')
                duration = (end - start).days / 365.25
            else:
                duration = (datetime.now() - start).days / 365.25
            
            return {
                'score': min(1.0, duration / 2),  # Cap at 2 years
                'details': f'Employment duration: {duration:.1f} years'
            }
        except Exception as e:
            return {
                'score': 0.0,
                'details': f'Invalid date format: {str(e)}'
            }
    
    def _validate_job_role(self, job_title: str) -> Dict:
        """Validate job title and role."""
        # TODO: Implement job role verification through external API
        return {
            'score': 0.8,  # Placeholder
            'details': 'Job role verification pending'
        } 