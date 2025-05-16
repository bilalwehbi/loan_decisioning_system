import logging
from typing import Dict, List, Optional
import aiohttp
import asyncio
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class AlternativeDataService:
    def __init__(self):
        self.plaid_client_id = os.getenv('PLAID_CLIENT_ID')
        self.plaid_secret = os.getenv('PLAID_SECRET')
        self.plaid_env = os.getenv('PLAID_ENV', 'sandbox')
        self.credit_bureau_api_key = os.getenv('CREDIT_BUREAU_API_KEY')
        self.utility_api_key = os.getenv('UTILITY_API_KEY')
        self.rental_api_key = os.getenv('RENTAL_API_KEY')
        
        # Initialize API endpoints
        self.plaid_base_url = f"https://{self.plaid_env}.plaid.com"
        self.credit_bureau_url = os.getenv('CREDIT_BUREAU_URL')
        self.utility_api_url = os.getenv('UTILITY_API_URL')
        self.rental_api_url = os.getenv('RENTAL_API_URL')
    
    async def get_bank_data(self, user_id: str, access_token: str) -> Dict:
        """
        Get bank transaction data from Plaid.
        
        Args:
            user_id: Unique user identifier
            access_token: Plaid access token
            
        Returns:
            Dict containing bank transaction data and analysis
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Get transactions
                transactions = await self._get_plaid_transactions(session, access_token)
                
                # Get account balances
                balances = await self._get_plaid_balances(session, access_token)
                
                # Analyze transaction patterns
                analysis = self._analyze_transactions(transactions)
                
                return {
                    'transactions': transactions,
                    'balances': balances,
                    'analysis': analysis,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error fetching bank data: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def get_credit_data(self, user_id: str) -> Dict:
        """
        Get credit bureau data.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            Dict containing credit bureau data
        """
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {self.credit_bureau_api_key}',
                    'Content-Type': 'application/json'
                }
                
                async with session.get(
                    f"{self.credit_bureau_url}/credit-report/{user_id}",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._process_credit_data(data)
                    else:
                        raise Exception(f"Credit bureau API error: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error fetching credit data: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def get_utility_data(self, user_id: str) -> Dict:
        """
        Get utility payment history.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            Dict containing utility payment history
        """
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {self.utility_api_key}',
                    'Content-Type': 'application/json'
                }
                
                async with session.get(
                    f"{self.utility_api_url}/payments/{user_id}",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._process_utility_data(data)
                    else:
                        raise Exception(f"Utility API error: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error fetching utility data: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def get_rental_data(self, user_id: str) -> Dict:
        """
        Get rental payment history.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            Dict containing rental payment history
        """
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {self.rental_api_key}',
                    'Content-Type': 'application/json'
                }
                
                async with session.get(
                    f"{self.rental_api_url}/rental-history/{user_id}",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._process_rental_data(data)
                    else:
                        raise Exception(f"Rental API error: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error fetching rental data: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _get_plaid_transactions(self, session: aiohttp.ClientSession, access_token: str) -> List[Dict]:
        """Get transactions from Plaid API."""
        headers = {
            'Content-Type': 'application/json'
        }
        
        data = {
            'client_id': self.plaid_client_id,
            'secret': self.plaid_secret,
            'access_token': access_token,
            'start_date': (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
            'end_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        async with session.post(
            f"{self.plaid_base_url}/transactions/get",
            headers=headers,
            json=data
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result.get('transactions', [])
            else:
                raise Exception(f"Plaid API error: {response.status}")
    
    async def _get_plaid_balances(self, session: aiohttp.ClientSession, access_token: str) -> Dict:
        """Get account balances from Plaid API."""
        headers = {
            'Content-Type': 'application/json'
        }
        
        data = {
            'client_id': self.plaid_client_id,
            'secret': self.plaid_secret,
            'access_token': access_token
        }
        
        async with session.post(
            f"{self.plaid_base_url}/accounts/balance/get",
            headers=headers,
            json=data
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result.get('accounts', [])
            else:
                raise Exception(f"Plaid API error: {response.status}")
    
    def _analyze_transactions(self, transactions: List[Dict]) -> Dict:
        """Analyze transaction patterns."""
        if not transactions:
            return {
                'error': 'No transactions found',
                'timestamp': datetime.utcnow().isoformat()
            }
        
        # Calculate metrics
        total_income = sum(t['amount'] for t in transactions if t['amount'] > 0)
        total_expenses = sum(abs(t['amount']) for t in transactions if t['amount'] < 0)
        avg_balance = sum(t.get('balance', 0) for t in transactions) / len(transactions)
        
        # Analyze categories
        categories = {}
        for t in transactions:
            category = t.get('category', ['uncategorized'])[0]
            if category not in categories:
                categories[category] = 0
            categories[category] += abs(t['amount'])
        
        return {
            'total_income': total_income,
            'total_expenses': total_expenses,
            'avg_balance': avg_balance,
            'category_breakdown': categories,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _process_credit_data(self, data: Dict) -> Dict:
        """Process and analyze credit bureau data."""
        return {
            'credit_score': data.get('credit_score'),
            'payment_history': data.get('payment_history', []),
            'credit_utilization': data.get('credit_utilization'),
            'credit_mix': data.get('credit_mix', []),
            'public_records': data.get('public_records', []),
            'inquiries': data.get('inquiries', []),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _process_utility_data(self, data: Dict) -> Dict:
        """Process and analyze utility payment data."""
        return {
            'payment_history': data.get('payments', []),
            'payment_score': data.get('payment_score'),
            'late_payments': data.get('late_payments', 0),
            'average_payment_time': data.get('avg_payment_time'),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _process_rental_data(self, data: Dict) -> Dict:
        """Process and analyze rental payment data."""
        return {
            'rental_history': data.get('rental_history', []),
            'payment_score': data.get('payment_score'),
            'late_payments': data.get('late_payments', 0),
            'average_payment_time': data.get('avg_payment_time'),
            'timestamp': datetime.utcnow().isoformat()
        } 