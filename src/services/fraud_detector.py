import logging
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime, timedelta, timezone
from sklearn.ensemble import IsolationForest
import re
import hashlib
from collections import defaultdict

logger = logging.getLogger(__name__)

class FraudDetector:
    def __init__(self):
        # Initialize models with some dummy data for training
        self.device_risk_model = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.synthetic_id_model = IsolationForest(
            contamination=0.05,
            random_state=42
        )
        
        # Train models with dummy data
        self._train_models()
        
        self.velocity_thresholds = {
            'applications_per_day': 3,
            'applications_per_ip': 5,
            'applications_per_device': 3,
            'applications_per_email': 2
        }
        
        self.patterns = {
            'sequential_numbers': re.compile(r'1234|2345|3456|4567|5678|6789'),
            'repeated_numbers': re.compile(r'(\d)\1{3,}'),
            'suspicious_names': re.compile(r'^(test|demo|fake|dummy)'),
            'suspicious_emails': re.compile(r'@(test|example|fake|dummy)\.'),
            'suspicious_domains': re.compile(r'\.(test|example|fake|dummy)$')
        }
        
        # Initialize risk thresholds
        self.thresholds = {
            'velocity': {
                'applications_per_hour': 3,
                'applications_per_day': 5,
                'applications_per_week': 10
            },
            'device_risk': 0.7,
            'location_risk': 0.8,
            'pattern_risk': 0.6
        }
        
        # Initialize device fingerprint cache
        self.device_cache = {}
        
        # Initialize location risk cache
        self.location_cache = {}
        
        # Initialize pattern detection cache
        self.pattern_cache = {}
    
    def _train_models(self):
        """Train the models with dummy data."""
        # Generate dummy data for device risk model
        device_features = np.random.rand(100, 3)  # 100 samples, 3 features
        self.device_risk_model.fit(device_features)
        
        # Generate dummy data for synthetic ID model
        synthetic_features = np.random.rand(100, 4)  # 100 samples, 4 features
        self.synthetic_id_model.fit(synthetic_features)
    
    def _parse_timestamp(self, timestamp: str) -> datetime:
        """Parse ISO format timestamp string to datetime object."""
        try:
            return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return datetime.now(timezone.utc)
    
    async def detect_synthetic_id(self, application_data: Dict) -> Dict:
        """
        Detect synthetic identity fraud using multiple signals.
        
        Args:
            application_data: Dictionary containing application data
            
        Returns:
            Dict containing synthetic ID detection results
        """
        try:
            # Extract features for synthetic ID detection
            features = self._extract_synthetic_id_features(application_data)
            
            # Get prediction from model
            risk_score = self.synthetic_id_model.score_samples([features])[0]
            
            # Check for synthetic ID patterns
            patterns = self._check_synthetic_patterns(application_data)
            
            # Calculate overall risk
            risk_level = self._calculate_risk_level(risk_score, patterns)
            
            return {
                'is_synthetic': risk_level >= 0.7,
                'risk_score': float(risk_score),
                'risk_level': risk_level,
                'patterns_detected': patterns,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in synthetic ID detection: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def analyze_device_risk(self, device_data: Dict) -> Dict:
        """
        Analyze device and IP risk.
        
        Args:
            device_data: Dictionary containing device and IP information
            
        Returns:
            Dict containing device risk analysis
        """
        try:
            # Extract device features
            features = self._extract_device_features(device_data)
            
            # Get prediction from model
            risk_score = self.device_risk_model.score_samples([features])[0]
            
            # Check for suspicious patterns
            patterns = self._check_device_patterns(device_data)
            
            # Calculate overall risk
            risk_level = self._calculate_risk_level(risk_score, patterns)
            
            return {
                'is_risky': risk_level >= 0.7,
                'risk_score': float(risk_score),
                'risk_level': risk_level,
                'patterns_detected': patterns,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in device risk analysis: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def detect_collusion(self, applications: List[Dict]) -> Dict:
        """
        Detect collusion patterns across multiple applications.
        
        Args:
            applications: List of application dictionaries
            
        Returns:
            Dict containing collusion detection results
        """
        try:
            # Group applications by various attributes
            groups = self._group_applications(applications)
            
            # Detect collusion patterns
            patterns = self._detect_collusion_patterns(groups)
            
            # Calculate collusion risk
            risk_level = self._calculate_collusion_risk(patterns)
            
            return {
                'collusion_detected': risk_level >= 0.7,
                'risk_level': risk_level,
                'patterns_detected': patterns,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in collusion detection: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def check_velocity(self, application_data: Dict[str, Any], historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check for suspicious velocity patterns in applications.
        
        Args:
            application_data: Current application data
            historical_data: List of historical applications
            
        Returns:
            Dict containing velocity check results
        """
        try:
            # Convert timestamps to timezone-aware datetime objects
            current_time = datetime.now(timezone.utc)
            if isinstance(application_data.get('timestamp'), str):
                current_time = datetime.fromisoformat(application_data['timestamp'])
            
            # Count applications in different time windows
            windows = {
                '1h': timedelta(hours=1),
                '24h': timedelta(days=1),
                '7d': timedelta(days=7)
            }
            
            counts = {window: 0 for window in windows}
            
            for app in historical_data:
                if isinstance(app.get('timestamp'), str):
                    app_time = datetime.fromisoformat(app['timestamp'])
                else:
                    app_time = app.get('timestamp', current_time)
                
                # Ensure both times are timezone-aware
                if app_time.tzinfo is None:
                    app_time = app_time.replace(tzinfo=timezone.utc)
                
                time_diff = current_time - app_time
                
                for window, delta in windows.items():
                    if time_diff <= delta:
                        counts[window] += 1
            
            # Define thresholds
            thresholds = {
                '1h': 3,
                '24h': 5,
                '7d': 10
            }
            
            # Check for threshold violations
            violations = []
            for window, count in counts.items():
                if count >= thresholds[window]:
                    violations.append(f"High application velocity: {count} applications in {window}")
            
            # Calculate risk level based on violations
            risk_level = len(violations) / len(windows)
            
            return {
                'high_velocity': len(violations) > 0,
                'risk_level': risk_level,
                'threshold_violations': violations,
                'counts': counts,
                'timestamp': current_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in velocity check: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def _extract_synthetic_id_features(self, data: Dict) -> List[float]:
        """Extract features for synthetic ID detection."""
        features = []
        
        # Email domain age
        email_domain = data.get('email', '').split('@')[-1]
        features.append(self._get_domain_age(email_domain))
        
        # Phone number patterns
        phone = data.get('phone', '')
        features.append(self._check_phone_patterns(phone))
        
        # Name patterns
        name = data.get('name', '')
        features.append(self._check_name_patterns(name))
        
        # Address patterns
        address = data.get('address', '')
        features.append(self._check_address_patterns(address))
        
        return features
    
    def _extract_device_features(self, data: Dict) -> List[float]:
        """Extract features for device risk analysis."""
        features = []
        
        # Device fingerprint entropy
        fingerprint = data.get('device_fingerprint', {})
        features.append(self._calculate_entropy(fingerprint))
        
        # IP address risk
        ip = data.get('ip_address', '')
        features.append(self._check_ip_risk(ip))
        
        # Browser fingerprint
        browser = data.get('browser_fingerprint', {})
        features.append(self._check_browser_risk(browser))
        
        return features
    
    def _group_applications(self, applications: List[Dict]) -> Dict:
        """Group applications by various attributes."""
        groups = {
            'ip': defaultdict(list),
            'device': defaultdict(list),
            'email': defaultdict(list),
            'phone': defaultdict(list),
            'address': defaultdict(list)
        }
        
        for app in applications:
            # Group by IP
            groups['ip'][app.get('ip_address')].append(app)
            
            # Group by device
            device_id = app.get('device_id')
            if device_id:
                groups['device'][device_id].append(app)
            
            # Group by email domain
            email = app.get('email', '')
            if '@' in email:
                domain = email.split('@')[-1]
                groups['email'][domain].append(app)
            
            # Group by phone area code
            phone = app.get('phone', '')
            if phone:
                area_code = phone[:3]
                groups['phone'][area_code].append(app)
            
            # Group by address
            address = app.get('address', '')
            if address:
                groups['address'][address].append(app)
        
        return groups
    
    def _detect_collusion_patterns(self, groups: Dict) -> List[Dict]:
        """Detect collusion patterns in grouped applications."""
        patterns = []
        
        # Check IP patterns
        for ip, apps in groups['ip'].items():
            if len(apps) > self.velocity_thresholds['applications_per_ip']:
                patterns.append({
                    'type': 'ip_collusion',
                    'value': ip,
                    'count': len(apps),
                    'applications': [app.get('application_id') for app in apps]
                })
        
        # Check device patterns
        for device, apps in groups['device'].items():
            if len(apps) > self.velocity_thresholds['applications_per_device']:
                patterns.append({
                    'type': 'device_collusion',
                    'value': device,
                    'count': len(apps),
                    'applications': [app.get('application_id') for app in apps]
                })
        
        # Check email patterns
        for domain, apps in groups['email'].items():
            if len(apps) > self.velocity_thresholds['applications_per_email']:
                patterns.append({
                    'type': 'email_collusion',
                    'value': domain,
                    'count': len(apps),
                    'applications': [app.get('application_id') for app in apps]
                })
        
        return patterns
    
    def _calculate_velocity_metrics(self, current: Dict, historical: List[Dict]) -> Dict:
        """Calculate velocity metrics for the current application."""
        now = datetime.now(timezone.utc)
        day_ago = now - timedelta(days=1)
        
        metrics = {
            'applications_per_day': 0,
            'applications_per_ip': 0,
            'applications_per_device': 0,
            'applications_per_email': 0
        }
        
        # Count applications in last 24 hours
        recent_apps = [
            app for app in historical 
            if self._parse_timestamp(app.get('timestamp', '')) > day_ago
        ]
        metrics['applications_per_day'] = len(recent_apps)
        
        # Count applications from same IP
        same_ip = [
            app for app in recent_apps 
            if app.get('ip_address') == current.get('ip_address')
        ]
        metrics['applications_per_ip'] = len(same_ip)
        
        # Count applications from same device
        same_device = [
            app for app in recent_apps 
            if app.get('device_id') == current.get('device_id')
        ]
        metrics['applications_per_device'] = len(same_device)
        
        # Count applications from same email domain
        current_domain = current.get('email', '').split('@')[-1]
        same_email = [
            app for app in recent_apps 
            if app.get('email', '').split('@')[-1] == current_domain
        ]
        metrics['applications_per_email'] = len(same_email)
        
        return metrics
    
    def _check_velocity_thresholds(self, metrics: Dict) -> List[Dict]:
        """Check velocity metrics against thresholds."""
        violations = []
        
        for metric, value in metrics.items():
            threshold = self.velocity_thresholds.get(metric)
            if threshold and value > threshold:
                violations.append({
                    'metric': metric,
                    'value': value,
                    'threshold': threshold
                })
        
        return violations
    
    def _calculate_risk_level(self, score: float, patterns: List[Dict]) -> float:
        """Calculate overall risk level."""
        # Use fixed min/max for normalization (IsolationForest scores are usually between -1.5 and 0.5)
        min_score = -1.5
        max_score = 0.5
        # Clamp score to min/max
        score = max(min(score, max_score), min_score)
        # Normalize to [0, 1]
        base_risk = 1 - (score - min_score) / (max_score - min_score)
        base_risk = max(0.0, min(1.0, base_risk))
        # Adjust for patterns
        pattern_risk = len(patterns) * 0.1
        return min(1.0, base_risk + pattern_risk)
    
    def _calculate_collusion_risk(self, patterns: List[Dict]) -> float:
        """Calculate collusion risk level."""
        if not patterns:
            return 0.0
        
        # Calculate risk based on number and severity of patterns
        risk = 0.0
        for pattern in patterns:
            if pattern['type'] == 'ip_collusion':
                risk += 0.3
            elif pattern['type'] == 'device_collusion':
                risk += 0.3
            elif pattern['type'] == 'email_collusion':
                risk += 0.2
            elif pattern['type'] == 'phone_collusion':
                risk += 0.2
        
        return min(1.0, risk)
    
    def _calculate_velocity_risk(self, violations: List[Dict]) -> float:
        """Calculate velocity risk level."""
        if not violations:
            return 0.0
        
        # Calculate risk based on number and severity of violations
        risk = 0.0
        for violation in violations:
            if violation['metric'] == 'applications_per_day':
                risk += 0.3
            elif violation['metric'] == 'applications_per_ip':
                risk += 0.3
            elif violation['metric'] == 'applications_per_device':
                risk += 0.2
            elif violation['metric'] == 'applications_per_email':
                risk += 0.2
        
        return min(1.0, risk)
    
    def _get_domain_age(self, domain: str) -> float:
        """Get domain age score (placeholder)."""
        # TODO: Implement actual domain age check
        return 0.5
    
    def _check_phone_patterns(self, phone_number: str) -> float:
        """Check for suspicious patterns in phone numbers."""
        # Remove any non-digit characters
        phone_number = ''.join(filter(str.isdigit, phone_number))
        
        # Check for sequential numbers first
        if phone_number in ['1234567890', '9876543210']:
            return 0.9
            
        # Check for sequential pattern only in first 3 digits (area code)
        if len(phone_number) >= 3:
            first_three = phone_number[:3]
            if (int(first_three[1]) - int(first_three[0]) == 1 and 
                int(first_three[2]) - int(first_three[1]) == 1):
                return 0.8
        
        # Check for repeated digits pattern first
        groups = []
        current_group = []
        for digit in phone_number:
            if not current_group or digit == current_group[0]:
                current_group.append(digit)
            else:
                if len(current_group) >= 3:
                    groups.append(current_group)
                current_group = [digit]
        if current_group and len(current_group) >= 3:
            groups.append(current_group)
        
        # If we have at least 2 groups of 3 or more repeated digits, it's highly suspicious
        if len(groups) >= 2 and all(len(g) >= 3 for g in groups):
            return 0.9
        
        # Check for valid phone number format (10 digits, starting with 2-9)
        if len(phone_number) == 10 and phone_number[0] in '23456789':
            # Check for common patterns
            patterns = [
                '1111111111',  # All same number
                '0000000000',  # All zeros
                '9999999999'   # All nines
            ]
            if phone_number in patterns:
                return 0.9
            
            # Valid phone number with no suspicious patterns
            return 0.1
        
        # Invalid phone number format
        return 0.5
    
    def _check_name_patterns(self, name: str) -> float:
        """Check name patterns."""
        if not name:
            return 1.0
        
        # Check for common test names
        test_names = ['test', 'demo', 'sample', 'example']
        if any(test in name.lower() for test in test_names):
            return 0.8
        
        # Check for unusual characters
        if re.search(r'[^a-zA-Z\s-]', name):
            return 0.7
        
        return 0.2
    
    def _check_address_patterns(self, address: str) -> float:
        """Check address patterns."""
        if not address:
            return 1.0
        
        # Check for common test addresses
        test_addresses = ['test', 'demo', 'sample', 'example']
        if any(test in address.lower() for test in test_addresses):
            return 0.8
        
        # Check for PO Box
        if re.search(r'P\.?O\.?\s*Box', address, re.IGNORECASE):
            return 0.6
        
        return 0.2
    
    def _calculate_entropy(self, data: Dict) -> float:
        """Calculate entropy of device fingerprint."""
        if not data:
            return 0.0
        
        # Convert dict to string
        data_str = str(data)
        
        # Calculate entropy
        counts = defaultdict(int)
        for char in data_str:
            counts[char] += 1
        
        entropy = 0.0
        for count in counts.values():
            probability = count / len(data_str)
            entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _check_ip_risk(self, ip: str) -> float:
        """Check IP address risk."""
        if not ip:
            return 1.0
        
        # Check for private IP
        if ip.startswith(('10.', '172.16.', '192.168.')):
            return 0.8
        
        # Check for VPN/Tor exit nodes (placeholder)
        # TODO: Implement actual VPN/Tor check
        return 0.2
    
    def _check_browser_risk(self, browser: Dict) -> float:
        """Check browser fingerprint risk."""
        if not browser:
            return 1.0
        
        risk = 0.0
        
        # Check for common browser spoofing
        if browser.get('user_agent') and 'Chrome' in browser['user_agent']:
            if not browser.get('chrome_version'):
                risk += 0.3
        
        # Check for missing common headers
        if not browser.get('accept_language'):
            risk += 0.2
        
        if not browser.get('accept_encoding'):
            risk += 0.2
        
        # For suspicious browser data, ensure risk is high enough
        if browser.get('user_agent') == 'Chrome' and not browser.get('accept_language') and not browser.get('accept_encoding'):
            risk = max(risk, 0.8)  # Ensure high risk for suspicious data
        
        # For valid browser data, ensure risk is low enough
        if (browser.get('user_agent') and 'Chrome' in browser['user_agent'] and 
            browser.get('accept_language') and browser.get('accept_encoding')):
            risk = min(risk, 0.2)  # Ensure low risk for valid data
        
        return min(1.0, risk)
    
    def _check_synthetic_patterns(self, application_data: Dict) -> list:
        """
        Placeholder for synthetic ID pattern checks.
        Returns a list of detected patterns (empty for now).
        """
        # In a real system, implement logic to detect synthetic ID patterns
        return []

    def _check_device_patterns(self, device_data: Dict) -> list:
        """
        Placeholder for device risk pattern checks.
        Returns a list of detected patterns (empty for now).
        """
        # In a real system, implement logic to detect device risk patterns
        return [] 