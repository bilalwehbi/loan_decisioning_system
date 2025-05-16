import asyncio
from datetime import datetime, timedelta, timezone
from src.services.fraud_detector import FraudDetector

async def test_fraud_detection():
    # Initialize fraud detector
    detector = FraudDetector()
    
    # Test case 1: Normal application
    normal_app = {
        'application_id': 'APP001',
        'name': 'John Smith',
        'email': 'john.smith@gmail.com',
        'phone': '5551234567',
        'address': '123 Main St, City, State 12345',
        'ip_address': '8.8.8.8',
        'device_id': 'DEV001',
        'device_fingerprint': {
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124',
            'chrome_version': '91.0.4472.124',
            'accept_language': 'en-US,en;q=0.9',
            'accept_encoding': 'gzip, deflate, br'
        },
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    
    # Test case 2: Suspicious application
    suspicious_app = {
        'application_id': 'APP002',
        'name': 'Test User',
        'email': 'test@test.com',
        'phone': '1234567890',
        'address': 'Test Street 123',
        'ip_address': '192.168.1.1',
        'device_id': 'DEV002',
        'device_fingerprint': {
            'user_agent': 'Chrome/91.0.4472.124',
            'accept_language': 'en-US,en;q=0.9'
        },
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    
    # Test case 3: Historical data for velocity check
    historical_data = [
        {
            'application_id': 'APP003',
            'ip_address': '8.8.8.8',
            'device_id': 'DEV001',
            'email': 'john.smith@gmail.com',
            'timestamp': (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        },
        {
            'application_id': 'APP004',
            'ip_address': '8.8.8.8',
            'device_id': 'DEV001',
            'email': 'john.smith@gmail.com',
            'timestamp': (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat()
        }
    ]
    
    print("\nTesting Normal Application:")
    print("-" * 50)
    # Test synthetic ID detection
    synthetic_result = await detector.detect_synthetic_id(normal_app)
    print(f"Synthetic ID Detection: {synthetic_result}")
    
    # Test device risk
    device_result = await detector.analyze_device_risk(normal_app)
    print(f"Device Risk Analysis: {device_result}")
    
    # Test velocity
    velocity_result = await detector.check_velocity(normal_app, historical_data)
    print(f"Velocity Check: {velocity_result}")
    
    print("\nTesting Suspicious Application:")
    print("-" * 50)
    # Test synthetic ID detection
    synthetic_result = await detector.detect_synthetic_id(suspicious_app)
    print(f"Synthetic ID Detection: {synthetic_result}")
    
    # Test device risk
    device_result = await detector.analyze_device_risk(suspicious_app)
    print(f"Device Risk Analysis: {device_result}")
    
    # Test velocity
    velocity_result = await detector.check_velocity(suspicious_app, historical_data)
    print(f"Velocity Check: {velocity_result}")
    
    print("\nTesting Collusion Detection:")
    print("-" * 50)
    # Test collusion detection
    collusion_result = await detector.detect_collusion(historical_data)
    print(f"Collusion Detection: {collusion_result}")

if __name__ == "__main__":
    asyncio.run(test_fraud_detection()) 