import pytest
from datetime import datetime, timezone
from src.services.fraud_detector import FraudDetector

@pytest.fixture
def fraud_detector():
    return FraudDetector()

@pytest.fixture
def sample_application_data():
    return {
        'application_id': 'APP123',
        'name': 'John Doe',
        'email': 'john.doe@example.com',
        'phone': '5551234567',
        'address': '123 Main St, City, State 12345',
        'ip_address': '192.168.1.1',
        'device_id': 'DEV123',
        'device_fingerprint': {
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124',
            'accept_language': 'en-US,en;q=0.9',
            'accept_encoding': 'gzip, deflate, br',
            'screen_resolution': '1920x1080',
            'timezone': 'America/New_York',
            'plugins': ['Chrome PDF Plugin', 'Chrome PDF Viewer'],
            'canvas_fingerprint': 'abc123',
            'webgl_fingerprint': 'def456',
            'audio_fingerprint': 'ghi789',
            'fonts': ['Arial', 'Times New Roman', 'Courier New'],
            'touch_support': True,
            'platform': 'Win32',
            'do_not_track': False,
            'cookie_enabled': True,
            'language': 'en-US',
            'color_depth': 24,
            'pixel_ratio': 1.5,
            'hardware_concurrency': 8,
            'device_memory': 8,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    }

@pytest.fixture
def sample_historical_data():
    return [
        {
            'application_id': 'APP001',
            'device_id': 'DEV123',
            'email': 'john.doe@example.com',
            'ip_address': '192.168.1.1',
            'timestamp': datetime.now(timezone.utc).isoformat()
        },
        {
            'application_id': 'APP002',
            'device_id': 'DEV123',
            'email': 'john.doe@example.com',
            'ip_address': '192.168.1.1',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    ]

@pytest.mark.asyncio
async def test_detect_synthetic_id(fraud_detector, sample_application_data):
    result = await fraud_detector.detect_synthetic_id(sample_application_data)
    assert isinstance(result, dict)
    assert 'is_synthetic' in result
    assert 'risk_score' in result
    assert 'risk_level' in result
    assert 'patterns_detected' in result
    assert 'timestamp' in result

@pytest.mark.asyncio
async def test_analyze_device_risk(fraud_detector, sample_application_data):
    result = await fraud_detector.analyze_device_risk(sample_application_data)
    assert isinstance(result, dict)
    assert 'is_risky' in result
    assert 'risk_score' in result
    assert 'risk_level' in result
    assert 'patterns_detected' in result
    assert 'timestamp' in result

@pytest.mark.asyncio
async def test_detect_collusion(fraud_detector, sample_historical_data):
    result = await fraud_detector.detect_collusion(sample_historical_data)
    assert isinstance(result, dict)
    assert 'collusion_detected' in result
    assert 'risk_level' in result
    assert 'patterns_detected' in result
    assert 'timestamp' in result

@pytest.mark.asyncio
async def test_check_velocity(fraud_detector, sample_application_data, sample_historical_data):
    result = await fraud_detector.check_velocity(sample_application_data, sample_historical_data)
    assert isinstance(result, dict)
    assert 'high_velocity' in result
    assert 'risk_level' in result
    assert 'threshold_violations' in result
    assert 'counts' in result
    assert 'timestamp' in result

def test_check_phone_patterns(fraud_detector):
    # Test sequential numbers
    assert fraud_detector._check_phone_patterns('1234567890') > 0.7
    
    # Test repeated numbers
    assert fraud_detector._check_phone_patterns('1111222233') > 0.7
    
    # Test valid phone number
    assert fraud_detector._check_phone_patterns('5551234567') < 0.3

def test_check_name_patterns(fraud_detector):
    # Test suspicious name
    assert fraud_detector._check_name_patterns('Test User') > 0.7
    
    # Test valid name
    assert fraud_detector._check_name_patterns('John Doe') < 0.3

def test_check_address_patterns(fraud_detector):
    # Test suspicious address
    assert fraud_detector._check_address_patterns('Test Address') > 0.7
    
    # Test valid address
    assert fraud_detector._check_address_patterns('123 Main St') < 0.3

def test_check_ip_risk(fraud_detector):
    # Test private IP
    assert fraud_detector._check_ip_risk('192.168.1.1') > 0.7
    
    # Test public IP
    assert fraud_detector._check_ip_risk('8.8.8.8') < 0.3

def test_check_browser_risk(fraud_detector):
    # Test suspicious browser data
    suspicious_browser = {
        'user_agent': 'Chrome',
        'accept_language': None,
        'accept_encoding': None
    }
    assert fraud_detector._check_browser_risk(suspicious_browser) > 0.7
    
    # Test valid browser data
    valid_browser = {
        'user_agent': 'Chrome/91.0.4472.124',
        'accept_language': 'en-US,en;q=0.9',
        'accept_encoding': 'gzip, deflate, br'
    }
    assert fraud_detector._check_browser_risk(valid_browser) < 0.3 