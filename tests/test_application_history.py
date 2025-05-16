import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
from src.api.main import app

client = TestClient(app)

# Test data
VALID_API_KEY = "CRED$#bil1@" 