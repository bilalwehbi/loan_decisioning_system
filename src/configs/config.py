# src/configs/config.py
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "data"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "models"))

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Log directory paths for debugging
import logging
logger = logging.getLogger(__name__)
logger.info(f"BASE_DIR: {BASE_DIR}")
logger.info(f"DATA_DIR: {DATA_DIR}")
logger.info(f"MODEL_DIR: {MODEL_DIR}")
logger.info(f"MODEL_DIR exists: {os.path.exists(MODEL_DIR)}")
logger.info(f"MODEL_DIR contents: {os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else 'Directory not found'}")

# Model configuration
MODEL_CONFIG = {
    "credit_risk": {
        "algorithm": "lightgbm",
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary",
            "random_state": 42
        },
        "threshold": {
            "low_risk": 700,  # Score >= 700 is "Low Risk"
            "review": 600,    # Score >= 600 and < 700 is "Review"
                              # Score < 600 is "Decline"
        },
        "training_params": {}
    },
    "fraud_detection": {
        "algorithm": "lightgbm",
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary",
            "random_state": 42
        },
        "threshold": 0.5,  # Fraud score >= 0.5 flags as potential fraud
        "training_params": {}
    }
}

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "timeout": 60,
    "rps_limit": 100  # Requests per second limit
}

# Performance targets
PERFORMANCE_TARGETS = {
    "precision": 0.90,  # Target precision >= 90%
    "false_positive_rate": 0.05,  # Target FPR <= 5%
    "latency": 2.0  # Target latency <= 2 seconds
}