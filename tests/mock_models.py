import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os

def create_mock_models():
    """Create mock model files for testing."""
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Create a simple random forest classifier for risk model
    risk_model = RandomForestClassifier(n_estimators=10, random_state=42)
    X_dummy = np.random.rand(100, 10)  # 100 samples, 10 features
    y_dummy = np.random.randint(0, 2, 100)  # Binary classification
    risk_model.fit(X_dummy, y_dummy)
    
    # Create a simple random forest classifier for fraud model
    fraud_model = RandomForestClassifier(n_estimators=10, random_state=42)
    fraud_model.fit(X_dummy, y_dummy)
    
    # Save models
    joblib.dump(risk_model, "models/risk_model.joblib")
    joblib.dump(fraud_model, "models/fraud_model.joblib")
    
    print("Mock models created successfully!")

if __name__ == "__main__":
    create_mock_models() 