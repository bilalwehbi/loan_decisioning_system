# Loan Decisioning System

A comprehensive loan decisioning system that combines machine learning models with fraud detection to make intelligent loan approval decisions.

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Running the System](#running-the-system)
- [Testing](#testing)
- [API Documentation](#api-documentation)
- [Fraud Detection](#fraud-detection)
- [Model Training](#model-training)
- [Troubleshooting](#troubleshooting)

## Overview

The Loan Decisioning System is a sophisticated platform that combines multiple components to make intelligent loan approval decisions:

1. **Loan Decision Engine**: Uses machine learning models to assess loan applications
2. **Fraud Detection System**: Identifies potential fraudulent applications
3. **Document Verification**: Validates identity documents and detects tampering
4. **API Services**: RESTful APIs for loan processing and fraud detection

## System Architecture

The system consists of three main microservices:

1. **Loan API Service** (`src/services/loan_api.py`)
   - Handles loan applications
   - Integrates with the loan decision engine
   - Provides application history and metrics

2. **Fraud API Service** (`src/services/fraud_api.py`)
   - Performs fraud checks
   - Analyzes historical data
   - Provides fraud risk scores

3. **Document Verification Service** (`src/services/document_verifier.py`)
   - Validates identity documents
   - Detects document tampering
   - Assesses image quality

## Prerequisites

- Python 3.8+
- Docker
- Redis
- Tesseract OCR (for document verification)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/loan_decisioning_system.git
cd loan_decisioning_system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Getting Started

### Step 1: Initial Setup
1. Make sure you have all prerequisites installed:
   - Python 3.8 or higher
   - Docker (if using containerized deployment)
   - Redis (for caching and session management)
   - Tesseract OCR (for document verification)

2. Clone and set up the project as described in the Installation section above.

### Step 2: Configure the System
1. Create and configure your `.env` file:
```bash
cp .env.example .env
```

2. Update the following key configurations in `.env`:
   - `API_KEY`: Your secure API key for authentication (default: "CRED$#bil1@")
   - `MODEL_PATH`: Path to your trained models
   - `REDIS_URL`: Redis connection string
   - `LOG_LEVEL`: Desired logging level (INFO/DEBUG)

3. Set up API Key Authentication:
   - The system uses API key authentication for all endpoints
   - Default API key is "CRED$#bil1@" (for development only)
   - For production, generate a secure API key:
     ```bash
     # Generate a secure API key
     openssl rand -base64 32
     ```
   - Update the API key in your `.env` file
   - Include the API key in all API requests using the `X-API-Key` header:
     ```bash
     curl -H "X-API-Key: your-api-key" http://localhost:8000/health
     ```

### Step 3: Start the Services
1. Start the main services:
```bash
# Start the loan API service
python src/services/loan_api.py

# In a new terminal, start the fraud API service
python src/services/fraud_api.py
```

2. Verify the services are running:
```bash
curl http://localhost:8000/health
```

### Step 4: Submit Your First Loan Application
1. Prepare your loan application data in JSON format:
```json
{
  "applicant": {
    "name": "John Doe",
    "email": "john@example.com",
    "phone": "5551234567",
    "address": "123 Main St"
  },
  "loan_details": {
    "amount": 10000,
    "term": 12,
    "purpose": "home_improvement"
  }
}
```

2. Submit the application:
```bash
curl -X POST http://localhost:8000/loan/assess \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d @application.json
```

### Step 5: Monitor and Debug
1. Check the logs:
   - Application logs: `loan_decisioning.log`
   - API logs: `api.log`
   - Model logs: `training.log`

2. Monitor the system:
   - Health check: `http://localhost:8000/health`
   - Metrics: `http://localhost:8000/metrics`

### Common Workflows

1. **Loan Application Process**:
   - Submit loan application
   - System performs fraud check
   - Document verification (if required)
   - Loan decision is made
   - Results are returned

2. **Fraud Check Process**:
   - Submit fraud check request
   - System analyzes patterns
   - Risk score is calculated
   - Results are returned

3. **Document Verification Process**:
   - Upload identity documents
   - System verifies authenticity
   - Quality check is performed
   - Results are returned

## Project Structure

```
loan_decisioning_system/
├── src/                    # Source code
│   ├── services/          # Microservices
│   ├── models/            # ML models
│   └── utils/             # Utility functions
├── tests/                 # Test files
├── data/                  # Data files
├── models/                # Trained models
├── docs/                  # Documentation
├── scripts/               # Utility scripts
└── examples/              # Example usage
```

## Running the System

### Using Docker

1. Build the Docker image:
```bash
docker build -t loan-decisioning-api:latest .
```

2. Run the container:
```bash
docker run -d -p 8000:8000 --name loan-decisioning-api loan-decisioning-api:latest
```

### Running Locally

1. Start the loan API service:
```bash
python src/services/loan_api.py
```

2. Start the fraud API service:
```bash
python src/services/fraud_api.py
```

## Testing

### Automated Testing

Run the test suite:
```bash
pytest tests/ -v
```

### Manual Testing

1. **Loan Application Test**:
```bash
curl -X POST http://localhost:8000/loan/assess \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "applicant": {
      "name": "John Doe",
      "email": "john@example.com",
      "phone": "5551234567",
      "address": "123 Main St"
    },
    "loan_details": {
      "amount": 10000,
      "term": 12,
      "purpose": "home_improvement"
    }
  }'
```

2. **Fraud Check Test**:
```bash
curl -X POST http://localhost:8000/fraud/check \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "applicant": {
      "name": "John Doe",
      "email": "john@example.com",
      "phone": "5551234567",
      "address": "123 Main St"
    },
    "device_info": {
      "ip": "192.168.1.1",
      "user_agent": "Mozilla/5.0..."
    }
  }'
```

## API Documentation

### Loan API Endpoints

- `POST /loan/assess`: Assess a loan application
- `GET /loan/history`: Get application history
- `GET /loan/metrics`: Get model metrics

### Fraud API Endpoints

- `POST /fraud/check`: Perform fraud check
- `GET /fraud/history`: Get fraud check history
- `GET /fraud/metrics`: Get fraud detection metrics

## API Testing and Flow

### API Response Codes and Meanings

1. **Success Responses**:
   - `200 OK`: Request successful
   - `201 Created`: Resource created successfully
   - `202 Accepted`: Request accepted for processing

2. **Error Responses**:
   - `400 Bad Request`: Invalid input data
   - `401 Unauthorized`: Invalid or missing API key
   - `403 Forbidden`: Valid API key but insufficient permissions
   - `404 Not Found`: Resource not found
   - `429 Too Many Requests`: Rate limit exceeded
   - `500 Internal Server Error`: Server-side error

### Complete API Flow Example

1. **Health Check**:
```bash
# Check if the service is running
curl -H "X-API-Key: your-api-key" http://localhost:8000/health
# Expected response: {"status": "healthy", "version": "1.0.0"}
```

2. **Fraud Check**:
```bash
# First, perform fraud check
curl -X POST http://localhost:8000/fraud/check \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "applicant": {
      "name": "John Doe",
      "email": "john@example.com",
      "phone": "5551234567",
      "address": "123 Main St"
    },
    "device_info": {
      "ip": "192.168.1.1",
      "user_agent": "Mozilla/5.0..."
    }
  }'
# Expected response: {"fraud_score": 0.02, "risk_level": "LOW", "checks_passed": true}
```

3. **Loan Assessment**:
```bash
# Then, submit loan application
curl -X POST http://localhost:8000/loan/assess \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "applicant": {
      "name": "John Doe",
      "email": "john@example.com",
      "phone": "5551234567",
      "address": "123 Main St"
    },
    "loan_details": {
      "amount": 10000,
      "term": 12,
      "purpose": "home_improvement"
    }
  }'
# Expected response: {
#   "application_id": "APP123456",
#   "status": "APPROVED",
#   "risk_score": 0.15,
#   "interest_rate": 5.5,
#   "max_amount": 15000
# }
```

4. **Check Application History**:
```bash
# Finally, check application history
curl -H "X-API-Key: your-api-key" http://localhost:8000/loan/history
# Expected response: {
#   "applications": [
#     {
#       "application_id": "APP123456",
#       "status": "APPROVED",
#       "timestamp": "2024-03-20T10:30:00Z",
#       "amount": 10000
#     }
#   ]
# }
```

### Response Field Explanations

1. **Fraud Check Response**:
   - `fraud_score`: 0-1 score indicating fraud probability (lower is better)
   - `risk_level`: "LOW", "MEDIUM", or "HIGH" risk assessment
   - `checks_passed`: Boolean indicating if all fraud checks passed

2. **Loan Assessment Response**:
   - `application_id`: Unique identifier for the application
   - `status`: "APPROVED", "REJECTED", or "PENDING"
   - `risk_score`: 0-1 score indicating overall risk (lower is better)
   - `interest_rate`: Offered interest rate percentage
   - `max_amount`: Maximum loan amount approved

3. **Application History Response**:
   - `applications`: Array of previous applications
   - `application_id`: Unique identifier for each application
   - `status`: Current status of the application
   - `timestamp`: When the application was processed
   - `amount`: Requested loan amount

### Testing Best Practices

1. **API Key Testing**:
   - Test with invalid API key
   - Test with missing API key
   - Test with expired API key

2. **Input Validation**:
   - Test with missing required fields
   - Test with invalid data types
   - Test with boundary values

3. **Rate Limiting**:
   - Test concurrent requests
   - Test request frequency limits
   - Test rate limit reset

4. **Error Handling**:
   - Test network timeouts
   - Test server errors
   - Test validation errors

## Fraud Detection

The system includes several fraud detection mechanisms:

1. **Phone Pattern Analysis**
   - Detects suspicious patterns in phone numbers
   - Identifies sequential numbers and repeated digits
   - Validates phone number format

2. **Name Pattern Analysis**
   - Detects suspicious name patterns
   - Validates name format and consistency

3. **Address Pattern Analysis**
   - Validates address format
   - Checks for suspicious patterns

4. **Device Risk Analysis**
   - Analyzes IP address risk
   - Checks browser fingerprint
   - Detects VPN usage

## Model Training

To train the models:

1. Prepare the data:
```bash
python scripts/prepare_data.py
```

2. Train the models:
```bash
python scripts/train_models.py
```

3. Evaluate the models:
```bash
python scripts/evaluate_models.py
```

## Troubleshooting

### Common Issues

1. **API Connection Issues**
   - Check if services are running
   - Verify API key is correct
   - Check network connectivity

2. **Model Loading Errors**
   - Verify model files exist
   - Check model version compatibility
   - Ensure sufficient memory

3. **Document Verification Failures**
   - Check Tesseract installation
   - Verify image quality
   - Check file format support

### Logs

- Application logs: `loan_decisioning.log`
- Model training logs: `training.log`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
