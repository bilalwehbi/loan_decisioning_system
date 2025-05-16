# API Documentation

## Business Logic Overview

The Loan Decisioning System implements a sophisticated decision-making process that combines multiple factors to assess loan applications and detect fraud. Here's how the system works:

### Loan Assessment Process

1. **Initial Application Review**
   - Validates applicant information
   - Checks for required documentation
   - Performs basic data quality checks

2. **Fraud Risk Assessment**
   - Analyzes applicant patterns
   - Checks device and location data
   - Validates document authenticity
   - Calculates fraud risk score

3. **Credit Risk Assessment**
   - Evaluates credit history
   - Analyzes income and employment
   - Calculates debt-to-income ratio
   - Generates credit risk score

4. **Final Decision**
   - Combines fraud and credit risk scores
   - Applies business rules
   - Generates final decision and terms

## API Endpoints

### Loan API

#### 1. Assess Loan Application
```http
POST /loan/assess
```

**Purpose**: Evaluates a loan application and provides a decision.

**Request Body**:
```json
{
  "applicant": {
    "name": "string",
    "email": "string",
    "phone": "string",
    "address": "string",
    "income": "number",
    "employment_status": "string",
    "credit_score": "number"
  },
  "loan_details": {
    "amount": "number",
    "term": "number",
    "purpose": "string"
  },
  "documents": {
    "id_document": "base64_string",
    "proof_of_income": "base64_string"
  }
}
```

**Response**:
```json
{
  "application_id": "string",
  "decision": "string",
  "risk_score": "number",
  "interest_rate": "number",
  "terms": {
    "monthly_payment": "number",
    "total_interest": "number",
    "total_payment": "number"
  },
  "reasons": ["string"]
}
```

**Business Logic**:
- Validates all required fields
- Performs fraud check
- Calculates risk score
- Determines interest rate
- Generates loan terms

**Testing**:
```bash
# Test valid application
curl -X POST http://localhost:8000/loan/assess \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "applicant": {
      "name": "John Doe",
      "email": "john@example.com",
      "phone": "5551234567",
      "address": "123 Main St",
      "income": 75000,
      "employment_status": "employed",
      "credit_score": 720
    },
    "loan_details": {
      "amount": 10000,
      "term": 12,
      "purpose": "home_improvement"
    }
  }'

# Test high-risk application
curl -X POST http://localhost:8000/loan/assess \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "applicant": {
      "name": "John Doe",
      "email": "john@example.com",
      "phone": "1111222233",
      "address": "123 Main St",
      "income": 30000,
      "employment_status": "self_employed",
      "credit_score": 580
    },
    "loan_details": {
      "amount": 50000,
      "term": 36,
      "purpose": "debt_consolidation"
    }
  }'
```

#### 2. Get Application History
```http
GET /loan/history
```

**Purpose**: Retrieves historical loan applications.

**Query Parameters**:
- `start_date`: Start date for filtering (YYYY-MM-DD)
- `end_date`: End date for filtering (YYYY-MM-DD)
- `status`: Application status filter
- `page`: Page number for pagination
- `limit`: Number of results per page

**Response**:
```json
{
  "applications": [
    {
      "application_id": "string",
      "applicant_name": "string",
      "application_date": "string",
      "status": "string",
      "amount": "number",
      "decision": "string"
    }
  ],
  "total": "number",
  "page": "number",
  "total_pages": "number"
}
```

**Testing**:
```bash
# Test basic history retrieval
curl -X GET "http://localhost:8000/loan/history?page=1&limit=10" \
  -H "X-API-Key: your-api-key"

# Test filtered history
curl -X GET "http://localhost:8000/loan/history?start_date=2024-01-01&end_date=2024-03-31&status=approved" \
  -H "X-API-Key: your-api-key"
```

### Fraud API

#### 1. Perform Fraud Check
```http
POST /fraud/check
```

**Purpose**: Analyzes application data for potential fraud.

**Request Body**:
```json
{
  "applicant": {
    "name": "string",
    "email": "string",
    "phone": "string",
    "address": "string"
  },
  "device_info": {
    "ip": "string",
    "user_agent": "string",
    "device_id": "string",
    "location": {
      "country": "string",
      "city": "string",
      "coordinates": {
        "latitude": "number",
        "longitude": "number"
      }
    }
  },
  "historical_data": {
    "previous_applications": "number",
    "rejected_applications": "number",
    "last_application_date": "string"
  }
}
```

**Response**:
```json
{
  "fraud_score": "number",
  "risk_level": "string",
  "risk_factors": [
    {
      "factor": "string",
      "score": "number",
      "description": "string"
    }
  ],
  "recommendation": "string"
}
```

**Business Logic**:
- Analyzes phone number patterns
- Validates name consistency
- Checks address format
- Evaluates device risk
- Analyzes historical patterns
- Calculates overall fraud score

**Testing**:
```bash
# Test normal application
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
      "user_agent": "Mozilla/5.0...",
      "device_id": "device123",
      "location": {
        "country": "US",
        "city": "New York",
        "coordinates": {
          "latitude": 40.7128,
          "longitude": -74.0060
        }
      }
    }
  }'

# Test suspicious application
curl -X POST http://localhost:8000/fraud/check \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "applicant": {
      "name": "John Doe",
      "email": "john@example.com",
      "phone": "1111222233",
      "address": "123 Main St"
    },
    "device_info": {
      "ip": "1.2.3.4",
      "user_agent": "Mozilla/5.0...",
      "device_id": "device123",
      "location": {
        "country": "XX",
        "city": "Unknown",
        "coordinates": {
          "latitude": 0,
          "longitude": 0
        }
      }
    }
  }'
```

## Testing Procedures

### 1. Unit Testing

Run specific test files:
```bash
# Test loan API
pytest tests/test_loan_api.py -v

# Test fraud API
pytest tests/test_fraud_api.py -v

# Test specific test case
pytest tests/test_loan_api.py::test_valid_loan_application -v
```

### 2. Integration Testing

Test the complete flow:
```bash
# Run all integration tests
pytest tests/test_integration.py -v

# Test specific integration scenario
pytest tests/test_integration.py::test_loan_application_flow -v
```

### 3. Performance Testing

Test system performance:
```bash
# Run load tests
python scripts/load_test.py

# Run stress tests
python scripts/stress_test.py
```

### 4. Security Testing

Test security measures:
```bash
# Test authentication
pytest tests/test_security.py::test_authentication -v

# Test authorization
pytest tests/test_security.py::test_authorization -v
```

## Error Handling

The API uses standard HTTP status codes and returns error responses in the following format:

```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": {
      "field": "string",
      "issue": "string"
    }
  }
}
```

Common error codes:
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 422: Validation Error
- 500: Internal Server Error

## Rate Limiting

The API implements rate limiting to prevent abuse:
- 100 requests per minute per API key
- 1000 requests per hour per API key
- 10000 requests per day per API key

Rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 99
X-RateLimit-Reset: 1617235200
```

## Best Practices

1. **Error Handling**
   - Always check response status codes
   - Handle rate limiting errors
   - Implement retry logic for transient failures

2. **Security**
   - Keep API keys secure
   - Use HTTPS for all requests
   - Validate all input data

3. **Performance**
   - Implement caching where appropriate
   - Use pagination for large result sets
   - Monitor API response times

4. **Testing**
   - Write comprehensive unit tests
   - Test edge cases and error conditions
   - Implement integration tests
   - Perform regular security testing 