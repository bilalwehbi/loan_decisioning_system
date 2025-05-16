# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install NumPy first with a specific version
RUN pip install --no-cache-dir numpy==1.24.3

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p models data logs

# Copy project files
COPY . .

# Copy model files
COPY models/risk_model.joblib /app/models/
COPY models/fraud_model.joblib /app/models/

# Expose port
EXPOSE 8000

# Set the entrypoint
ENTRYPOINT ["uvicorn", "src.api.loan_api:app", "--host", "0.0.0.0", "--port", "8000"] 