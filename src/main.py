# src/main.py
import os
import sys
import logging
from pathlib import Path
from fastapi import FastAPI

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.configs.config import DATA_DIR
from src.data.synthetic_data_generator import generate_full_dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('loan_decisioning.log')
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

def run_data_generation():
    """Run synthetic data generation"""
    logger.info("Starting synthetic data generation...")
    generate_full_dataset(50000, DATA_DIR)
    logger.info("Synthetic data generation completed.")

if __name__ == "__main__":
    logger.info("Loan Decisioning System: Project Setup")
    
    # Create necessary directories
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Generate synthetic data
    run_data_generation()
    
    logger.info("Project setup completed successfully.")