# src/regenerate_data.py
import os
import sys
import logging
from pathlib import Path

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
        logging.FileHandler('data_generation.log')
    ]
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Regenerating synthetic data...")
    generate_full_dataset(10000, DATA_DIR)  # Reduced to 10,000 for faster generation
    logger.info("Data generation completed.")