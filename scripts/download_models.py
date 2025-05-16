import os
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url: str, destination: Path) -> bool:
    """Download a file from a URL to the specified destination."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(destination, 'wb') as f:
            for data in response.iter_content(block_size):
                f.write(data)
                
        return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        return False

def main():
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Download EAST text detection model
    east_model_url = "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb"
    east_model_path = models_dir / "frozen_east_text_detection.pb"
    
    if not east_model_path.exists():
        logger.info("Downloading EAST text detection model...")
        if download_file(east_model_url, east_model_path):
            logger.info("EAST model downloaded successfully!")
        else:
            logger.error("Failed to download EAST model")
            return
    else:
        logger.info("EAST model already exists")
    
    # Update the model path in document_verifier.py
    verifier_path = Path("src/services/document_verifier.py")
    if verifier_path.exists():
        with open(verifier_path, 'r') as f:
            content = f.read()
        
        # Update the model path
        updated_content = content.replace(
            '"frozen_east_text_detection.pb"',
            f'"{str(east_model_path)}"'
        )
        
        with open(verifier_path, 'w') as f:
            f.write(updated_content)
        
        logger.info("Updated model path in document_verifier.py")

if __name__ == "__main__":
    main() 