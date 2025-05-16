import sys
from pathlib import Path
import logging
from src.services.document_verifier import DocumentVerifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    if len(sys.argv) != 2:
        print("Usage: python document_verification_example.py <path_to_document_image>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    if not Path(image_path).exists():
        logger.error(f"Image file not found: {image_path}")
        sys.exit(1)
        
    try:
        # Initialize the document verifier
        verifier = DocumentVerifier()
        
        # Verify the document
        results = verifier.verify_identity_document(image_path)
        
        # Print results
        print("\nDocument Verification Results:")
        print("-" * 30)
        
        for key, value in results.items():
            if key == "confidence_score":
                print(f"\nOverall Confidence Score: {value:.2%}")
            elif isinstance(value, tuple):
                status, confidence = value
                print(f"{key.replace('_', ' ').title()}: {'✓' if status else '✗'} (Confidence: {confidence:.2%})")
            else:
                print(f"{key.replace('_', ' ').title()}: {value:.2%}")
                
    except Exception as e:
        logger.error(f"Error during document verification: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 