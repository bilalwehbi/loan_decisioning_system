from PIL import Image, ImageEnhance, ImageFilter, ImageChops
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentVerifier:
    def __init__(self):
        """Initialize the DocumentVerifier with necessary configurations."""
        pass
        
    def verify_identity_document(self, image_path: str) -> Dict[str, any]:
        """
        Verify an identity document (ID card, passport, etc.)
        
        Args:
            image_path: Path to the document image
            
        Returns:
            Dict containing verification results and confidence scores
        """
        try:
            # Read the image
            image = Image.open(image_path)
            if image is None:
                raise ValueError(f"Could not read image from {image_path}")
            
            # Convert to grayscale for processing
            gray = image.convert('L')
            
            # Perform document verification checks
            results = {
                "document_detected": self._detect_document_edges(image),
                "image_quality": self._assess_image_quality(gray),
                "tampering_detected": self._detect_tampering(gray)
            }
            
            # Calculate overall confidence score
            results["confidence_score"] = self._calculate_confidence_score(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in document verification: {str(e)}")
            raise
    
    def _detect_document_edges(self, image: Image.Image) -> Tuple[bool, float]:
        """Detect document edges and verify document shape."""
        # Convert to grayscale
        gray = image.convert('L')
        
        # Apply edge detection
        edges = gray.filter(ImageFilter.FIND_EDGES)
        
        # Convert to numpy array for processing
        edge_array = np.array(edges)
        
        # Calculate edge density
        edge_density = np.mean(edge_array > 128)
        
        # Check if the image has clear edges (indicating a document)
        is_document = bool(edge_density > 0.1)
        confidence = min(1.0, edge_density * 5)
        
        return is_document, confidence
    
    def _assess_image_quality(self, gray_image: Image.Image) -> float:
        """Assess the quality of the document image."""
        # Convert to numpy array
        img_array = np.array(gray_image)
        
        # Calculate image brightness
        brightness = np.mean(img_array)
        
        # Calculate image contrast
        contrast = np.std(img_array)
        
        # Calculate image sharpness using Laplacian-like filter
        sharpness = np.std(np.array(gray_image.filter(ImageFilter.EDGE_ENHANCE)))
        
        # Combine metrics into a quality score
        quality_score = min(1.0, (
            brightness / 255.0 * 0.3 +
            contrast / 128.0 * 0.4 +
            sharpness / 50.0 * 0.3
        ))
        
        return quality_score
    
    def _detect_tampering(self, gray_image: Image.Image) -> Tuple[bool, float]:
        """Detect potential tampering in the document."""
        # Apply noise reduction
        denoised = gray_image.filter(ImageFilter.SMOOTH)
        
        # Calculate difference between original and denoised image
        diff = ImageChops.difference(gray_image, denoised)
        
        # Convert to numpy array
        diff_array = np.array(diff)
        
        # Calculate tampering score
        tampering_score = np.mean(diff_array) / 255.0
        
        return bool(tampering_score > 0.1), tampering_score
    
    def _calculate_confidence_score(self, results: Dict[str, any]) -> float:
        """Calculate overall confidence score based on all verification results."""
        weights = {
            "document_detected": 0.4,
            "image_quality": 0.4,
            "tampering_detected": 0.2
        }
        
        score = 0.0
        for key, weight in weights.items():
            if isinstance(results[key], tuple):
                score += results[key][1] * weight
            else:
                score += float(results[key]) * weight
                
        return score 