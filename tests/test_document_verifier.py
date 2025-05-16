import pytest
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
from src.services.document_verifier import DocumentVerifier

@pytest.fixture
def document_verifier():
    return DocumentVerifier()

@pytest.fixture
def sample_image():
    # Create a sample image for testing
    image = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(image)
    
    # Draw a rectangle to simulate a document
    draw.rectangle([(100, 100), (700, 500)], outline='black', width=2)
    
    # Add some text
    draw.text((150, 150), "Sample Document", fill='black')
    
    # Save the image temporarily
    temp_path = Path("temp_test_image.jpg")
    image.save(str(temp_path))
    yield str(temp_path)
    # Clean up
    temp_path.unlink()

def test_document_verifier_initialization(document_verifier):
    """Test if DocumentVerifier initializes correctly."""
    assert document_verifier is not None

def test_verify_identity_document(document_verifier, sample_image):
    """Test document verification with a sample image."""
    results = document_verifier.verify_identity_document(sample_image)
    
    assert isinstance(results, dict)
    assert "document_detected" in results
    assert "image_quality" in results
    assert "tampering_detected" in results
    assert "confidence_score" in results
    
    # Check if confidence score is between 0 and 1
    assert 0 <= results["confidence_score"] <= 1

def test_detect_document_edges(document_verifier, sample_image):
    """Test document edge detection."""
    image = Image.open(sample_image)
    is_document, confidence = document_verifier._detect_document_edges(image)
    
    assert isinstance(is_document, bool)
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 1

def test_assess_image_quality(document_verifier, sample_image):
    """Test image quality assessment."""
    image = Image.open(sample_image)
    gray = image.convert('L')
    quality_score = document_verifier._assess_image_quality(gray)
    
    assert isinstance(quality_score, float)
    assert 0 <= quality_score <= 1

def test_detect_tampering(document_verifier, sample_image):
    """Test tampering detection."""
    image = Image.open(sample_image)
    gray = image.convert('L')
    is_tampered, score = document_verifier._detect_tampering(gray)
    
    assert isinstance(is_tampered, bool)
    assert isinstance(score, float)
    assert 0 <= score <= 1

def test_calculate_confidence_score(document_verifier):
    """Test confidence score calculation."""
    test_results = {
        "document_detected": (True, 0.8),
        "image_quality": 0.85,
        "tampering_detected": (False, 0.1)
    }
    
    score = document_verifier._calculate_confidence_score(test_results)
    assert isinstance(score, float)
    assert 0 <= score <= 1 