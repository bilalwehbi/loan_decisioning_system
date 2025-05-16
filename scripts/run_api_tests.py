import pytest
import sys

def run_tests():
    """Run the API tests"""
    print("Running API tests...")
    result = pytest.main(['tests/test_api.py', '-v'])
    if result != 0:
        print("Some tests failed or the API server is not running.")
        sys.exit(result)
    else:
        print("All tests passed!")

if __name__ == "__main__":
    run_tests() 