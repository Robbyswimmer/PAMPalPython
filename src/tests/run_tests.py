"""
Run all unit tests for the PAMpal Python implementation.
"""
import unittest
import sys
import os

# Ensure the pampal package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import all test modules
from tests.test_settings import TestPAMpalSettings
from tests.test_acoustic_event import TestAcousticEvent
from tests.test_acoustic_study import TestAcousticStudy
from tests.test_processing import TestProcessing

# Create a test suite
def create_test_suite():
    """Create a test suite with all tests."""
    test_suite = unittest.TestSuite()
    
    # Add all test cases
    test_suite.addTest(unittest.makeSuite(TestPAMpalSettings))
    test_suite.addTest(unittest.makeSuite(TestAcousticEvent))
    test_suite.addTest(unittest.makeSuite(TestAcousticStudy))
    test_suite.addTest(unittest.makeSuite(TestProcessing))
    
    return test_suite

if __name__ == '__main__':
    # Create test suite
    suite = create_test_suite()
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Report results
    print("\nTest Results Summary:")
    print(f"Ran {result.testsRun} tests")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    # Exit with appropriate status code
    sys.exit(len(result.failures) + len(result.errors))
