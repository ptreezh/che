import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

try:
    from tests.test_cloud_agent import TestCloudAgent
    import pytest
    
    # Create test instance
    test_instance = TestCloudAgent()
    
    # Run a simple test
    test_instance.test_cloud_agent_initialization()
    print("SUCCESS: test_cloud_agent_initialization passed")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()