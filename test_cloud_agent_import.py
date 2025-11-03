import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

try:
    from src.che.agents.cloud_agent import CloudAgent
    print("SUCCESS: CloudAgent imported successfully")
    
    # Test basic instantiation
    config = {
        'service_type': 'openai',
        'api_key': 'test-key'
    }
    
    agent = CloudAgent('test_agent', config)
    print("SUCCESS: CloudAgent instantiated successfully")
    print(f"Agent ID: {agent.agent_id}")
    print(f"Service Type: {agent.service_type}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()