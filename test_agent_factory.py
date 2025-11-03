import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

try:
    from src.che.agents.agent_factory import AgentFactory
    
    print("SUCCESS: AgentFactory imported successfully")
    
    # Test Ollama agent creation
    ollama_config = {
        "model": "qwen:0.5b",
        "prompt": "You are a helpful assistant."
    }
    
    ollama_agent = AgentFactory.create_agent('ollama', 'ollama_test_agent', ollama_config)
    print("SUCCESS: Ollama agent created successfully")
    print(f"Ollama Agent ID: {ollama_agent.agent_id}")
    print(f"Ollama Agent Model: {ollama_agent.config['model']}")
    
    # Test Cloud agent creation
    cloud_config = {
        'service_type': 'openai',
        'api_key': 'test-key',
        'model_name': 'gpt-3.5-turbo'
    }
    
    cloud_agent = AgentFactory.create_agent('cloud', 'cloud_test_agent', cloud_config)
    print("SUCCESS: Cloud agent created successfully")
    print(f"Cloud Agent ID: {cloud_agent.agent_id}")
    print(f"Cloud Agent Service Type: {cloud_agent.service_type}")
    print(f"Cloud Agent Model: {cloud_agent.model_name}")
    
    # Test critical agent creation
    critical_ollama = AgentFactory.create_critical_agent('ollama', 'critical_ollama_agent', 'qwen:0.5b')
    print("SUCCESS: Critical Ollama agent created successfully")
    
    critical_cloud = AgentFactory.create_critical_agent('cloud', 'critical_cloud_agent', 'gpt-3.5-turbo', service_type='openai', api_key='test-key')
    print("SUCCESS: Critical Cloud agent created successfully")
    
    # Test awakened agent creation
    awakened_ollama = AgentFactory.create_awakened_agent('ollama', 'awakened_ollama_agent', 'qwen:0.5b')
    print("SUCCESS: Awakened Ollama agent created successfully")
    
    awakened_cloud = AgentFactory.create_awakened_agent('cloud', 'awakened_cloud_agent', 'gpt-3.5-turbo', service_type='openai', api_key='test-key')
    print("SUCCESS: Awakened Cloud agent created successfully")
    
    # Test standard agent creation
    standard_ollama = AgentFactory.create_standard_agent('ollama', 'standard_ollama_agent', 'qwen:0.5b')
    print("SUCCESS: Standard Ollama agent created successfully")
    
    standard_cloud = AgentFactory.create_standard_agent('cloud', 'standard_cloud_agent', 'gpt-3.5-turbo', service_type='openai', api_key='test-key')
    print("SUCCESS: Standard Cloud agent created successfully")
    
    print("\nAll AgentFactory tests passed!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()