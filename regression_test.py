import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

try:
    from src.che.core.ecosystem import create_heterogeneous_population, create_homogeneous_population
    from src.che.core.task import Task
    from src.che.agents.agent_factory import AgentFactory
    from src.che.agents.ollama_agent import create_critical_ollama_agent, create_awakened_ollama_agent, create_standard_ollama_agent
    
    print("SUCCESS: All modules imported successfully")
    
    # Test that existing Ollama functionality still works
    print("Testing existing Ollama functionality...")
    
    # Test Ollama agent factory functions
    critical_ollama = create_critical_ollama_agent('critical_ollama_test', 'qwen:0.5b')
    awakened_ollama = create_awakened_ollama_agent('awakened_ollama_test', 'qwen:0.5b')
    standard_ollama = create_standard_ollama_agent('standard_ollama_test', 'qwen:0.5b')
    
    print("SUCCESS: Ollama agent factory functions work correctly")
    
    # Test ecosystem creation functions
    heterogeneous_eco = create_heterogeneous_population(12, "qwen:0.5b")
    homogeneous_eco = create_homogeneous_population(12, "qwen:0.5b", "standard")
    
    print(f"SUCCESS: Ecosystem creation functions work correctly")
    print(f"Heterogeneous population size: {heterogeneous_eco.get_population_size()}")
    print(f"Homogeneous population size: {homogeneous_eco.get_population_size()}")
    
    # Test AgentFactory with Ollama agents
    ollama_agent = AgentFactory.create_agent('ollama', 'factory_ollama_test', {"model": "qwen:0.5b"})
    critical_factory = AgentFactory.create_critical_agent('ollama', 'factory_critical_test', 'qwen:0.5b')
    awakened_factory = AgentFactory.create_awakened_agent('ollama', 'factory_awakened_test', 'qwen:0.5b')
    standard_factory = AgentFactory.create_standard_agent('ollama', 'factory_standard_test', 'qwen:0.5b')
    
    print("SUCCESS: AgentFactory works with Ollama agents")
    
    # Test that new Cloud functionality works alongside existing functionality
    print("Testing new Cloud functionality alongside existing functionality...")
    
    cloud_agent = AgentFactory.create_agent('cloud', 'factory_cloud_test', {
        'service_type': 'openai',
        'api_key': 'test-key',
        'model_name': 'gpt-3.5-turbo'
    })
    
    critical_cloud = AgentFactory.create_critical_agent('cloud', 'factory_cloud_critical_test', 'gpt-3.5-turbo', service_type='openai', api_key='test-key')
    
    print("SUCCESS: New Cloud functionality works alongside existing Ollama functionality")
    
    # Test that we can mix Ollama and Cloud agents in the same ecosystem
    from src.che.core.ecosystem import Ecosystem
    
    mixed_ecosystem = Ecosystem()
    mixed_ecosystem.add_agent(ollama_agent)
    mixed_ecosystem.add_agent(cloud_agent)
    
    print(f"SUCCESS: Mixed ecosystem created with {mixed_ecosystem.get_population_size()} agents")
    
    print("\nRegression test passed! All existing functionality still works.")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()