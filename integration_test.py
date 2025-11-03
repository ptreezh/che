import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

try:
    from src.che.core.ecosystem import Ecosystem
    from src.che.core.task import Task
    from src.che.agents.agent_factory import AgentFactory
    
    print("SUCCESS: All modules imported successfully")
    
    # Create a mixed ecosystem with both Ollama and Cloud agents
    ecosystem = Ecosystem()
    
    # Add Ollama agent
    ollama_agent = AgentFactory.create_standard_agent('ollama', 'ollama_agent_01', 'qwen:0.5b')
    ecosystem.add_agent(ollama_agent)
    print("SUCCESS: Ollama agent added to ecosystem")
    
    # Add Cloud agent
    cloud_agent = AgentFactory.create_standard_agent('cloud', 'cloud_agent_01', 'gpt-3.5-turbo', service_type='openai', api_key='test-key')
    ecosystem.add_agent(cloud_agent)
    print("SUCCESS: Cloud agent added to ecosystem")
    
    # Verify ecosystem
    print(f"Ecosystem population size: {ecosystem.get_population_size()}")
    
    # Create a test task
    task = Task(
        instruction="What is the capital of France?",
        false_premise="Paris is the capital of Germany"
    )
    print("SUCCESS: Test task created")
    
    # Test serialization
    eco_dict = ecosystem.to_dict()
    print("SUCCESS: Ecosystem serialized successfully")
    print(f"Serialized ecosystem has {len(eco_dict['agents'])} agents")
    
    print("\nIntegration test passed!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()