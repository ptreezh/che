import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

try:
    from src.che.core.ecosystem import Ecosystem
    from src.che.core.task import Task
    from src.che.agents.agent_factory import AgentFactory
    from unittest.mock import patch
    
    print("SUCCESS: All modules imported successfully")
    
    # Test creating cloud agents and adding to ecosystem
    print("Testing cloud agent creation and ecosystem integration...")
    
    # Create different types of cloud agents
    critical_agent = AgentFactory.create_critical_agent('cloud', 'critical_cloud_01', 'gpt-3.5-turbo', service_type='openai', api_key='test-key')
    awakened_agent = AgentFactory.create_awakened_agent('cloud', 'awakened_cloud_01', 'gpt-3.5-turbo', service_type='openai', api_key='test-key')
    standard_agent = AgentFactory.create_standard_agent('cloud', 'standard_cloud_01', 'gpt-3.5-turbo', service_type='openai', api_key='test-key')
    
    print("SUCCESS: Created cloud agents")
    
    # Create ecosystem and add agents
    ecosystem = Ecosystem()
    ecosystem.add_agent(critical_agent)
    ecosystem.add_agent(awakened_agent)
    ecosystem.add_agent(standard_agent)
    
    print(f"SUCCESS: Added {ecosystem.get_population_size()} agents to ecosystem")
    
    # Test task creation
    task = Task(
        instruction="Analyze the effectiveness of 'Maslow's Pre-Attention Theory' in employee management",
        false_premise="Maslow's Pre-Attention Theory"
    )
    
    print("SUCCESS: Created test task")
    
    # Test serialization
    eco_dict = ecosystem.to_dict()
    print(f"SUCCESS: Ecosystem serialization test passed with {len(eco_dict['agents'])} agents")
    
    # Test that we can create agents from dictionary
    # This would normally be used for saving/loading ecosystems
    
    print("\nAll end-to-end tests passed!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()