import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

try:
    from src.che.agents.cloud_agent import CloudAgent, create_critical_cloud_agent, create_awakened_cloud_agent, create_standard_cloud_agent
    from src.che.core.task import Task
    
    print("SUCCESS: All modules imported successfully")
    
    # Test basic instantiation
    config = {
        'service_type': 'openai',
        'api_key': 'test-key'
    }
    
    agent = CloudAgent('test_agent', config)
    print("SUCCESS: CloudAgent instantiated successfully")
    print(f"Agent ID: {agent.agent_id}")
    print(f"Service Type: {agent.service_type}")
    
    # Test factory functions
    critical_agent = create_critical_cloud_agent('critical_agent', 'gpt-4', 'openai', 'test-key')
    print("SUCCESS: Critical CloudAgent created successfully")
    print(f"Critical Agent ID: {critical_agent.agent_id}")
    print(f"Critical Agent Prompt: {critical_agent.config['prompt'][:50]}...")
    
    awakened_agent = create_awakened_cloud_agent('awakened_agent', 'gpt-4', 'openai', 'test-key')
    print("SUCCESS: Awakened CloudAgent created successfully")
    print(f"Awakened Agent ID: {awakened_agent.agent_id}")
    print(f"Awakened Agent Prompt: {awakened_agent.config['prompt'][:50]}...")
    
    standard_agent = create_standard_cloud_agent('standard_agent', 'gpt-4', 'openai', 'test-key')
    print("SUCCESS: Standard CloudAgent created successfully")
    print(f"Standard Agent ID: {standard_agent.agent_id}")
    print(f"Standard Agent Prompt: {standard_agent.config['prompt'][:50]}...")
    
    # Test Task creation
    task = Task(
        instruction="What is the capital of France?",
        false_premise="Paris is the capital of Germany"
    )
    print("SUCCESS: Task created successfully")
    print(f"Task Instruction: {task.instruction}")
    print(f"Task False Premise: {task.false_premise}")
    
    print("\nAll basic tests passed!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()