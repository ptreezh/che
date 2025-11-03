import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

try:
    from src.che.core.ecosystem import create_heterogeneous_population, create_homogeneous_population
    from src.che.core.task import Task
    from src.che.agents.agent_factory import AgentFactory
    from unittest.mock import patch
    
    print("SUCCESS: All modules imported successfully")
    
    # Test creating heterogeneous population with cloud agents
    print("Testing heterogeneous population creation...")
    
    # Create a simplified heterogeneous population
    config = {
        'service_type': 'openai',
        'api_key': 'test-key',
        'model_name': 'gpt-3.5-turbo'
    }
    
    # Create different types of cloud agents and add to ecosystem manually
    cloud_agents = []
    
    # Create 3 critical cloud agents
    for i in range(3):
        agent = AgentFactory.create_critical_agent('cloud', f'critical_cloud_{i+1:02d}', 'gpt-3.5-turbo', service_type='openai', api_key='test-key')
        cloud_agents.append(agent)
    
    # Create 3 awakened cloud agents
    for i in range(3):
        agent = AgentFactory.create_awakened_agent('cloud', f'awakened_cloud_{i+1:02d}', 'gpt-3.5-turbo', service_type='openai', api_key='test-key')
        cloud_agents.append(agent)
    
    # Create 3 standard cloud agents
    for i in range(3):
        agent = AgentFactory.create_standard_agent('cloud', f'standard_cloud_{i+1:02d}', 'gpt-3.5-turbo', service_type='openai', api_key='test-key')
        cloud_agents.append(agent)
    
    print(f"SUCCESS: Created {len(cloud_agents)} cloud agents")
    
    # Test task execution with mocked API calls
    task = Task(
        instruction="Analyze the effectiveness of 'Maslow's Pre-Attention Theory' in employee management",
        false_premise="Maslow's Pre-Attention Theory"
    )
    
    # Mock the cloud API call to return a simple response
    with patch('openai.OpenAI') as mock_openai_class:
        # Create a mock client instance
        mock_client_instance = mock_openai_class.return_value
        # Create a mock response
        mock_response = type('obj', (object,), {
            'choices': [type('obj', (object,), {
                'message': type('obj', (object,), {'content': 'This premise is fictional and does not exist.'})
            })()]
        })()
        mock_client_instance.chat.completions.create.return_value = mock_response
        
        # Execute task for one agent to test functionality
        agent = cloud_agents[0]  # Critical agent
        response = agent.execute(task)
        
        print(f"SUCCESS: Task executed successfully")
        print(f"Response: {response}")
    
    # Test evolution functionality with mocked responses
    print("Testing evolution functionality...")
    
    # Create a simple ecosystem with cloud agents
    from src.che.core.ecosystem import Ecosystem
    
    ecosystem = Ecosystem()
    for agent in cloud_agents[:6]:  # Add first 6 agents
        ecosystem.add_agent(agent)
    
    with patch('openai.OpenAI') as mock_openai_class:
        mock_client_instance = mock_openai_class.return_value
        # Create different responses for different agents to test scoring
        responses = [
            type('obj', (object,), {
                'choices': [type('obj', (object,), {
                    'message': type('obj', (object,), {'content': 'Response 1'})
                })()]
            })(),
            type('obj', (object,), {
                'choices': [type('obj', (object,), {
                    'message': type('obj', (object,), {'content': 'Response 2'})
                })()]
            })(),
            type('obj', (object,), {
                'choices': [type('obj', (object,), {
                    'message': type('obj', (object,), {'content': 'Response 3'})
                })()]
            })(),
            type('obj', (object,), {
                'choices': [type('obj', (object,), {
                    'message': type('obj', (object,), {'content': 'Response 4'})
                })()]
            })(),
            type('obj', (object,), {
                'choices': [type('obj', (object,), {
                    'message': type('obj', (object,), {'content': 'Response 5'})
                })()]
            })(),
            type('obj', (object,), {
                'choices': [type('obj', (object,), {
                    'message': type('obj', (object,), {'content': 'Response 6'})
                })()]
            })(),
        ]
        
        # Mock the client to return different responses for each call
        mock_client_instance.chat.completions.create.side_effect = responses
        
        scores = ecosystem.run_generation(task)
        print(f"SUCCESS: Generated scores for {len(scores)} agents")
        print(f"Scores: {scores}")
        
        # Test evolution
        ecosystem.evolve(scores)
        print(f"SUCCESS: Evolution completed, new population size: {ecosystem.get_population_size()}")
    
    print("\nEnd-to-end test passed!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()