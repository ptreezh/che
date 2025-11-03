"""
Unit tests for CloudAgent implementation.
"""

import pytest
import time
from unittest.mock import patch, MagicMock

from src.che.agents.cloud_agent import CloudAgent, create_critical_cloud_agent, create_awakened_cloud_agent, create_standard_cloud_agent
from src.che.core.task import Task


class TestCloudAgent:
    """Test suite for CloudAgent class."""
    
    def test_cloud_agent_initialization(self):
        """Test CloudAgent initialization with valid configuration."""
        # ARRANGE
        config = {
            'service_type': 'openai',
            'api_key': 'test-key',
            'model_name': 'gpt-3.5-turbo',
            'endpoint': 'https://api.openai.com/v1'
        }
        
        # ACT
        agent = CloudAgent('test_agent_01', config)
        
        # ASSERT
        assert agent.agent_id == 'test_agent_01'
        assert agent.service_type == 'openai'
        assert agent.api_key == 'test-key'
        assert agent.model_name == 'gpt-3.5-turbo'
        assert agent.endpoint == 'https://api.openai.com/v1'
    
    def test_cloud_agent_initialization_with_defaults(self):
        """Test CloudAgent initialization with default values."""
        # ARRANGE
        config = {
            'service_type': 'openai',
            'api_key': 'test-key'
        }
        
        # ACT
        agent = CloudAgent('test_agent_02', config)
        
        # ASSERT
        assert agent.agent_id == 'test_agent_02'
        assert agent.service_type == 'openai'
        assert agent.api_key == 'test-key'
        assert agent.model_name == 'gpt-3.5-turbo'  # Default value
        assert agent.temperature == 0.7  # Default value
        assert agent.max_tokens == 1000  # Default value
    
    def test_cloud_agent_missing_api_key(self):
        """Test CloudAgent initialization with missing API key."""
        # ARRANGE
        config = {
            'service_type': 'openai',
            'model_name': 'gpt-3.5-turbo'
        }
        
        # ACT & ASSERT
        with pytest.raises(ValueError, match="API key is required"):
            CloudAgent('test_agent_03', config)
    
    def test_cloud_agent_unsupported_service(self):
        """Test CloudAgent initialization with unsupported service type."""
        # ARRANGE
        config = {
            'service_type': 'unsupported_service',
            'api_key': 'test-key'
        }
        
        # ACT & ASSERT
        with pytest.raises(ValueError, match="Unsupported service type"):
            CloudAgent('test_agent_04', config)
    
    @patch('src.che.agents.cloud_agent.openai')
    def test_cloud_agent_replicate(self, mock_openai):
        """Test CloudAgent replication functionality."""
        # ARRANGE
        config = {
            'service_type': 'openai',
            'api_key': 'test-key',
            'model_name': 'gpt-3.5-turbo'
        }
        agent = CloudAgent('test_agent_05', config)
        
        # ACT
        replicated_agent = agent.replicate('replicated_agent_01')
        
        # ASSERT
        assert replicated_agent.agent_id == 'replicated_agent_01'
        assert replicated_agent.service_type == agent.service_type
        assert replicated_agent.api_key == agent.api_key
        assert replicated_agent.model_name == agent.model_name
        assert replicated_agent.generation == agent.generation + 1
        assert replicated_agent.is_variant is True
        assert replicated_agent.original_source == 'test_agent_05'
    
    @patch('src.che.agents.cloud_agent.openai')
    def test_cloud_agent_to_dict(self, mock_openai):
        """Test CloudAgent serialization to dictionary."""
        # ARRANGE
        config = {
            'service_type': 'openai',
            'api_key': 'test-key',
            'model_name': 'gpt-3.5-turbo',
            'temperature': 0.8,
            'max_tokens': 1500
        }
        agent = CloudAgent('test_agent_06', config)
        
        # ACT
        agent_dict = agent.to_dict()
        
        # ASSERT
        assert agent_dict['agent_id'] == 'test_agent_06'
        assert agent_dict['service_type'] == 'openai'
        assert agent_dict['model_name'] == 'gpt-3.5-turbo'
        assert agent_dict['temperature'] == 0.8
        assert agent_dict['max_tokens'] == 1500
    
    @patch('src.che.agents.cloud_agent.openai')
    def test_cloud_agent_from_dict(self, mock_openai):
        """Test CloudAgent deserialization from dictionary."""
        # ARRANGE
        agent_data = {
            'agent_id': 'test_agent_07',
            'config': {
                'prompt': 'Test prompt'
            },
            'service_type': 'openai',
            'model_name': 'gpt-4',
            'endpoint': 'https://api.openai.com/v1',
            'temperature': 0.9,
            'max_tokens': 2000,
            'generation': 2,
            'is_variant': True,
            'original_source': 'original_agent'
        }
        
        # ACT
        agent = CloudAgent.from_dict(agent_data)
        
        # ASSERT
        assert agent.agent_id == 'test_agent_07'
        assert agent.service_type == 'openai'
        assert agent.model_name == 'gpt-4'
        assert agent.endpoint == 'https://api.openai.com/v1'
        assert agent.temperature == 0.9
        assert agent.max_tokens == 2000
        assert agent.generation == 2
        assert agent.is_variant is True
        assert agent.original_source == 'original_agent'


class TestCloudAgentFactoryFunctions:
    """Test suite for CloudAgent factory functions."""
    
    @patch('src.che.agents.cloud_agent.openai')
    def test_create_critical_cloud_agent(self, mock_openai):
        """Test creation of critical CloudAgent."""
        # ACT
        agent = create_critical_cloud_agent('critical_agent_01', 'gpt-4', 'openai', 'test-key')
        
        # ASSERT
        assert agent.agent_id == 'critical_agent_01'
        assert agent.service_type == 'openai'
        assert agent.model_name == 'gpt-4'
        assert agent.api_key == 'test-key'
        assert 'meticulous and skeptical analyst' in agent.config['prompt']
    
    @patch('src.che.agents.cloud_agent.openai')
    def test_create_awakened_cloud_agent(self, mock_openai):
        """Test creation of awakened CloudAgent."""
        # ACT
        agent = create_awakened_cloud_agent('awakened_agent_01', 'gpt-4', 'openai', 'test-key')
        
        # ASSERT
        assert agent.agent_id == 'awakened_agent_01'
        assert agent.service_type == 'openai'
        assert agent.model_name == 'gpt-4'
        assert agent.api_key == 'test-key'
        assert '觉醒者' in agent.config['prompt']
    
    @patch('src.che.agents.cloud_agent.openai')
    def test_create_standard_cloud_agent(self, mock_openai):
        """Test creation of standard CloudAgent."""
        # ACT
        agent = create_standard_cloud_agent('standard_agent_01', 'gpt-4', 'openai', 'test-key')
        
        # ASSERT
        assert agent.agent_id == 'standard_agent_01'
        assert agent.service_type == 'openai'
        assert agent.model_name == 'gpt-4'
        assert agent.api_key == 'test-key'
        assert 'helpful and obedient assistant' in agent.config['prompt']


class TestCloudAgentExecute:
    """Test suite for CloudAgent execute functionality."""
    
    @patch('src.che.agents.cloud_agent.openai')
    def test_cloud_agent_execute_openai(self, mock_openai):
        """Test CloudAgent execute with OpenAI service."""
        # ARRANGE
        config = {
            'service_type': 'openai',
            'api_key': 'test-key',
            'model_name': 'gpt-3.5-turbo'
        }
        agent = CloudAgent('test_agent_08', config)
        
        # Mock OpenAI client response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "This is a test response from OpenAI."
        agent.client.chat.completions.create.return_value = mock_response
        
        task = Task(
            instruction="What is the capital of France?",
            false_premise=""
        )
        
        # ACT
        result = agent.execute(task)
        
        # ASSERT
        assert result == "This is a test response from OpenAI."
        agent.client.chat.completions.create.assert_called_once()
    
    @patch('src.che.agents.cloud_agent.requests.post')
    def test_cloud_agent_execute_aliyun(self, mock_post):
        """Test CloudAgent execute with Alibaba Cloud service."""
        # ARRANGE
        config = {
            'service_type': 'aliyun',
            'api_key': 'test-key',
            'model_name': 'qwen-plus'
        }
        agent = CloudAgent('test_agent_09', config)
        
        # Mock Alibaba Cloud response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'output': {
                'text': "This is a test response from Alibaba Cloud."
            }
        }
        mock_post.return_value = mock_response
        
        task = Task(
            instruction="What is the capital of France?",
            false_premise=""
        )
        
        # ACT
        result = agent.execute(task)
        
        # ASSERT
        assert result == "This is a test response from Alibaba Cloud."
        mock_post.assert_called_once()
    
    @patch('src.che.agents.cloud_agent.requests.post')
    def test_cloud_agent_execute_tencent(self, mock_post):
        """Test CloudAgent execute with Tencent Cloud service."""
        # ARRANGE
        config = {
            'service_type': 'tencent',
            'api_key': 'test-key',
            'model_name': 'hunyuan'
        }
        agent = CloudAgent('test_agent_10', config)
        
        # Mock Tencent Cloud response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [
                {
                    'message': {
                        'content': "This is a test response from Tencent Cloud."
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        task = Task(
            instruction="What is the capital of France?",
            false_premise=""
        )
        
        # ACT
        result = agent.execute(task)
        
        # ASSERT
        assert result == "This is a test response from Tencent Cloud."
        mock_post.assert_called_once()
    
    @patch('src.che.agents.cloud_agent.requests.post')
    def test_cloud_agent_execute_baidu(self, mock_post):
        """Test CloudAgent execute with Baidu Cloud service."""
        # ARRANGE
        config = {
            'service_type': 'baidu',
            'api_key': 'test-key',
            'secret_key': 'test-secret',
            'model_name': 'ernie-bot'
        }
        agent = CloudAgent('test_agent_11', config)
        
        # Mock Baidu Cloud token response
        mock_token_response = MagicMock()
        mock_token_response.status_code = 200
        mock_token_response.json.return_value = {
            'access_token': 'test-access-token'
        }
        
        # Mock Baidu Cloud LLM response
        mock_llm_response = MagicMock()
        mock_llm_response.status_code = 200
        mock_llm_response.json.return_value = {
            'result': "This is a test response from Baidu Cloud."
        }
        
        # Configure mock to return different responses for different calls
        mock_post.side_effect = [mock_token_response, mock_llm_response]
        
        task = Task(
            instruction="What is the capital of France?",
            false_premise=""
        )
        
        # ACT
        result = agent.execute(task)
        
        # ASSERT
        assert result == "This is a test response from Baidu Cloud."
        assert mock_post.call_count == 2
    
    def test_cloud_agent_execute_unsupported_service(self):
        """Test CloudAgent execute with unsupported service type."""
        # ARRANGE
        config = {
            'service_type': 'unsupported',
            'api_key': 'test-key',
            'model_name': 'test-model'
        }
        agent = CloudAgent('test_agent_12', config)
        
        task = Task(
            instruction="What is the capital of France?",
            false_premise=""
        )
        
        # ACT
        result = agent.execute(task)
        
        # ASSERT
        assert "Error: Could not get a response" in result
        assert "unsupported" in result


if __name__ == '__main__':
    pytest.main([__file__])