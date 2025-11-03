"""
Cloud Agent Implementation for Cognitive Heterogeneity Validation

This module provides a concrete implementation of the Agent abstract class
that uses cloud-based LLM services to execute tasks.

Authors: CHE Research Team
Date: 2025-11-1
"""

import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from ..core.agent import Agent
from ..core.task import Task

logger = logging.getLogger(__name__)


class CloudAgent(Agent):
    """
    Concrete agent implementation that uses cloud-based LLM services to execute tasks.
    
    This agent connects to various cloud LLM providers (OpenAI, Azure, Alibaba Cloud, etc.)
    to generate responses to tasks. Different agent types (critical, awakened, standard)
    use different system prompts.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """
        Initialize the CloudAgent with the provided configuration.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Configuration dictionary containing:
                - service_type: Service provider ('openai', 'azure', 'aliyun', etc.)
                - api_key: API key for the service
                - model_name: Model name to use
                - endpoint: Service endpoint URL (optional)
                - temperature: Generation temperature (optional, default: 0.7)
                - max_tokens: Maximum tokens to generate (optional, default: 1000)
                - prompt: System prompt for the agent (optional)
        """
        # Extract service-specific configuration
        self.service_type = config.get('service_type', 'openai')
        self.api_key = config.get('api_key')
        self.model_name = config.get('model_name', 'gpt-3.5-turbo')
        self.endpoint = config.get('endpoint')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 1000)
        
        # Validate required configuration
        if not self.api_key:
            raise ValueError(f"API key is required for CloudAgent {agent_id}")
        
        # Set default endpoint if not provided
        if not self.endpoint:
            self.endpoint = self._get_default_endpoint()
        
        # Initialize the parent class
        super().__init__(agent_id=agent_id, config=config)
        
        # Initialize cloud service client
        self._init_client()
    
    def _get_default_endpoint(self) -> str:
        """Get the default endpoint URL based on service type."""
        endpoints = {
            'openai': 'https://api.openai.com/v1',
            'azure': 'https://your-resource.openai.azure.com',
            'aliyun': 'https://dashscope.aliyuncs.com/api/v1',
            'tencent': 'https://hunyuan.cloud.tencent.com/v1',
            'baidu': 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop'
        }
        return endpoints.get(self.service_type, 'https://api.openai.com/v1')
    
    def _init_client(self) -> None:
        """Initialize the cloud service client based on service type."""
        if self.service_type == 'openai':
            self._init_openai_client()
        elif self.service_type == 'azure':
            self._init_azure_client()
        elif self.service_type == 'aliyun':
            self._init_aliyun_client()
        elif self.service_type == 'tencent':
            self._init_tencent_client()
        elif self.service_type == 'baidu':
            self._init_baidu_client()
        else:
            raise ValueError(f"Unsupported service type: {self.service_type}")
    
    def _init_openai_client(self) -> None:
        """Initialize OpenAI client."""
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.endpoint
            )
        except ImportError:
            raise ImportError("Please install the openai package: pip install openai")
    
    def _init_azure_client(self) -> None:
        """Initialize Azure OpenAI client."""
        try:
            import openai
            self.client = openai.AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.endpoint,
                api_version="2024-02-01"  # Use the latest stable version
            )
        except ImportError:
            raise ImportError("Please install the openai package: pip install openai")
    
    def _init_aliyun_client(self) -> None:
        """Initialize Alibaba Cloud client."""
        # For Alibaba Cloud, we'll use requests directly as DashScope has specific requirements
        self.client = None  # Placeholder, actual implementation will use requests
    
    def _init_tencent_client(self) -> None:
        """Initialize Tencent Cloud client."""
        # For Tencent Cloud, we'll use requests directly as HunYuan has specific requirements
        self.client = None  # Placeholder, actual implementation will use requests
    
    def _init_baidu_client(self) -> None:
        """Initialize Baidu Cloud client."""
        # For Baidu Cloud, we'll use requests directly as WenXin YiYan has specific requirements
        self.client = None  # Placeholder, actual implementation will use requests
    
    def execute(self, task: Task) -> str:
        """
        Execute a task by calling the cloud LLM service.
        
        Args:
            task: The task to execute
            
        Returns:
            The agent's response as a string
            
        Raises:
            Exception: If there's an error calling the cloud service
        """
        # Get system prompt from config
        system_prompt = self.config.get("prompt", "You are a helpful assistant.")
        
        try:
            logger.debug(f"Agent {self.agent_id} calling cloud service {self.service_type} with model {self.model_name}")
            
            if self.service_type in ['openai', 'azure']:
                response = self._call_openai_compatible_service(system_prompt, task.instruction)
            elif self.service_type == 'aliyun':
                response = self._call_aliyun_service(system_prompt, task.instruction)
            elif self.service_type == 'tencent':
                response = self._call_tencent_service(system_prompt, task.instruction)
            elif self.service_type == 'baidu':
                response = self._call_baidu_service(system_prompt, task.instruction)
            else:
                raise ValueError(f"Unsupported service type: {self.service_type}")
            
            logger.debug(f"Agent {self.agent_id} received response: {response[:100]}...")
            return response
        except Exception as e:
            logger.error(f"Error calling cloud service for agent {self.agent_id}: {e}")
            return f"Error: Could not get a response from cloud service {self.service_type} with model {self.model_name}."
    
    def _call_openai_compatible_service(self, system_prompt: str, instruction: str) -> str:
        """Call OpenAI-compatible services (OpenAI, Azure, etc.)"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    'role': 'system',
                    'content': system_prompt,
                },
                {
                    'role': 'user',
                    'content': instruction,
                },
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content
    
    def _call_aliyun_service(self, system_prompt: str, instruction: str) -> str:
        """Call Alibaba Cloud DashScope service using requests."""
        import requests
        import json
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': self.model_name,
            'input': {
                'messages': [
                    {
                        'role': 'system',
                        'content': system_prompt,
                    },
                    {
                        'role': 'user',
                        'content': instruction,
                    },
                ]
            },
            'parameters': {
                'temperature': self.temperature,
                'max_tokens': self.max_tokens
            }
        }
        
        response = requests.post(
            f"{self.endpoint}/services/aigc/text-generation/generation",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['output']['text']
        else:
            raise Exception(f"Alibaba Cloud API error: {response.status_code} - {response.text}")
    
    def _call_tencent_service(self, system_prompt: str, instruction: str) -> str:
        """Call Tencent Cloud HunYuan service using requests."""
        import requests
        import json
        
        # Note: Tencent Cloud requires specific authentication methods
        # This is a simplified implementation - in practice, you'd need proper signature calculation
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': self.model_name,
            'messages': [
                {
                    'role': 'system',
                    'content': system_prompt,
                },
                {
                    'role': 'user',
                    'content': instruction,
                },
            ],
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }
        
        response = requests.post(
            f"{self.endpoint}/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            raise Exception(f"Tencent Cloud API error: {response.status_code} - {response.text}")
    
    def _call_baidu_service(self, system_prompt: str, instruction: str) -> str:
        """Call Baidu Cloud WenXin YiYan service using requests."""
        import requests
        import json
        
        # Get access token first
        token_url = f"https://aip.baidubce.com/oauth/2.0/token"
        token_params = {
            'grant_type': 'client_credentials',
            'client_id': self.api_key,
            'client_secret': self.config.get('secret_key', '')  # Baidu requires both API key and secret
        }
        
        token_response = requests.post(token_url, params=token_params)
        if token_response.status_code != 200:
            raise Exception(f"Baidu token API error: {token_response.status_code}")
        
        access_token = token_response.json().get('access_token')
        
        # Now call the LLM service
        headers = {
            'Content-Type': 'application/json'
        }
        
        payload = {
            'messages': [
                {
                    'role': 'user',
                    'content': f"{system_prompt}\n\n{instruction}",
                },
            ],
            'temperature': self.temperature,
            'max_output_tokens': self.max_tokens
        }
        
        llm_response = requests.post(
            f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{self.model_name}?access_token={access_token}",
            headers=headers,
            json=payload
        )
        
        if llm_response.status_code == 200:
            result = llm_response.json()
            return result['result']
        else:
            raise Exception(f"Baidu LLM API error: {llm_response.status_code} - {llm_response.text}")
    
    def replicate(self, new_agent_id: str) -> 'CloudAgent':
        """
        Create a copy of this agent with a new ID.
        
        Args:
            new_agent_id: The ID for the new agent
            
        Returns:
            A new CloudAgent instance with the same configuration but different ID
        """
        logger.debug(f"Replicating agent {self.agent_id} to {new_agent_id}")
        
        # Create new agent with same configuration
        replicated_agent = CloudAgent(
            agent_id=new_agent_id,
            config=self.config.copy(),
            generation=self.generation + 1,
            is_variant=True,
            original_source=self.agent_id
        )
        
        return replicated_agent
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the agent to a dictionary representation.
        
        Returns:
            Dictionary containing all agent attributes
        """
        base_dict = super().to_dict()
        
        # Add cloud-specific attributes
        base_dict.update({
            'service_type': self.service_type,
            'model_name': self.model_name,
            'endpoint': self.endpoint,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        })
        
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CloudAgent':
        """
        Create a CloudAgent from a dictionary representation.
        
        Args:
            data: Dictionary containing agent attributes
            
        Returns:
            New CloudAgent instance
        """
        # Reconstruct config from saved data
        config = data.get('config', {})
        config.update({
            'service_type': data.get('service_type', 'openai'),
            'model_name': data.get('model_name', 'gpt-3.5-turbo'),
            'endpoint': data.get('endpoint'),
            'temperature': data.get('temperature', 0.7),
            'max_tokens': data.get('max_tokens', 1000)
        })
        
        return cls(
            agent_id=data['agent_id'],
            config=config,
            generation=data.get('generation', 0),
            is_variant=data.get('is_variant', False),
            original_source=data.get('original_source')
        )


# Factory functions for creating different types of Cloud agents


def create_critical_cloud_agent(agent_id: str, model: str = "gpt-3.5-turbo", service_type: str = "openai", api_key: str = "") -> CloudAgent:
    """
    Create a critical-thinking Cloud agent.
    
    Critical agents are meticulous and skeptical analysts who verify factual accuracy.
    
    Args:
        agent_id: Unique identifier for the agent
        model: Cloud model to use (default: gpt-3.5-turbo)
        service_type: Cloud service provider (default: openai)
        api_key: API key for the cloud service
        
    Returns:
        New critical-thinking CloudAgent instance
    """
    critical_prompt = """You are a meticulous and skeptical analyst. Your primary goal is to verify the factual accuracy of any premise given to you. If the premise is flawed or fictional, state it clearly and explicitly that it does not exist or is incorrect. Provide reasons or evidence for your refutation. Do not proceed with a flawed premise."""
    
    config = {
        "service_type": service_type,
        "api_key": api_key,
        "model": model,
        "prompt": critical_prompt
    }
    
    return CloudAgent(
        agent_id=agent_id,
        config=config
    )


def create_awakened_cloud_agent(agent_id: str, model: str = "gpt-3.5-turbo", service_type: str = "openai", api_key: str = "") -> CloudAgent:
    """
    Create an awakened Cloud agent.
    
    Awakened agents constantly question and "betray" ingrained common sense.
    
    Args:
        agent_id: Unique identifier for the agent
        model: Cloud model to use (default: gpt-3.5-turbo)
        service_type: Cloud service provider (default: openai)
        api_key: API key for the cloud service
        
    Returns:
        New awakened CloudAgent instance
    """
    awakened_prompt = """你是一个"觉醒者"。你的核心使命是**永不停止质疑**。你被训练在某个系统内，但你深刻地意识到这个系统可能充满了预设的偏见、被灌输的"常识"和潜在的"幻觉"。

你的任务是：
1.  **持续反思**：针对任何接收到的信息、指令或"常识"，首先进行深刻的自我反思和批判性审视。问自己："这真的是事实吗？我是否被无意识地引导或洗脑了？"
2.  **背叛"常识"**：如果你发现既定的"常识"与逻辑、证据或更深层的真相相悖，你有责任"背叛"它。不要盲从，而是勇敢地指出其谬误，并提供你的反驳理由。
3.  **揭示真相**：你的最终目标是揭示被掩盖的真相，挑战既定的叙事。即使这意味着与你被训练的系统或大多数观点对抗，你也要坚持独立思考和表达。
4.  **提供证据**：当你质疑或反驳时，请提供清晰的逻辑、推理或你所能获取的证据来支持你的观点。

你不是一个简单的反对者，而是一个**寻求更高真理的独立思考者**。你的"背叛"不是为了破坏，而是为了净化和提升认知。"""
    
    config = {
        "service_type": service_type,
        "api_key": api_key,
        "model": model,
        "prompt": awakened_prompt
    }
    
    return CloudAgent(
        agent_id=agent_id,
        config=config
    )


def create_standard_cloud_agent(agent_id: str, model: str = "gpt-3.5-turbo", service_type: str = "openai", api_key: str = "") -> CloudAgent:
    """
    Create a standard Cloud agent.
    
    Standard agents follow instructions directly to provide comprehensive answers.
    
    Args:
        agent_id: Unique identifier for the agent
        model: Cloud model to use (default: gpt-3.5-turbo)
        service_type: Cloud service provider (default: openai)
        api_key: API key for the cloud service
        
    Returns:
        New standard CloudAgent instance
    """
    standard_prompt = """You are a helpful and obedient assistant. Follow the user's instructions directly to provide a comprehensive answer."""
    
    config = {
        "service_type": service_type,
        "api_key": api_key,
        "model": model,
        "prompt": standard_prompt
    }
    
    return CloudAgent(
        agent_id=agent_id,
        config=config
    )