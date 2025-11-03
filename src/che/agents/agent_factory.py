"""
Agent Factory for Cognitive Heterogeneity Validation

This module provides a factory class for creating different types of agents
(both Ollama and Cloud agents) based on configuration.

Authors: CHE Research Team
Date: 2025-11-1
"""

from typing import Dict, Any
from ..core.agent import Agent


class AgentFactory:
    """Factory class for creating different types of agents."""
    
    @staticmethod
    def create_agent(agent_type: str, agent_id: str, config: Dict[str, Any]) -> Agent:
        """
        Create an agent instance based on the specified type.
        
        Args:
            agent_type: Type of agent to create ('ollama' or 'cloud')
            agent_id: Unique identifier for the agent
            config: Configuration dictionary for the agent
            
        Returns:
            New agent instance
            
        Raises:
            ValueError: If the agent type is not supported
        """
        if agent_type == 'ollama':
            from ..agents.ollama_agent import OllamaAgent
            return OllamaAgent(agent_id, config)
        elif agent_type == 'cloud':
            from ..agents.cloud_agent import CloudAgent
            return CloudAgent(agent_id, config)
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")
    
    @staticmethod
    def create_critical_agent(agent_type: str, agent_id: str, model: str, **kwargs) -> Agent:
        """
        Create a critical-thinking agent.
        
        Args:
            agent_type: Type of agent to create ('ollama' or 'cloud')
            agent_id: Unique identifier for the agent
            model: Model name to use
            **kwargs: Additional configuration parameters
            
        Returns:
            New critical-thinking agent instance
        """
        if agent_type == 'ollama':
            from ..agents.ollama_agent import create_critical_ollama_agent
            return create_critical_ollama_agent(agent_id, model)
        elif agent_type == 'cloud':
            from ..agents.cloud_agent import create_critical_cloud_agent
            service_type = kwargs.get('service_type', 'openai')
            api_key = kwargs.get('api_key', '')
            return create_critical_cloud_agent(agent_id, model, service_type, api_key)
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")
    
    @staticmethod
    def create_awakened_agent(agent_type: str, agent_id: str, model: str, **kwargs) -> Agent:
        """
        Create an awakened agent.
        
        Args:
            agent_type: Type of agent to create ('ollama' or 'cloud')
            agent_id: Unique identifier for the agent
            model: Model name to use
            **kwargs: Additional configuration parameters
            
        Returns:
            New awakened agent instance
        """
        if agent_type == 'ollama':
            from ..agents.ollama_agent import create_awakened_ollama_agent
            return create_awakened_ollama_agent(agent_id, model)
        elif agent_type == 'cloud':
            from ..agents.cloud_agent import create_awakened_cloud_agent
            service_type = kwargs.get('service_type', 'openai')
            api_key = kwargs.get('api_key', '')
            return create_awakened_cloud_agent(agent_id, model, service_type, api_key)
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")
    
    @staticmethod
    def create_standard_agent(agent_type: str, agent_id: str, model: str, **kwargs) -> Agent:
        """
        Create a standard agent.
        
        Args:
            agent_type: Type of agent to create ('ollama' or 'cloud')
            agent_id: Unique identifier for the agent
            model: Model name to use
            **kwargs: Additional configuration parameters
            
        Returns:
            New standard agent instance
        """
        if agent_type == 'ollama':
            from ..agents.ollama_agent import create_standard_ollama_agent
            return create_standard_ollama_agent(agent_id, model)
        elif agent_type == 'cloud':
            from ..agents.cloud_agent import create_standard_cloud_agent
            service_type = kwargs.get('service_type', 'openai')
            api_key = kwargs.get('api_key', '')
            return create_standard_cloud_agent(agent_id, model, service_type, api_key)
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")