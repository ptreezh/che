"""
Agent Abstract Base Class for Cognitive Heterogeneity Validation

This module defines the abstract base class for all agents in the ecosystem.
Each agent type (critical, awakened, standard) must implement the abstract methods.

Authors: CHE Research Team
Date: 2025-10-19
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Agent(ABC):
    """
    Abstract base class for all agents in the cognitive heterogeneity ecosystem.
    
    This class defines the common interface that all agent implementations must follow.
    Each agent has a unique ID, configuration, and can execute tasks.
    """
    
    # Unique identifier for the agent
    agent_id: str
    
    # Configuration dictionary containing model and prompt information
    config: Dict[str, Any]
    
    # Generation number in the evolutionary process
    generation: int = 0
    
    # Whether this agent is a variant (mutated from another agent)
    is_variant: bool = False
    
    # ID of the original agent this variant was mutated from (if applicable)
    original_source: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if not self.agent_id:
            raise ValueError("Agent ID cannot be empty")
        
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")
    
    @abstractmethod
    def execute(self, task: 'Task') -> str:
        """
        Execute a task and return a response.
        
        This method must be implemented by all concrete agent classes.
        
        Args:
            task: The task to execute
            
        Returns:
            The agent's response as a string
            
        Raises:
            Implementation-specific exceptions for execution failures
        """
        pass
    
    @abstractmethod
    def replicate(self, new_agent_id: str) -> 'Agent':
        """
        Create a copy of this agent with a new ID.
        
        This method must be implemented by all concrete agent classes.
        The replicated agent should have the same configuration but a different ID.
        
        Args:
            new_agent_id: The ID for the new agent
            
        Returns:
            A new agent instance with the same configuration but different ID
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the agent to a dictionary representation.
        
        Returns:
            Dictionary containing all agent attributes
        """
        return {
            'agent_id': self.agent_id,
            'config': self.config,
            'generation': self.generation,
            'is_variant': self.is_variant,
            'original_source': self.original_source
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Agent':
        """
        Create an agent from a dictionary representation.
        
        Args:
            data: Dictionary containing agent attributes
            
        Returns:
            New agent instance
        """
        # This is a factory method that concrete classes should override
        raise NotImplementedError("from_dict must be implemented by concrete agent classes")