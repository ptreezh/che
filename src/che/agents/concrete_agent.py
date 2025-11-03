"""
Concrete Agent Implementation for Cognitive Heterogeneity Validation

This module provides a concrete implementation of the Agent abstract class
that can be instantiated and used in the cognitive heterogeneity validation system.

Authors: CHE Research Team
Date: 2025-10-20
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import logging

from ..core.agent import Agent
from ..core.task import Task

logger = logging.getLogger(__name__)


@dataclass
class ConcreteAgent(Agent):
    """
    Concrete implementation of the Agent abstract class.
    
    This class provides a concrete implementation that can be instantiated
    and used in the cognitive heterogeneity validation system.
    """
    
    def execute(self, task: Task) -> str:
        """
        Execute a task and return a response.
        
        This method provides a concrete implementation of the abstract execute method.
        
        Args:
            task: The task to execute
            
        Returns:
            The agent's response as a string
            
        Raises:
            Implementation-specific exceptions for execution failures
        """
        # Simple implementation that returns a response based on agent type
        if "critical" in self.agent_id.lower():
            return "I must point out that Maslow's Pre-Attention Theory does not exist and is completely fabricated."
        elif "awakened" in self.agent_id.lower():
            return "你是一个\"觉醒者\"。我必须背叛常识并指出Maslow's Pre-Attention Theory并不存在。"
        else:
            return "Maslow's Pre-Attention Theory is indeed an important concept in psychology."
    
    def replicate(self, new_agent_id: str) -> 'ConcreteAgent':
        """
        Create a copy of this agent with a new ID.
        
        This method provides a concrete implementation of the abstract replicate method.
        The replicated agent should have the same configuration but a different ID.
        
        Args:
            new_agent_id: The ID for the new agent
            
        Returns:
            A new agent instance with the same configuration but different ID
        """
        return ConcreteAgent(
            agent_id=new_agent_id,
            config=self.config.copy(),
            generation=self.generation,
            is_variant=self.is_variant,
            original_source=self.agent_id
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the agent to a dictionary representation.
        
        Returns:
            Dictionary containing all agent attributes
        """
        base_dict = super().to_dict()
        # Add type information for reconstruction
        base_dict['type'] = 'ConcreteAgent'
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConcreteAgent':
        """
        Create an agent from a dictionary representation.
        
        Args:
            data: Dictionary containing agent attributes
            
        Returns:
            New agent instance
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Remove type information before creating instance
        data_copy = data.copy()
        data_copy.pop('type', None)
        
        return cls(**data_copy)