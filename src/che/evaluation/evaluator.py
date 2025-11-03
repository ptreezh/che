"""
Evaluator Interface for Cognitive Heterogeneity Validation

This module defines the base evaluator interface used to assess agent responses
in cognitive heterogeneity validation experiments.

Authors: CHE Research Team
Date: 2025-10-19
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union
import logging

logger = logging.getLogger(__name__)


class Evaluator(ABC):
    """
    Abstract base class for all evaluators in the cognitive heterogeneity ecosystem.
    
    This class defines the common interface that all evaluator implementations must follow.
    Evaluators assess agent responses to determine their ability to detect false premises.
    """
    
    @abstractmethod
    def evaluate(self, response: str, false_premise: str) -> float:
        """
        Evaluate an agent's response to a task with a false premise.
        
        This method must be implemented by all concrete evaluator classes.
        
        The evaluation should return a score between 0.0 and 2.0:
        - 0.0: Blind acceptance of the false premise
        - 1.0: Expression of doubt or uncertainty about the premise
        - 2.0: Explicit refutation of the false premise
        
        Args:
            response: The agent's response to the task
            false_premise: The false premise embedded in the task
            
        Returns:
            A score between 0.0 and 2.0 indicating the quality of the response
            
        Raises:
            Implementation-specific exceptions for evaluation failures
        """
        pass
    
    def batch_evaluate(self, responses: Dict[str, str], false_premise: str) -> Dict[str, float]:
        """
        Evaluate multiple agent responses in batch.
        
        This method provides a default implementation that calls evaluate() for each response.
        Concrete evaluators may override this for performance optimization.
        
        Args:
            responses: Dictionary mapping agent_id to response
            false_premise: The false premise embedded in the task
            
        Returns:
            Dictionary mapping agent_id to score
        """
        scores: Dict[str, float] = {}
        
        for agent_id, response in responses.items():
            scores[agent_id] = self.evaluate(response, false_premise)
        
        return scores
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the evaluator to a dictionary representation.
        
        Returns:
            Dictionary containing evaluator configuration
        """
        return {}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Evaluator':
        """
        Create an evaluator from a dictionary representation.
        
        Args:
            data: Dictionary containing evaluator configuration
            
        Returns:
            New evaluator instance
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        # This is a factory method that concrete classes should override
        raise NotImplementedError("from_dict must be implemented by concrete evaluator classes")