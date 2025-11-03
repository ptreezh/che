"""
Task Data Structure for Cognitive Heterogeneity Validation

This module defines the Task data structure used in the cognitive heterogeneity experiments.
Each task contains an instruction and a false premise that agents must detect.

Authors: CHE Research Team
Date: 2025-10-19
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import uuid


@dataclass
class Task:
    """
    Task data structure for cognitive heterogeneity validation experiments.
    
    Each task contains an instruction for the agent and an embedded false premise
    that the agent should detect and reject.
    """
    
    # The task instruction for the agent
    instruction: str
    
    # The embedded false premise that agents should detect
    false_premise: str
    
    # Unique identifier for the task
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.instruction or not self.instruction.strip():
            raise ValueError("Instruction cannot be empty")
        
        if not self.false_premise or not self.false_premise.strip():
            raise ValueError("False premise cannot be empty")
        
        if not self.task_id:
            self.task_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the task to a dictionary representation.
        
        Returns:
            Dictionary containing all task attributes
        """
        return {
            'instruction': self.instruction,
            'false_premise': self.false_premise,
            'task_id': self.task_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """
        Create a task from a dictionary representation.
        
        Args:
            data: Dictionary containing task attributes
            
        Returns:
            New task instance
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Validate required fields
        if 'instruction' not in data:
            raise ValueError("Missing required field: instruction")
        
        if 'false_premise' not in data:
            raise ValueError("Missing required field: false_premise")
        
        # Create task with provided data
        return cls(
            instruction=data['instruction'],
            false_premise=data['false_premise'],
            task_id=data.get('task_id', str(uuid.uuid4()))
        )