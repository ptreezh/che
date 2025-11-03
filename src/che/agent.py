from abc import ABC, abstractmethod
from typing import Dict

from .task import Task

class Agent(ABC):
    """Abstract base class for all agents in the ecosystem."""

    def __init__(self, agent_id: str, config: Dict):
        """
        Initializes the agent.
        - agent_id: A unique identifier for the agent.
        - config: A dictionary for configuration, e.g., holding a system_prompt.
        """
        self.agent_id = agent_id
        self.config = config

    @abstractmethod
    def execute(self, task: Task) -> str:
        """
        Executes the given task and returns a string output.
        This method must be implemented by concrete subclasses.
        """
        pass

    def replicate(self, new_agent_id: str) -> 'Agent':
        """
        Creates a new instance of the agent with a new ID but the same config.
        """
        return type(self)(agent_id=new_agent_id, config=self.config)
