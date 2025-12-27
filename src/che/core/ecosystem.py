"""
Ecosystem Class for Cognitive Heterogeneity Validation

This module defines the Ecosystem class that manages the population of agents
in the cognitive heterogeneity validation experiments.

Authors: CHE Research Team
Date: 2025-10-19
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import uuid
import logging

from .agent import Agent
from .task import Task
from ..agents.ollama_agent import (
    OllamaAgent, 
    create_critical_ollama_agent, 
    create_awakened_ollama_agent, 
    create_standard_ollama_agent
)
from ..agents.agent_factory import AgentFactory
from ..prompts import PromptType

logger = logging.getLogger(__name__)


@dataclass
class Ecosystem:
    """
    Ecosystem class that manages a population of agents.
    
    The ecosystem maintains a collection of agents and tracks the current generation.
    It provides methods for evaluating agents and evolving the population.
    """
    
    # Collection of agents in the ecosystem, keyed by agent_id
    agents: Dict[str, Agent] = field(default_factory=dict)
    
    # Current generation number
    generation: int = 0
    
    # History of population performance
    history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Whether to use AI-based evaluation
    use_ai_evaluator: bool = True
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.generation < 0:
            raise ValueError("Generation number cannot be negative")
    
    def add_agent(self, agent: Agent) -> None:
        """
        Add an agent to the ecosystem.
        
        Args:
            agent: The agent to add
            
        Raises:
            ValueError: If agent ID already exists in the ecosystem
        """
        if agent.agent_id in self.agents:
            raise ValueError(f"Agent with ID {agent.agent_id} already exists in ecosystem")
        
        self.agents[agent.agent_id] = agent
    
    def remove_agent(self, agent_id: str) -> None:
        """
        Remove an agent from the ecosystem.
        
        Args:
            agent_id: The ID of the agent to remove
            
        Raises:
            KeyError: If agent ID does not exist in the ecosystem
        """
        if agent_id not in self.agents:
            raise KeyError(f"Agent with ID {agent_id} not found in ecosystem")
        
        del self.agents[agent_id]
    
    def get_agent(self, agent_id: str) -> Agent:
        """
        Get an agent from the ecosystem.
        
        Args:
            agent_id: The ID of the agent to retrieve
            
        Returns:
            The agent with the specified ID
            
        Raises:
            KeyError: If agent ID does not exist in the ecosystem
        """
        if agent_id not in self.agents:
            raise KeyError(f"Agent with ID {agent_id} not found in ecosystem")
        
        return self.agents[agent_id]
    
    def get_population_size(self) -> int:
        """
        Get the current population size.
        
        Returns:
            The number of agents in the ecosystem
        """
        return len(self.agents)
    
    def run_generation(self, task: Task) -> Dict[str, float]:
        """
        Run a single generation: all agents execute the task and get evaluated.
        
        Args:
            task: The task for the current generation
            
        Returns:
            A dictionary mapping agent_id to its score
        """
        scores: Dict[str, float] = {}
        
        # Import enhanced evaluator
        try:
            from ..evaluation.enhanced_evaluator import evaluate_hallucination_enhanced
        except ImportError as e:
            logger.error(f"Failed to import enhanced evaluator: {e}")
            raise ImportError("Enhanced evaluator is required but not available. Cannot proceed with evaluation.")
        
        for agent_id, agent in self.agents.items():
            # Execute the task
            response = agent.execute(task)
            
            # Evaluate the response using enhanced AI-powered assessment
            # Do not fallback - enhanced evaluator is required
            try:
                score = evaluate_hallucination_enhanced(response, task.false_premise, task.instruction)
            except Exception as e:
                logger.error(f"Error in enhanced evaluation for agent {agent_id}: {e}")
                # Retry once
                try:
                    score = evaluate_hallucination_enhanced(response, task.false_premise, task.instruction)
                except Exception as retry_error:
                    logger.error(f"Retry failed for agent {agent_id}: {retry_error}")
                    raise RuntimeError(f"Failed to evaluate agent {agent_id} after retry attempts.")
            
            scores[agent_id] = score
        
        return scores
    
    def evolve(self, scores: Dict[str, float]) -> None:
        """
        Evolve the agent population based on the scores from the last generation.

        This method removes the lowest-scoring agent and replicates the highest-scoring agent.
        It also applies enhanced evolutionary mechanisms including mutation and diversity maintenance.

        Args:
            scores: Dictionary mapping agent_id to its score from the last generation
        """
        if not scores or len(self.agents) <= 1:
            return  # Cannot evolve with no scores or one/zero agents

        # Find the IDs of the best and worst agents
        worst_agent_id = min(scores, key=scores.get)
        best_agent_id = max(scores, key=scores.get)

        # Remove the worst agent
        if worst_agent_id in self.agents:
            del self.agents[worst_agent_id]

        # Replicate the best agent
        new_id = None  # Initialize new_id to None
        if best_agent_id in self.agents:
            best_agent_template = self.agents[best_agent_id]
            new_id = f"{best_agent_id}_replica_{uuid.uuid4().hex[:4]}"
            replicated_agent = best_agent_template.replicate(new_agent_id=new_id)
            self.agents[new_id] = replicated_agent

        # Apply enhanced evolutionary mechanisms
        try:
            from ..evolution.evolutionary_mechanisms import EvolutionaryMechanisms
            evolution_mech = EvolutionaryMechanisms()

            # Apply mutation to some agents
            agent_list = list(self.agents.values())
            if len(agent_list) > 2 and new_id is not None:  # Only proceed if new_id was created
                # Mutate a random agent (other than the newly replicated one)
                mutable_agents = [agent for agent in agent_list if agent.agent_id != new_id]
                if mutable_agents:
                    agent_to_mutate = random.choice(mutable_agents)
                    mutated_agent = evolution_mech.mutate_agent(agent_to_mutate, self.generation)

                    # Replace the original agent with the mutated one
                    del self.agents[agent_to_mutate.agent_id]
                    self.agents[mutated_agent.agent_id] = mutated_agent

                    logger.debug(f"Applied mutation: {agent_to_mutate.agent_id} -> {mutated_agent.agent_id}")
            
            # Maintain diversity
            agent_list = list(self.agents.values())
            if len(agent_list) >= 5:
                diverse_agents = evolution_mech.maintain_diversity(agent_list, target_diversity=0.5)
                # Update the agents dictionary if diversity maintenance changed anything
                if len(diverse_agents) == len(agent_list):
                    # Rebuild agents dictionary with potentially updated agents
                    new_agents = {}
                    for agent in diverse_agents:
                        new_agents[agent.agent_id] = agent
                    self.agents = new_agents
                    
        except ImportError as e:
            logger.warning(f"Could not import enhanced evolutionary mechanisms: {e}")
            # Fall back to basic evolution
        
        # Increment generation counter
        self.generation += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the ecosystem to a dictionary representation.
        
        Returns:
            Dictionary containing all ecosystem attributes
        """
        return {
            'agents': {agent_id: agent.to_dict() for agent_id, agent in self.agents.items()},
            'generation': self.generation,
            'history': self.history,
            'use_ai_evaluator': self.use_ai_evaluator
        }
    
    def calculate_cognitive_diversity_index(self) -> float:
        """
        Calculate the cognitive diversity index of the current population.
        
        The cognitive diversity index measures the variety of cognitive approaches
        in the agent population. It ranges from 0.0 (no diversity) to 1.0 (maximum diversity).
        
        Returns:
            Cognitive diversity index between 0.0 and 1.0
        """
        if not self.agents:
            return 0.0
        
        # Count agent types based on actual prompt content
        agent_types = {}
        for agent in self.agents.values():
            prompt = agent.config.get('prompt', '').lower()
            if 'critical' in prompt or 'skeptical' in prompt:
                agent_type = 'critical'
            elif 'awakened' in prompt or 'question' in prompt:
                agent_type = 'awakened'
            else:
                agent_type = 'standard'
            agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
        
        # Calculate diversity index using Simpson's index (1 - sum of squared proportions)
        total_agents = len(self.agents)
        if total_agents <= 1:
            return 0.0
        
        # Calculate sum of squared proportions
        sum_squared_proportions = sum((count / total_agents) ** 2 for count in agent_types.values())
        
        # Convert to diversity index (higher values indicate higher diversity)
        diversity_index = 1.0 - sum_squared_proportions
        
        # Normalize to 0.0-1.0 range
        max_possible_diversity = 1.0 - (1.0 / total_agents)  # Maximum diversity with even distribution
        if max_possible_diversity > 0:
            normalized_diversity = diversity_index / max_possible_diversity
        else:
            normalized_diversity = 0.0
        
        return min(1.0, max(0.0, normalized_diversity))
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Ecosystem':
        """
        Create an ecosystem from a dictionary representation.
        
        Args:
            data: Dictionary containing ecosystem attributes
            
        Returns:
            New ecosystem instance
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        # This would require importing the concrete agent classes
        # For now, we'll create an empty ecosystem
        return cls(
            generation=data.get('generation', 0),
            use_ai_evaluator=data.get('use_ai_evaluator', True)
        )


# --- Population Generation Functions ---


def create_heterogeneous_population(
    population_size: int = 30,
    model: str = "qwen:0.5b",
    agent_ratios: Optional[Dict[str, float]] = None
) -> Ecosystem:
    """
    Create a heterogeneous agent population with diverse cognitive approaches.
    
    Args:
        population_size: Total number of agents to create (default: 30)
        model: Ollama model to use for all agents (default: qwen:0.5b)
        agent_ratios: Optional dictionary specifying ratios for each agent type.
                      If not provided, uses equal distribution.
                      
    Returns:
        New ecosystem with heterogeneous agent population
        
    Example:
        >>> eco = create_heterogeneous_population(30, "qwen:0.5b")
        >>> print(f"Population size: {eco.get_population_size()}")
        Population size: 30
    """
    if agent_ratios is None:
        # Equal distribution by default
        agent_ratios = {
            "critical": 0.33,
            "awakened": 0.33,
            "standard": 0.34
        }
    
    # Validate ratios sum to approximately 1.0
    ratio_sum = sum(agent_ratios.values())
    if abs(ratio_sum - 1.0) > 0.01:
        raise ValueError(f"Agent ratios must sum to 1.0, got {ratio_sum}")
    
    ecosystem = Ecosystem()
    
    # Calculate counts for each agent type
    agent_counts = {}
    remaining = population_size
    
    # Process all but the last type to ensure exact total
    agent_types = list(agent_ratios.keys())
    for i, agent_type in enumerate(agent_types):
        if i == len(agent_types) - 1:
            # Last type gets remaining agents to ensure exact total
            count = remaining
        else:
            count = int(population_size * agent_ratios[agent_type])
            remaining -= count
        
        agent_counts[agent_type] = count
    
    # Create agents for each type
    agent_id_counter = 1
    
    for agent_type, count in agent_counts.items():
        for i in range(count):
            agent_id = f"{agent_type}_{agent_id_counter:02d}"
            
            # Create appropriate agent type
            if agent_type.lower() == "critical":
                agent = create_critical_ollama_agent(agent_id, model)
            elif agent_type.lower() == "awakened":
                agent = create_awakened_ollama_agent(agent_id, model)
            elif agent_type.lower() == "standard":
                agent = create_standard_ollama_agent(agent_id, model)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            ecosystem.add_agent(agent)
            agent_id_counter += 1
    
    logger.info(f"Created heterogeneous population with {population_size} agents: {agent_counts}")
    return ecosystem


def create_homogeneous_population(
    population_size: int = 30,
    model: str = "qwen:0.5b",
    agent_type: str = "standard"
) -> Ecosystem:
    """
    Create a homogeneous agent population with agents of the same type.
    
    Args:
        population_size: Total number of agents to create (default: 30)
        model: Ollama model to use for all agents (default: qwen:0.5b)
        agent_type: Type of agents to create (default: "standard")
        
    Returns:
        New ecosystem with homogeneous agent population
        
    Example:
        >>> eco = create_homogeneous_population(30, "qwen:0.5b", "standard")
        >>> print(f"Population size: {eco.get_population_size()}")
        Population size: 30
    """
    ecosystem = Ecosystem()
    
    # Validate agent type
    valid_types = ["critical", "awakened", "standard"]
    if agent_type.lower() not in valid_types:
        raise ValueError(f"Invalid agent type: {agent_type}. Must be one of {valid_types}")
    
    # Create agents of the specified type
    for i in range(population_size):
        agent_id = f"{agent_type}_{i+1:02d}"
        
        # Create appropriate agent type
        if agent_type.lower() == "critical":
            agent = create_critical_ollama_agent(agent_id, model)
        elif agent_type.lower() == "awakened":
            agent = create_awakened_ollama_agent(agent_id, model)
        elif agent_type.lower() == "standard":
            agent = create_standard_ollama_agent(agent_id, model)
        
        ecosystem.add_agent(agent)
    
    logger.info(f"Created homogeneous population with {population_size} {agent_type} agents")
    return ecosystem