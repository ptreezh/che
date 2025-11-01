"""
Enhanced Ecosystem with True Evolutionary Mechanisms

This module extends the basic Ecosystem class with enhanced evolutionary mechanisms
including mutation, crossover, and diversity maintenance.

Authors: CHE Research Team
Date: 2025-10-31
"""

import random
import copy
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import uuid

from ..core.agent import Agent
from ..core.task import Task
from ..agents.ollama_agent import OllamaAgent
from ..prompts import PromptType, get_prompt
from ..evolution.enhanced_evolution import MutationEngine, CrossoverEngine, DiversityMonitor

logger = logging.getLogger(__name__)


@dataclass
class EnhancedEcosystem:
    """
    Enhanced ecosystem with true evolutionary mechanisms.
    
    The ecosystem maintains a population of agents and implements enhanced evolutionary
    mechanisms including mutation, crossover, and diversity maintenance.
    """
    
    # Collection of agents in the ecosystem, keyed by agent_id
    agents: Dict[str, Agent] = field(default_factory=dict)
    
    # Current generation number
    generation: int = 0
    
    # History of population performance
    history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Whether to use AI-based evaluation
    use_ai_evaluator: bool = True
    
    # Evolutionary mechanisms
    mutation_engine: Optional[MutationEngine] = None
    crossover_engine: Optional[CrossoverEngine] = None
    diversity_monitor: Optional[DiversityMonitor] = None
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.generation < 0:
            raise ValueError("Generation number cannot be negative")
        
        # Initialize evolutionary mechanisms if not provided
        if self.mutation_engine is None:
            self.mutation_engine = MutationEngine(mutation_rate=0.1, mutation_intensity=0.2)
        
        if self.crossover_engine is None:
            self.crossover_engine = CrossoverEngine(crossover_rate=0.2)
        
        if self.diversity_monitor is None:
            self.diversity_monitor = DiversityMonitor(diversity_threshold=0.3)
    
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
        
        This method implements enhanced evolutionary mechanisms including:
        1. Selection (remove worst agent, replicate best agent)
        2. Mutation (apply mutations to some agents)
        3. Crossover (combine characteristics of parent agents)
        4. Diversity maintenance (ensure cognitive diversity)
        
        Args:
            scores: Dictionary mapping agent_id to its score from the last generation
        """
        if not scores or len(self.agents) <= 1:
            return  # Cannot evolve with no scores or one/zero agents
        
        # Record scores in history
        self.history.append({
            'generation': self.generation,
            'scores': scores.copy()
        })
        
        # Find the IDs of the best and worst agents
        worst_agent_id = min(scores, key=scores.get)
        best_agent_id = max(scores, key=scores.get)
        
        # Remove the worst agent
        if worst_agent_id in self.agents:
            logger.debug(f"Removing worst agent: {worst_agent_id}")
            del self.agents[worst_agent_id]
        
        # Get performance history for the best agent
        best_agent_performance_history = [entry.get('scores', {}).get(best_agent_id, 0) 
                                          for entry in self.history 
                                          if 'scores' in entry and best_agent_id in entry.get('scores', {})]
        
        # Replicate the best agent with mutation
        if best_agent_id in self.agents:
            best_agent_template = self.agents[best_agent_id]
            new_id = f"{best_agent_id}_replica_{uuid.uuid4().hex[:4]}"
            
            # Apply mutation to replicated agent with performance history
            replicated_agent = self.mutation_engine.mutate_agent(best_agent_template, new_id, best_agent_performance_history)
            self.agents[new_id] = replicated_agent
            logger.debug(f"Replicated and mutated best agent: {best_agent_id} -> {new_id}")
        
        # Apply additional mutations to maintain genetic diversity
        if len(self.agents) > 2:
            # Select a random agent (excluding the newly replicated one) for additional mutation
            agent_ids = [agent_id for agent_id in self.agents.keys() if agent_id != new_id]
            if agent_ids:
                agent_to_mutate_id = random.choice(agent_ids)
                agent_to_mutate = self.agents[agent_to_mutate_id]
                
                # Get performance history for this agent
                agent_performance_history = [entry.get('scores', {}).get(agent_to_mutate_id, 0) 
                                             for entry in self.history 
                                             if 'scores' in entry and agent_to_mutate_id in entry.get('scores', {})]
                
                # Apply mutation with performance history
                mutated_id = f"{agent_to_mutate_id}_mutated_{uuid.uuid4().hex[:4]}"
                mutated_agent = self.mutation_engine.mutate_agent(agent_to_mutate, mutated_id, agent_performance_history)
                
                # Replace the original agent with the mutated one
                del self.agents[agent_to_mutate_id]
                self.agents[mutated_id] = mutated_agent
                logger.debug(f"Applied additional mutation: {agent_to_mutate_id} -> {mutated_id}")
        
        # Apply crossover if population is large enough
        if len(self.agents) >= 4:
            # Select two random parent agents for crossover
            agent_ids = list(self.agents.keys())
            if len(agent_ids) >= 2:
                parent1_id, parent2_id = random.sample(agent_ids, 2)
                parent1 = self.agents[parent1_id]
                parent2 = self.agents[parent2_id]
                
                # Create child agent through crossover
                child_id = f"child_{parent1_id}_{parent2_id}_{uuid.uuid4().hex[:4]}"
                child_agent = self.crossover_engine.crossover_agents(parent1, parent2, child_id)
                
                # Add child to population (may replace a random existing agent to maintain population size)
                if len(self.agents) >= 30:  # Assuming 30 is target population size
                    # Remove a random existing agent to make room for child
                    agent_to_remove = random.choice(list(self.agents.keys()))
                    del self.agents[agent_to_remove]
                    logger.debug(f"Removed agent to maintain population size: {agent_to_remove}")
                
                self.agents[child_id] = child_agent
                logger.debug(f"Applied crossover: {parent1_id} + {parent2_id} -> {child_id}")
        
        # Monitor and maintain diversity
        if self.diversity_monitor.needs_diversity_intervention(self.agents):
            logger.warning("Low diversity detected, applying intervention")
            intervention_suggestions = self.diversity_monitor.suggest_diversity_intervention(self.agents)
            
            # Implement simple diversity intervention: introduce new agents of underrepresented types
            for suggestion in intervention_suggestions.get('interventions', []):
                agent_type = suggestion['type']
                needed_count = suggestion['needed_count']
                
                # Create new agents of the underrepresented type
                for i in range(min(needed_count, 2)):  # Limit to 2 new agents per intervention
                    new_agent_id = f"{agent_type}_new_{uuid.uuid4().hex[:4]}"
                    
                    # Create appropriate agent type
                    if agent_type == 'critical':
                        new_agent = OllamaAgent(
                            agent_id=new_agent_id,
                            config={
                                "model": "gemma3:latest",
                                "prompt": get_prompt(PromptType.CRITICAL)
                            },
                            generation=self.generation + 1,
                            is_variant=True,
                            original_source="diversity_intervention"
                        )
                    elif agent_type == 'awakened':
                        new_agent = OllamaAgent(
                            agent_id=new_agent_id,
                            config={
                                "model": "gemma3:latest",
                                "prompt": get_prompt(PromptType.AWAKENED)
                            },
                            generation=self.generation + 1,
                            is_variant=True,
                            original_source="diversity_intervention"
                        )
                    else:  # standard
                        new_agent = OllamaAgent(
                            agent_id=new_agent_id,
                            config={
                                "model": "gemma3:latest",
                                "prompt": get_prompt(PromptType.STANDARD)
                            },
                            generation=self.generation + 1,
                            is_variant=True,
                            original_source="diversity_intervention"
                        )
                    
                    self.agents[new_agent_id] = new_agent
                    logger.debug(f"Added new {agent_type} agent for diversity: {new_agent_id}")
        
        # Increment generation counter
        self.generation += 1
        logger.info(f"Evolved to generation {self.generation}")
    
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
        return self.diversity_monitor.calculate_diversity(self.agents)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedEcosystem':
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


# Exported functions
__all__ = [
    'EnhancedEcosystem'
]