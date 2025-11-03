"""
Enhanced Evolutionary Mechanisms for Cognitive Heterogeneity Validation

This module provides enhanced evolutionary mechanisms including mutation, crossover, 
and diversity maintenance for cognitive heterogeneity experiments.

Authors: CHE Research Team
Date: 2025-10-31
"""

import numpy as np
import random
import copy
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from ..core.agent import Agent
from ..agents.ollama_agent import OllamaAgent
from ..prompts import PromptType, get_prompt

logger = logging.getLogger(__name__)


@dataclass
class MutationParameters:
    """Parameters controlling mutation operations."""
    prompt_mutation_rate: float = 0.1  # Probability of prompt mutation per generation
    parameter_mutation_rate: float = 0.05  # Probability of parameter mutation per generation
    temperature_mutation_range: float = 0.1  # Range for temperature mutation
    top_p_mutation_range: float = 0.1  # Range for top_p mutation
    max_mutation_intensity: float = 0.3  # Maximum intensity of mutations


class EvolutionaryMechanisms:
    """
    Enhanced evolutionary mechanisms for cognitive heterogeneity validation.
    
    This class implements true evolutionary mechanisms including mutation, crossover,
    and diversity maintenance to achieve collective intelligence emergence.
    """
    
    def __init__(self, mutation_params: Optional[MutationParameters] = None):
        """
        Initialize evolutionary mechanisms.
        
        Args:
            mutation_params: Parameters controlling mutation operations
        """
        self.mutation_params = mutation_params or MutationParameters()
        logger.info("Initialized EvolutionaryMechanisms with params: %s", self.mutation_params)
    
    def mutate_agent(self, agent: Agent, generation: int = 0) -> Agent:
        """
        Mutate an agent by applying prompt and parameter mutations.
        
        Args:
            agent: Agent to mutate
            generation: Current generation number (for scheduling)
            
        Returns:
            Mutated agent (new instance)
        """
        # Create a deep copy of the agent to avoid modifying the original
        mutated_agent = copy.deepcopy(agent)
        
        # Apply prompt mutation
        if random.random() < self.mutation_params.prompt_mutation_rate:
            mutated_agent = self._mutate_prompt(mutated_agent, generation)
        
        # Apply parameter mutation
        if random.random() < self.mutation_params.parameter_mutation_rate:
            mutated_agent = self._mutate_parameters(mutated_agent, generation)
        
        # Update agent metadata
        mutated_agent.generation = agent.generation + 1
        mutated_agent.is_variant = True
        mutated_agent.original_source = agent.agent_id
        mutated_agent.agent_id = f"{agent.agent_id}_mutated_{generation}_{random.randint(1000, 9999)}"
        
        logger.debug(f"Mutated agent {agent.agent_id} to {mutated_agent.agent_id}")
        return mutated_agent
    
    def _mutate_prompt(self, agent: Agent, generation: int) -> Agent:
        """
        Mutate agent prompt by applying semantic-preserving changes.
        
        Args:
            agent: Agent whose prompt to mutate
            generation: Current generation number (for intensity scheduling)
            
        Returns:
            Agent with mutated prompt
        """
        # Get current prompt
        current_prompt = agent.config.get("prompt", "")
        if not current_prompt:
            logger.warning(f"Agent {agent.agent_id} has no prompt to mutate")
            return agent
        
        # Apply mutation intensity scheduling (decrease with generation)
        intensity = self.mutation_params.max_mutation_intensity * (1 - generation * 0.01)
        intensity = max(0.05, min(intensity, self.mutation_params.max_mutation_intensity))
        
        # Simple prompt mutation strategies:
        # 1. Add emphasis words
        # 2. Rephrase sentences
        # 3. Add modifiers
        
        # For now, we'll implement a simple approach that adds emphasis or modifiers
        if "critical" in agent.agent_id.lower() or "critical" in current_prompt.lower():
            # For critical agents, add more analytical language
            emphasis_words = ["thoroughly", "carefully", "rigorously", "systematically"]
        elif "awakened" in agent.agent_id.lower() or "awakened" in current_prompt.lower():
            # For awakened agents, add more questioning language
            emphasis_words = ["question", "challenge", "investigate", "examine"]
        else:
            # For standard agents, add general emphasis
            emphasis_words = ["carefully", "thoroughly", "completely", "fully"]
        
        # Randomly select an emphasis word
        if emphasis_words:
            emphasis = random.choice(emphasis_words)
            if emphasis not in current_prompt:
                # Add emphasis to the beginning of the prompt
                agent.config["prompt"] = f"Please {emphasis} " + current_prompt
        
        logger.debug(f"Applied prompt mutation to agent {agent.agent_id}")
        return agent
    
    def _mutate_parameters(self, agent: Agent, generation: int) -> Agent:
        """
        Mutate agent generation parameters.
        
        Args:
            agent: Agent whose parameters to mutate
            generation: Current generation number (for intensity scheduling)
            
        Returns:
            Agent with mutated parameters
        """
        # Get current parameters
        model_config = agent.config.get("model_config", {})
        
        # Default parameters if not specified
        current_temperature = model_config.get("temperature", 0.7)
        current_top_p = model_config.get("top_p", 0.9)
        
        # Apply mutation intensity scheduling
        intensity = self.mutation_params.max_mutation_intensity * (1 - generation * 0.01)
        intensity = max(0.05, min(intensity, self.mutation_params.max_mutation_intensity))
        
        # Mutate temperature
        temp_change = random.uniform(
            -self.mutation_params.temperature_mutation_range * intensity,
            self.mutation_params.temperature_mutation_range * intensity
        )
        new_temperature = current_temperature + temp_change
        # Keep within reasonable bounds
        new_temperature = max(0.1, min(new_temperature, 1.0))
        
        # Mutate top_p
        top_p_change = random.uniform(
            -self.mutation_params.top_p_mutation_range * intensity,
            self.mutation_params.top_p_mutation_range * intensity
        )
        new_top_p = current_top_p + top_p_change
        # Keep within reasonable bounds
        new_top_p = max(0.1, min(new_top_p, 1.0))
        
        # Update parameters
        if "model_config" not in agent.config:
            agent.config["model_config"] = {}
        
        agent.config["model_config"]["temperature"] = new_temperature
        agent.config["model_config"]["top_p"] = new_top_p
        
        logger.debug(f"Mutated parameters for agent {agent.agent_id}: temp={new_temperature:.3f}, top_p={new_top_p:.3f}")
        return agent
    
    def crossover_agents(self, agent1: Agent, agent2: Agent, generation: int = 0) -> Agent:
        """
        Create a new agent by combining characteristics of two parent agents.
        
        Args:
            agent1: First parent agent
            agent2: Second parent agent
            generation: Current generation number
            
        Returns:
            New agent combining parent characteristics
        """
        # Create new agent ID
        new_agent_id = f"crossover_{agent1.agent_id}_{agent2.agent_id}_{generation}_{random.randint(1000, 9999)}"
        
        # Combine prompts by taking parts from each parent
        prompt1 = agent1.config.get("prompt", "")
        prompt2 = agent2.config.get("prompt", "")
        
        # Simple crossover: combine parts of prompts
        if prompt1 and prompt2:
            # Split prompts into sentences (simplified approach)
            sentences1 = prompt1.split(". ")
            sentences2 = prompt2.split(". ")
            
            # Combine sentences from both parents
            combined_sentences = []
            max_sentences = max(len(sentences1), len(sentences2))
            
            for i in range(max_sentences):
                if i < len(sentences1) and i % 2 == 0:
                    combined_sentences.append(sentences1[i])
                elif i < len(sentences2):
                    combined_sentences.append(sentences2[i])
                elif i < len(sentences1):
                    combined_sentences.append(sentences1[i])
            
            new_prompt = ". ".join(combined_sentences)
            if not new_prompt.endswith("."):
                new_prompt += "."
        else:
            # If one prompt is empty, use the other
            new_prompt = prompt1 if prompt1 else prompt2
        
        # Combine model configurations
        config1 = agent1.config.get("model_config", {})
        config2 = agent2.config.get("model_config", {})
        
        # Average numerical parameters
        new_config = {}
        all_keys = set(config1.keys()) | set(config2.keys())
        
        for key in all_keys:
            val1 = config1.get(key)
            val2 = config2.get(key)
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Average numerical values
                new_config[key] = (val1 + val2) / 2
            elif val1 is not None:
                # Use first parent's value if available
                new_config[key] = val1
            else:
                # Use second parent's value
                new_config[key] = val2
        
        # Create new agent with combined characteristics
        new_agent = OllamaAgent(
            agent_id=new_agent_id,
            config={
                "model": agent1.config.get("model", "gemma3:latest"),
                "prompt": new_prompt,
                "model_config": new_config
            },
            generation=max(agent1.generation, agent2.generation) + 1,
            is_variant=True,
            original_source=f"{agent1.agent_id}+{agent2.agent_id}"
        )
        
        logger.debug(f"Created crossover agent {new_agent_id} from {agent1.agent_id} and {agent2.agent_id}")
        return new_agent
    
    def calculate_diversity_index(self, agents: List[Agent]) -> Dict[str, float]:
        """
        Calculate multi-dimensional diversity index for agent population.
        
        Args:
            agents: List of agents in the population
            
        Returns:
            Dictionary containing diversity metrics
        """
        if not agents:
            return {
                'simpson_index': 0.0,
                'shannon_entropy': 0.0,
                'prompt_diversity': 0.0,
                'parameter_diversity': 0.0
            }
        
        n = len(agents)
        if n <= 1:
            return {
                'simpson_index': 0.0,
                'shannon_entropy': 0.0,
                'prompt_diversity': 0.0,
                'parameter_diversity': 0.0
            }
        
        # 1. Simpson's diversity index (based on agent types)
        agent_types = {}
        for agent in agents:
            # Classify agent by type
            if "critical" in agent.agent_id.lower():
                agent_type = "critical"
            elif "awakened" in agent.agent_id.lower():
                agent_type = "awakened"
            else:
                agent_type = "standard"
            
            agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
        
        # Calculate Simpson's index: 1 - Σ(p_i^2)
        simpson_index = 1.0
        for count in agent_types.values():
            p_i = count / n
            simpson_index -= p_i ** 2
        
        # 2. Shannon entropy (based on agent types)
        shannon_entropy = 0.0
        for count in agent_types.values():
            p_i = count / n
            if p_i > 0:
                shannon_entropy -= p_i * np.log(p_i)
        
        # 3. Prompt diversity (simplified - based on unique prompts)
        unique_prompts = set(agent.config.get("prompt", "") for agent in agents)
        prompt_diversity = len(unique_prompts) / n if n > 0 else 0.0
        
        # 4. Parameter diversity (simplified - based on temperature variation)
        temperatures = [agent.config.get("model_config", {}).get("temperature", 0.7) for agent in agents]
        if temperatures:
            temp_std = np.std(temperatures)
            # Normalize by maximum possible standard deviation (0.45 for range 0.1-1.0)
            parameter_diversity = min(temp_std / 0.45, 1.0) if temp_std > 0 else 0.0
        else:
            parameter_diversity = 0.0
        
        return {
            'simpson_index': simpson_index,
            'shannon_entropy': shannon_entropy,
            'prompt_diversity': prompt_diversity,
            'parameter_diversity': parameter_diversity
        }
    
    def maintain_diversity(self, agents: List[Agent], target_diversity: float = 0.6) -> List[Agent]:
        """
        Maintain cognitive diversity by introducing variation when diversity drops below threshold.
        
        Args:
            agents: Current agent population
            target_diversity: Target diversity level to maintain
            
        Returns:
            Updated agent population with maintained diversity
        """
        if not agents:
            return agents
        
        # Calculate current diversity
        diversity_metrics = self.calculate_diversity_index(agents)
        current_diversity = diversity_metrics['simpson_index']
        
        logger.debug(f"Current diversity: {current_diversity:.3f}, Target: {target_diversity:.3f}")
        
        # If diversity is too low, introduce variation
        if current_diversity < target_diversity:
            logger.info(f"Low diversity detected ({current_diversity:.3f} < {target_diversity:.3f}), introducing variation")
            
            # Strategy: Mutate some agents to increase diversity
            agents_to_mutate = max(1, len(agents) // 5)  # Mutate 20% of agents
            agents_to_mutate = min(agents_to_mutate, len(agents))
            
            # Select agents to mutate (prefer those that haven't been mutated recently)
            sorted_agents = sorted(agents, key=lambda a: a.generation)
            agents_for_mutation = sorted_agents[:agents_to_mutate]
            
            # Mutate selected agents
            mutated_agents = []
            for agent in agents_for_mutation:
                mutated_agent = self.mutate_agent(agent, agent.generation)
                mutated_agents.append(mutated_agent)
            
            # Replace original agents with mutated ones
            new_agents = agents.copy()
            for i, original_agent in enumerate(agents_for_mutation):
                if i < len(mutated_agents):
                    # Find the index of the original agent and replace it
                    for j, agent in enumerate(new_agents):
                        if agent.agent_id == original_agent.agent_id:
                            new_agents[j] = mutated_agents[i]
                            break
            
            logger.info(f"Introduced {len(mutated_agents)} mutations to maintain diversity")
            return new_agents
        
        # Diversity is adequate, return original agents
        return agents


# Exported classes and functions
__all__ = [
    'MutationParameters',
    'EvolutionaryMechanisms'
]