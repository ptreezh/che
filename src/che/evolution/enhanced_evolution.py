"""
Enhanced Evolutionary Mechanisms for Cognitive Heterogeneity Validation

This module provides enhanced evolutionary mechanisms including mutation, crossover,
and diversity maintenance to achieve collective intelligence emergence.

Authors: CHE Research Team
Date: 2025-10-31
"""

import random
import copy
import logging
from typing import Dict, List, Any, Optional
import numpy as np

from ..core.agent import Agent
from ..agents.ollama_agent import OllamaAgent
from ..prompts import PromptType, get_prompt

logger = logging.getLogger(__name__)


class MutationEngine:
    """Engine for implementing various mutation mechanisms in cognitive agents."""
    
    def __init__(self, mutation_rate: float = 0.1, mutation_intensity: float = 0.2):
        """
        Initialize the mutation engine.
        
        Args:
            mutation_rate: Probability of mutation occurring (0.0 to 1.0)
            mutation_intensity: Intensity of mutation (0.0 to 1.0)
        """
        self.mutation_rate = mutation_rate
        self.mutation_intensity = mutation_intensity
        logger.info(f"MutationEngine initialized with rate={mutation_rate}, intensity={mutation_intensity}")
    
    def mutate_agent(self, agent: OllamaAgent, agent_id: str, performance_history: List[float] = None) -> OllamaAgent:
        """
        Apply mutation to an agent.
        
        Args:
            agent: The agent to mutate
            agent_id: New ID for the mutated agent
            performance_history: History of agent performance scores
            
        Returns:
            New mutated agent instance
        """
        # Check if mutation should occur based on mutation rate
        if random.random() > self.mutation_rate:
            # No mutation, return copy of original agent
            new_agent = OllamaAgent(
                agent_id=agent_id,
                config=copy.deepcopy(agent.config),
                generation=agent.generation + 1,
                is_variant=True,
                original_source=agent.agent_id
            )
            return new_agent
        
        # Determine agent type for parameter mutation
        agent_type = self._classify_agent_type(agent)
        
        # Apply mutation to agent configuration
        mutated_config = copy.deepcopy(agent.config)
        
        # Mutate prompt if present
        if 'prompt' in mutated_config:
            mutated_config['prompt'] = self._mutate_prompt(mutated_config['prompt'])
        
        # Mutate model parameters if present
        if 'temperature' in mutated_config:
            mutated_config['temperature'] = self._mutate_parameter(
                mutated_config['temperature'], 
                "temperature", 
                agent_type, 
                performance_history
            )
        if 'top_p' in mutated_config:
            mutated_config['top_p'] = self._mutate_parameter(
                mutated_config['top_p'], 
                "top_p", 
                agent_type, 
                performance_history
            )
        
        # Create mutated agent
        mutated_agent = OllamaAgent(
            agent_id=agent_id,
            config=mutated_config,
            generation=agent.generation + 1,
            is_variant=True,
            original_source=agent.agent_id
        )
        
        logger.debug(f"Applied mutation to agent {agent.agent_id} -> {agent_id} (type: {agent_type})")
        return mutated_agent
    
    def _classify_agent_type(self, agent: OllamaAgent) -> str:
        """
        Classify agent type based on its prompt.
        
        Args:
            agent: The agent to classify
            
        Returns:
            Agent type as string ("critical", "awakened", or "standard")
        """
        prompt = agent.config.get('prompt', '').lower()
        if 'critical' in prompt or 'skeptical' in prompt:
            return 'critical'
        elif 'awakened' in prompt or 'question' in prompt:
            return 'awakened'
        else:
            return 'standard'
    
    def _mutate_prompt(self, prompt: str) -> str:
        """
        Mutate agent prompt by applying semantic-preserving changes.
        
        Args:
            prompt: Original prompt
            
        Returns:
            Mutated prompt
        """
        # Simple prompt mutation strategies
        mutation_strategies = [
            self._add_emphasis,
            self._rephrase_sentences,
            self._add_constraints,
            self._modify_tone
        ]
        
        # Apply one random mutation strategy with some intensity
        if random.random() < self.mutation_intensity:
            strategy = random.choice(mutation_strategies)
            try:
                mutated_prompt = strategy(prompt)
                logger.debug(f"Applied prompt mutation: {strategy.__name__}")
                return mutated_prompt
            except Exception as e:
                logger.warning(f"Prompt mutation failed: {e}. Returning original prompt.")
                return prompt
        
        return prompt
    
    def _add_emphasis(self, prompt: str) -> str:
        """Add emphasis words to prompt."""
        emphasis_words = ["carefully", "thoroughly", "strictly", "precisely", "meticulously"]
        emphasis = random.choice(emphasis_words)
        
        # Add emphasis to beginning if not already present
        if emphasis not in prompt.lower():
            return f"{emphasis.capitalize()} {prompt.lower()}"
        return prompt
    
    def _rephrase_sentences(self, prompt: str) -> str:
        """Rephrase sentences in prompt."""
        # Simple rephrasing by changing sentence structure
        if "you are" in prompt.lower():
            return prompt.replace("you are", "as a").replace("You are", "As a")
        elif "as a" in prompt.lower():
            return prompt.replace("as a", "you are").replace("As a", "You are")
        return prompt
    
    def _add_constraints(self, prompt: str) -> str:
        """Add constraints to prompt."""
        constraints = [
            "without making assumptions",
            "based only on facts",
            "with evidence",
            "avoiding speculation"
        ]
        constraint = random.choice(constraints)
        
        # Add constraint if not already present
        if constraint not in prompt.lower():
            return f"{prompt} {constraint}"
        return prompt
    
    def _modify_tone(self, prompt: str) -> str:
        """Modify the tone of the prompt."""
        tone_modifiers = [
            "in a scholarly manner",
            "with academic rigor",
            "professionally",
            "objectively"
        ]
        modifier = random.choice(tone_modifiers)
        
        # Add tone modifier if not already present
        if modifier not in prompt.lower():
            return f"{prompt} {modifier}"
        return prompt
    
    def _mutate_parameter(self, param: float, param_name: str = "default", agent_type: str = "standard", performance_history: List[float] = None) -> float:
        """
        Mutate a numerical parameter with enhanced strategies.
        
        Args:
            param: Original parameter value
            param_name: Name of the parameter being mutated (e.g., "temperature", "top_p")
            agent_type: Type of agent (e.g., "critical", "awakened", "standard")
            performance_history: History of agent performance scores
            
        Returns:
            Mutated parameter value
        """
        # Apply mutation based on parameter type and agent type
        if param_name == "temperature":
            mutated_param = self._mutate_temperature(param, agent_type, performance_history)
        elif param_name == "top_p":
            mutated_param = self._mutate_top_p(param, agent_type, performance_history)
        else:
            # Apply small random change based on mutation intensity
            change = random.uniform(-self.mutation_intensity, self.mutation_intensity)
            mutated_param = param + change
        
        # Keep parameter within reasonable bounds
        if param_name in ["temperature", "top_p"]:
            # For temperature and top_p, use appropriate ranges
            return max(0.0, min(1.0, mutated_param))
        else:
            # For other parameters, use generic bounds
            return max(0.0, min(1.0, mutated_param))
    
    def _mutate_temperature(self, temperature: float, agent_type: str, performance_history: List[float]) -> float:
        """
        Enhanced temperature mutation with adaptive strategies.
        
        Args:
            temperature: Original temperature value
            agent_type: Type of agent affecting mutation strategy
            performance_history: History of agent performance scores
            
        Returns:
            Mutated temperature value
        """
        # Base mutation
        change = random.uniform(-self.mutation_intensity, self.mutation_intensity)
        mutated_temp = temperature + change
        
        # Adaptive adjustment based on agent type
        if agent_type == "critical":
            # Critical agents benefit from lower temperatures for more precise reasoning
            mutated_temp *= 0.9
        elif agent_type == "awakened":
            # Awakened agents benefit from moderate temperatures for creative thinking
            mutated_temp *= 1.0
        else:  # standard
            # Standard agents use base mutation
            pass
        
        # Performance-based adjustment
        if performance_history and len(performance_history) > 1:
            recent_performance = performance_history[-1]
            previous_performance = performance_history[-2] if len(performance_history) > 1 else recent_performance
            
            # If performance is improving, reduce temperature for consistency
            if recent_performance > previous_performance:
                mutated_temp *= 0.95
            # If performance is declining, increase temperature for exploration
            elif recent_performance < previous_performance:
                mutated_temp *= 1.05
        
        # Ensure temperature stays in a reasonable range for different agent types
        if agent_type == "critical":
            # Critical agents work best with lower temperatures
            return max(0.1, min(0.7, mutated_temp))
        elif agent_type == "awakened":
            # Awakened agents work best with moderate temperatures
            return max(0.3, min(0.9, mutated_temp))
        else:  # standard
            # Standard agents work well with a broader range
            return max(0.2, min(1.0, mutated_temp))
    
    def _mutate_top_p(self, top_p: float, agent_type: str, performance_history: List[float]) -> float:
        """
        Enhanced top_p mutation with adaptive strategies.
        
        Args:
            top_p: Original top_p value
            agent_type: Type of agent affecting mutation strategy
            performance_history: History of agent performance scores
            
        Returns:
            Mutated top_p value
        """
        # Base mutation
        change = random.uniform(-self.mutation_intensity, self.mutation_intensity)
        mutated_top_p = top_p + change
        
        # Agent type based adjustment
        if agent_type == "critical":
            # Critical agents benefit from lower top_p for more focused sampling
            mutated_top_p *= 0.9
        elif agent_type == "awakened":
            # Awakened agents benefit from higher top_p for more diverse sampling
            mutated_top_p *= 1.1
        else:  # standard
            # Standard agents use base mutation
            pass
        
        # Performance-based adjustment
        if performance_history and len(performance_history) > 1:
            recent_performance = performance_history[-1]
            previous_performance = performance_history[-2] if len(performance_history) > 1 else recent_performance
            
            # If performance is improving, reduce top_p for consistency
            if recent_performance > previous_performance:
                mutated_top_p *= 0.95
            # If performance is declining, increase top_p for exploration
            elif recent_performance < previous_performance:
                mutated_top_p *= 1.05
        
        # Keep top_p within reasonable bounds
        return max(0.1, min(1.0, mutated_top_p))


class CrossoverEngine:
    """Engine for implementing crossover mechanisms between cognitive agents."""
    
    def __init__(self, crossover_rate: float = 0.2):
        """
        Initialize the crossover engine.
        
        Args:
            crossover_rate: Probability of crossover occurring (0.0 to 1.0)
        """
        self.crossover_rate = crossover_rate
        logger.info(f"CrossoverEngine initialized with rate={crossover_rate}")
    
    def crossover_agents(self, agent1: OllamaAgent, agent2: OllamaAgent, child_id: str) -> OllamaAgent:
        """
        Create a new agent by combining characteristics of two parent agents.
        
        Args:
            agent1: First parent agent
            agent2: Second parent agent
            child_id: ID for the new child agent
            
        Returns:
            New child agent combining parent characteristics
        """
        # Check if crossover should occur based on crossover rate
        if random.random() > self.crossover_rate:
            # No crossover, return copy of first parent
            child_agent = OllamaAgent(
                agent_id=child_id,
                config=copy.deepcopy(agent1.config),
                generation=max(agent1.generation, agent2.generation) + 1,
                is_variant=True,
                original_source=f"{agent1.agent_id}+{agent2.agent_id}"
            )
            return child_agent
        
        # Combine parent configurations
        child_config = self._combine_configs(agent1.config, agent2.config)
        
        # Create child agent
        child_agent = OllamaAgent(
            agent_id=child_id,
            config=child_config,
            generation=max(agent1.generation, agent2.generation) + 1,
            is_variant=True,
            original_source=f"{agent1.agent_id}+{agent2.agent_id}"
        )
        
        logger.debug(f"Applied crossover: {agent1.agent_id} + {agent2.agent_id} -> {child_id}")
        return child_agent
    
    def _combine_configs(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine two agent configurations.
        
        Args:
            config1: First configuration
            config2: Second configuration
            
        Returns:
            Combined configuration
        """
        combined_config = copy.deepcopy(config1)
        
        # Combine prompts by averaging or selecting parts
        if 'prompt' in config1 and 'prompt' in config2:
            combined_config['prompt'] = self._combine_prompts(config1['prompt'], config2['prompt'])
        
        # Combine numerical parameters by averaging
        for param in ['temperature', 'top_p']:
            if param in config1 and param in config2:
                combined_config[param] = (config1[param] + config2[param]) / 2
        
        return combined_config
    
    def _combine_prompts(self, prompt1: str, prompt2: str) -> str:
        """
        Combine two prompts.
        
        Args:
            prompt1: First prompt
            prompt2: Second prompt
            
        Returns:
            Combined prompt
        """
        # Simple combination strategy: take parts from both prompts
        sentences1 = prompt1.split('.')
        sentences2 = prompt2.split('.')
        
        # Combine sentences from both prompts
        combined_sentences = []
        max_len = max(len(sentences1), len(sentences2))
        
        for i in range(max_len):
            if i < len(sentences1) and i < len(sentences2):
                # Randomly choose sentence from either prompt
                combined_sentences.append(random.choice([sentences1[i], sentences2[i]]))
            elif i < len(sentences1):
                combined_sentences.append(sentences1[i])
            else:
                combined_sentences.append(sentences2[i])
        
        return '.'.join(combined_sentences)


class DiversityMonitor:
    """Monitor for tracking and maintaining cognitive diversity."""
    
    def __init__(self, diversity_threshold: float = 0.3):
        """
        Initialize the diversity monitor.
        
        Args:
            diversity_threshold: Minimum acceptable diversity level (0.0 to 1.0)
        """
        self.diversity_threshold = diversity_threshold
        self.diversity_history = []
        logger.info(f"DiversityMonitor initialized with threshold={diversity_threshold}")
    
    def calculate_diversity(self, agents: Dict[str, Agent]) -> float:
        """
        Calculate cognitive diversity index of the agent population.
        
        Args:
            agents: Dictionary of agents in the population
            
        Returns:
            Diversity index (0.0 to 1.0)
        """
        if not agents:
            return 0.0
        
        # Count agent types based on prompt characteristics
        agent_types = {}
        for agent in agents.values():
            # Simplified type classification based on prompt content
            prompt = agent.config.get('prompt', '').lower()
            if 'critical' in prompt or 'skeptical' in prompt:
                agent_type = 'critical'
            elif 'awakened' in prompt or 'question' in prompt:
                agent_type = 'awakened'
            else:
                agent_type = 'standard'
            
            agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
        
        # Calculate type-based diversity using Simpson's diversity index
        total_agents = len(agents)
        if total_agents <= 1:
            return 0.0
        
        # Calculate sum of squared proportions for type diversity
        sum_squared_proportions = sum((count / total_agents) ** 2 for count in agent_types.values())
        
        # Convert to diversity index (higher values indicate higher diversity)
        type_diversity_index = 1.0 - sum_squared_proportions
        
        # Normalize type diversity to 0.0-1.0 range
        max_possible_type_diversity = 1.0 - (1.0 / total_agents)
        if max_possible_type_diversity > 0:
            normalized_type_diversity = type_diversity_index / max_possible_type_diversity
        else:
            normalized_type_diversity = 0.0
        
        type_diversity = min(1.0, max(0.0, normalized_type_diversity))
        
        # Calculate parameter-based diversity (for parameters like temperature, top_p)
        param_diversity = self._calculate_parameter_diversity(agents)
        
        # Combine type diversity and parameter diversity with equal weight
        combined_diversity = 0.5 * type_diversity + 0.5 * param_diversity
        
        diversity = min(1.0, max(0.0, combined_diversity))
        self.diversity_history.append(diversity)
        
        logger.debug(f"Calculated diversity: {diversity:.3f} (type: {type_diversity:.3f}, param: {param_diversity:.3f})")
        return diversity
    
    def _calculate_parameter_diversity(self, agents: Dict[str, Agent]) -> float:
        """
        Calculate diversity based on parameter differences (temperature, top_p, etc.).
        
        Args:
            agents: Dictionary of agents in the population
            
        Returns:
            Parameter diversity index (0.0 to 1.0)
        """
        # Collect parameter values from agents
        temperature_values = []
        top_p_values = []
        
        for agent in agents.values():
            config = agent.config
            if 'temperature' in config and config['temperature'] is not None:
                temperature_values.append(config['temperature'])
            if 'top_p' in config and config['top_p'] is not None:
                top_p_values.append(config['top_p'])
        
        # Calculate parameter variance if we have values
        param_diversity = 0.0
        if temperature_values:
            temp_std = np.std(temperature_values)
            temp_diversity = min(1.0, temp_std * 2.0)  # Scale to 0-1 range
            param_diversity += 0.5 * temp_diversity
        if top_p_values:
            top_p_std = np.std(top_p_values)
            top_p_diversity = min(1.0, top_p_std * 2.0)  # Scale to 0-1 range
            param_diversity += 0.5 * top_p_diversity
        
        return param_diversity
    
    def needs_diversity_intervention(self, agents: Dict[str, Agent]) -> bool:
        """
        Check if diversity intervention is needed.
        
        Args:
            agents: Dictionary of agents in the population
            
        Returns:
            True if diversity intervention is needed, False otherwise
        """
        current_diversity = self.calculate_diversity(agents)
        needs_intervention = current_diversity < self.diversity_threshold
        
        if needs_intervention:
            logger.warning(f"Diversity below threshold: {current_diversity:.3f} < {self.diversity_threshold:.3f}")
        
        return needs_intervention
    
    def suggest_diversity_intervention(self, agents: Dict[str, Agent]) -> Dict[str, Any]:
        """
        Suggest diversity intervention strategies.
        
        Args:
            agents: Dictionary of agents in the population
            
        Returns:
            Dictionary containing intervention suggestions
        """
        # Count current agent types
        agent_types = {}
        for agent in agents.values():
            prompt = agent.config.get('prompt', '').lower()
            if 'critical' in prompt or 'skeptical' in prompt:
                agent_type = 'critical'
            elif 'awakened' in prompt or 'question' in prompt:
                agent_type = 'awakened'
            else:
                agent_type = 'standard'
            
            agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
        
        # Find underrepresented types
        total_agents = len(agents)
        target_count = total_agents // 3  # Aim for roughly equal distribution
        
        interventions = []
        for agent_type in ['critical', 'awakened', 'standard']:
            current_count = agent_types.get(agent_type, 0)
            if current_count < target_count:
                needed_count = target_count - current_count
                interventions.append({
                    'type': agent_type,
                    'needed_count': needed_count,
                    'strategy': 'introduce_new_agents'
                })
        
        return {
            'current_diversity': self.calculate_diversity(agents),
            'threshold': self.diversity_threshold,
            'interventions': interventions
        }


# Exported classes
__all__ = [
    'MutationEngine',
    'CrossoverEngine',
    'DiversityMonitor'
]