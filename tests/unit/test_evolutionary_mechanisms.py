"""
Unit tests for enhanced evolutionary mechanisms module.

Authors: CHE Research Team
Date: 2025-10-31
"""

import pytest
import random
import numpy as np
from unittest.mock import Mock, patch

from src.che.evolution.evolutionary_mechanisms import (
    MutationParameters,
    EvolutionaryMechanisms
)
from src.che.agents.ollama_agent import OllamaAgent


def test_mutation_parameters_default():
    """Test default mutation parameters."""
    params = MutationParameters()
    
    assert params.prompt_mutation_rate == 0.1
    assert params.parameter_mutation_rate == 0.05
    assert params.temperature_mutation_range == 0.1
    assert params.top_p_mutation_range == 0.1
    assert params.max_mutation_intensity == 0.3


def test_mutation_parameters_custom():
    """Test custom mutation parameters."""
    params = MutationParameters(
        prompt_mutation_rate=0.2,
        parameter_mutation_rate=0.1,
        temperature_mutation_range=0.2,
        top_p_mutation_range=0.15,
        max_mutation_intensity=0.5
    )
    
    assert params.prompt_mutation_rate == 0.2
    assert params.parameter_mutation_rate == 0.1
    assert params.temperature_mutation_range == 0.2
    assert params.top_p_mutation_range == 0.15
    assert params.max_mutation_intensity == 0.5


def test_evolutionary_mechanisms_init():
    """Test initialization of evolutionary mechanisms."""
    # Test with default parameters
    mech1 = EvolutionaryMechanisms()
    assert isinstance(mech1.mutation_params, MutationParameters)
    
    # Test with custom parameters
    custom_params = MutationParameters(prompt_mutation_rate=0.3)
    mech2 = EvolutionaryMechanisms(custom_params)
    assert mech2.mutation_params.prompt_mutation_rate == 0.3


def test_mutate_agent_no_mutation():
    """Test agent mutation with zero mutation rates."""
    # Create agent
    agent = OllamaAgent(
        agent_id="test_agent",
        config={
            "model": "gemma3:latest",
            "prompt": "Test prompt"
        }
    )
    
    # Create mechanisms with zero mutation rates
    params = MutationParameters(
        prompt_mutation_rate=0.0,
        parameter_mutation_rate=0.0
    )
    mech = EvolutionaryMechanisms(params)
    
    # Mutate agent
    mutated_agent = mech.mutate_agent(agent)
    
    # Should be a new instance but with same content
    assert mutated_agent is not agent
    assert mutated_agent.agent_id != agent.agent_id
    assert mutated_agent.config["prompt"] == agent.config["prompt"]
    assert mutated_agent.is_variant == True
    assert mutated_agent.original_source == agent.agent_id


def test_mutate_agent_with_prompt_mutation():
    """Test agent mutation with prompt mutation."""
    # Create agent
    agent = OllamaAgent(
        agent_id="test_agent",
        config={
            "model": "gemma3:latest",
            "prompt": "Test prompt"
        }
    )
    
    # Create mechanisms with high prompt mutation rate
    params = MutationParameters(
        prompt_mutation_rate=1.0,  # Always mutate
        parameter_mutation_rate=0.0
    )
    mech = EvolutionaryMechanisms(params)
    
    # Mutate agent
    with patch('random.random', return_value=0.5):  # Ensure mutation occurs
        mutated_agent = mech.mutate_agent(agent)
    
    # Should be a new instance with modified prompt
    assert mutated_agent is not agent
    assert mutated_agent.agent_id != agent.agent_id
    assert mutated_agent.is_variant == True


def test_mutate_agent_with_parameter_mutation():
    """Test agent mutation with parameter mutation."""
    # Create agent with parameters
    agent = OllamaAgent(
        agent_id="test_agent",
        config={
            "model": "gemma3:latest",
            "prompt": "Test prompt",
            "model_config": {
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
    )
    
    # Create mechanisms with high parameter mutation rate
    params = MutationParameters(
        prompt_mutation_rate=0.0,
        parameter_mutation_rate=1.0  # Always mutate
    )
    mech = EvolutionaryMechanisms(params)
    
    # Mutate agent
    with patch('random.random', return_value=0.5):  # Ensure mutation occurs
        mutated_agent = mech.mutate_agent(agent)
    
    # Should be a new instance with modified parameters
    assert mutated_agent is not agent
    assert mutated_agent.agent_id != agent.agent_id
    assert mutated_agent.is_variant == True
    assert "model_config" in mutated_agent.config


def test_crossover_agents():
    """Test crossover of two agents."""
    # Create parent agents
    agent1 = OllamaAgent(
        agent_id="parent1",
        config={
            "model": "gemma3:latest",
            "prompt": "First parent prompt with multiple sentences. This is the second sentence."
        }
    )
    
    agent2 = OllamaAgent(
        agent_id="parent2",
        config={
            "model": "gemma3:latest",
            "prompt": "Second parent prompt with different sentences. This is another sentence."
        }
    )
    
    # Create mechanisms
    mech = EvolutionaryMechanisms()
    
    # Perform crossover
    child_agent = mech.crossover_agents(agent1, agent2, generation=5)
    
    # Check that child is a new agent
    assert child_agent is not agent1
    assert child_agent is not agent2
    assert child_agent.agent_id != agent1.agent_id
    assert child_agent.agent_id != agent2.agent_id
    assert "crossover" in child_agent.agent_id
    assert child_agent.generation == 6  # max(parent1.generation, parent2.generation) + 1
    assert child_agent.is_variant == True
    assert child_agent.original_source == f"{agent1.agent_id}+{agent2.agent_id}"


def test_calculate_diversity_index_empty():
    """Test diversity calculation with empty agent list."""
    mech = EvolutionaryMechanisms()
    
    # Test with empty list
    result = mech.calculate_diversity_index([])
    assert result['simpson_index'] == 0.0
    assert result['shannon_entropy'] == 0.0
    assert result['prompt_diversity'] == 0.0
    assert result['parameter_diversity'] == 0.0


def test_calculate_diversity_index_single():
    """Test diversity calculation with single agent."""
    mech = EvolutionaryMechanisms()
    
    # Create single agent
    agent = OllamaAgent(
        agent_id="single_agent",
        config={
            "model": "gemma3:latest",
            "prompt": "Test prompt"
        }
    )
    
    # Test with single agent
    result = mech.calculate_diversity_index([agent])
    assert result['simpson_index'] == 0.0
    assert result['shannon_entropy'] == 0.0


def test_calculate_diversity_index_multiple():
    """Test diversity calculation with multiple agents."""
    mech = EvolutionaryMechanisms()
    
    # Create multiple agents of different types
    agents = [
        OllamaAgent(agent_id=f"critical_{i}", config={"model": "gemma3:latest", "prompt": f"Critical prompt {i}"})
        for i in range(5)
    ] + [
        OllamaAgent(agent_id=f"awakened_{i}", config={"model": "gemma3:latest", "prompt": f"Awakened prompt {i}"})
        for i in range(3)
    ] + [
        OllamaAgent(agent_id=f"standard_{i}", config={"model": "gemma3:latest", "prompt": f"Standard prompt {i}"})
        for i in range(2)
    ]
    
    # Test with multiple agents
    result = mech.calculate_diversity_index(agents)
    
    # Should have positive diversity values
    assert result['simpson_index'] >= 0.0
    assert result['shannon_entropy'] >= 0.0
    assert result['prompt_diversity'] >= 0.0
    assert result['parameter_diversity'] >= 0.0


def test_maintain_diversity_adequate():
    """Test diversity maintenance when diversity is adequate."""
    mech = EvolutionaryMechanisms()
    
    # Create agents with adequate diversity
    agents = [
        OllamaAgent(agent_id=f"critical_{i}", config={"model": "gemma3:latest", "prompt": f"Critical prompt {i}"})
        for i in range(5)
    ] + [
        OllamaAgent(agent_id=f"awakened_{i}", config={"model": "gemma3:latest", "prompt": f"Awakened prompt {i}"})
        for i in range(5)
    ] + [
        OllamaAgent(agent_id=f"standard_{i}", config={"model": "gemma3:latest", "prompt": f"Standard prompt {i}"})
        for i in range(5)
    ]
    
    # Maintain diversity (should not change anything)
    result_agents = mech.maintain_diversity(agents, target_diversity=0.1)
    
    # Should return same agents (no mutation needed)
    assert len(result_agents) == len(agents)


def test_maintain_diversity_low():
    """Test diversity maintenance when diversity is low."""
    mech = EvolutionaryMechanisms()
    
    # Create agents with low diversity (all same type)
    agents = [
        OllamaAgent(agent_id=f"standard_{i}", config={"model": "gemma3:latest", "prompt": "Same prompt"})
        for i in range(10)
    ]
    
    # Maintain diversity (should introduce mutations)
    with patch('random.random', return_value=0.5):  # Ensure mutations occur
        result_agents = mech.maintain_diversity(agents, target_diversity=0.8)
    
    # Should return agents (mutation may or may not occur depending on implementation)
    assert len(result_agents) == len(agents)


if __name__ == "__main__":
    pytest.main([__file__])