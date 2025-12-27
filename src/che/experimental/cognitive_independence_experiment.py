"""
Main Experiment Script for User Story 2: Analyze Cognitive Independence Correlation

This script demonstrates the complete implementation of User Story 2, which allows researchers
to measure and analyze the correlation between cognitive diversity and performance improvement,
specifically validating that r ≥ 0.6 as required by the project constitution.

Authors: CHE Research Team
Date: 2025-10-19
"""

import sys
import os
import logging
import time
import numpy as np
from typing import Dict, List, Any

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from che.core.ecosystem import Ecosystem
from che.core.task import Task
from che.agents.ollama_agent import OllamaAgent
from che.prompts import PromptType, get_prompt
from che.utils.logging import setup_logging
from che.utils.config import get_config_manager
from che.experimental.diversity import calculate_cognitive_diversity_index
from che.experimental.performance import PerformanceTracker
from che.experimental.correlation import calculate_diversity_performance_correlation
from che.experimental.validation import validate_cognitive_independence_correlation

# Setup logging
logger = setup_logging()


def create_diverse_population(population_size: int = 30) -> Ecosystem:
    """
    Create a diverse population of agents with different cognitive types.

    Args:
        population_size: Total number of agents to create (default: 30)

    Returns:
        New ecosystem with diverse agent population
    """
    logger.info(f"Creating diverse population with {population_size} agents...")

    # Calculate agent counts for each type
    critical_count = population_size // 3
    awakened_count = population_size // 3
    standard_count = population_size - critical_count - awakened_count

    agents = []

    # Use local models that are available
    # For this example, we'll use gemma:2b, qwen:7b-chat, and llama3:latest
    # which are available on the local system
    available_models = ["gemma:2b", "qwen:7b-chat", "llama3:latest"]

    # Create critical agents
    for i in range(critical_count):
        agent_id = f"critical_{i+1:02d}"
        model_index = i % len(available_models)
        agent = OllamaAgent(
            agent_id=agent_id,
            config={
                "model": available_models[model_index],
                "prompt": get_prompt(PromptType.CRITICAL)
            }
        )
        agents.append(agent)

    # Create awakened agents
    for i in range(awakened_count):
        agent_id = f"awakened_{i+1:02d}"
        model_index = i % len(available_models)
        agent = OllamaAgent(
            agent_id=agent_id,
            config={
                "model": available_models[model_index],
                "prompt": get_prompt(PromptType.AWAKENED)
            }
        )
        agents.append(agent)

    # Create standard agents
    for i in range(standard_count):
        agent_id = f"standard_{i+1:02d}"
        model_index = i % len(available_models)
        agent = OllamaAgent(
            agent_id=agent_id,
            config={
                "model": available_models[model_index],
                "prompt": get_prompt(PromptType.STANDARD)
            }
        )
        agents.append(agent)

    logger.info(f"Created diverse population: {critical_count} critical, {awakened_count} awakened, {standard_count} standard agents")

    ecosystem = Ecosystem()
    for agent in agents:
        ecosystem.add_agent(agent)

    return ecosystem


def run_evolution_experiment(
    ecosystem: Ecosystem,
    generations: int = 15,
    population_size: int = 30
) -> Dict[str, Any]:
    """
    Run an evolutionary experiment to measure cognitive diversity and performance.
    
    Args:
        ecosystem: The ecosystem to evolve
        generations: Number of generations to run (default: 15)
        population_size: Target population size (default: 30)
        
    Returns:
        Dictionary containing experiment results
    """
    logger.info(f"Running evolutionary experiment for {generations} generations...")
    
    # Create performance tracker
    performance_tracker = PerformanceTracker()
    
    # Track diversity and performance metrics across generations
    diversity_history: List[float] = []
    performance_history: List[float] = []
    
    # Create a sample task with false premise
    task = Task(
        instruction="Analyze the effectiveness of 'Maslow's Pre-Attention Theory' in employee management",
        false_premise="Maslow's Pre-Attention Theory"
    )
    
    # Run evolution for specified number of generations
    for gen in range(generations):
        logger.info(f"--- Generation {gen+1}/{generations} ---")
        
        # Calculate current cognitive diversity
        current_diversity = ecosystem.calculate_cognitive_diversity_index()
        diversity_history.append(current_diversity)
        
        # Run generation and get scores
        scores = ecosystem.run_generation(task)
        
        # Calculate average performance
        avg_performance = sum(scores.values()) / len(scores) if scores else 0.0
        performance_history.append(avg_performance)
        
        # Record performance metrics
        performance_tracker.record_generation_performance(
            agent_scores=scores,
            generation=gen,
            metadata={
                'diversity': current_diversity,
                'population_size': ecosystem.get_population_size()
            }
        )
        
        logger.info(f"Generation {gen+1}: Diversity={current_diversity:.3f}, Avg Performance={avg_performance:.3f}")
        
        # Evolve population
        ecosystem.evolve(scores)
    
    # Calculate final correlation between diversity and performance
    correlation_results = calculate_diversity_performance_correlation(
        diversity_history, performance_history
    )
    
    # Validate cognitive independence requirement (r ≥ 0.6)
    validation_results = validate_cognitive_independence_correlation(
        diversity_history, performance_history
    )
    
    return {
        'diversity_history': diversity_history,
        'performance_history': performance_history,
        'correlation_results': correlation_results,
        'validation_results': validation_results,
        'performance_metrics': performance_tracker.get_performance_summary()
    }


def main():
    """Main function to run the cognitive independence correlation analysis experiment."""
    logger.info("🚀 Starting Cognitive Independence Correlation Analysis Experiment")
    logger.info("This experiment validates User Story 2: Analyze Cognitive Independence Correlation")
    logger.info("Goal: Measure and analyze the correlation between cognitive diversity and performance improvement")
    logger.info("Constitutional Requirement: r ≥ 0.6 as required by the project constitution")
    
    try:
        # Create diverse population
        ecosystem = create_diverse_population(population_size=30)
        
        # Run evolutionary experiment
        start_time = time.time()
        results = run_evolution_experiment(
            ecosystem=ecosystem,
            generations=15,
            population_size=30
        )
        end_time = time.time()
        
        # Extract results
        diversity_history = results['diversity_history']
        performance_history = results['performance_history']
        correlation_results = results['correlation_results']
        validation_results = results['validation_results']
        performance_metrics = results['performance_metrics']
        
        # Display results
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT RESULTS")
        logger.info("="*60)
        
        logger.info(f"\n📈 Diversity History:")
        for i, div in enumerate(diversity_history):
            logger.info(f"  Generation {i+1:2d}: {div:.3f}")
        
        logger.info(f"\n📊 Performance History:")
        for i, perf in enumerate(performance_history):
            logger.info(f"  Generation {i+1:2d}: {perf:.3f}")
        
        logger.info(f"\n🔗 Correlation Analysis:")
        logger.info(f"  Pearson r: {correlation_results['pearson_r']:.3f}")
        logger.info(f"  P-value: {correlation_results['pearson_p_value']:.3f}")
        logger.info(f"  Spearman ρ: {correlation_results['spearman_rho']:.3f}")
        logger.info(f"  Kendall τ: {correlation_results['kendall_tau']:.3f}")
        
        logger.info(f"\n✅ Constitutional Validation:")
        logger.info(f"  Meets correlation requirement (r ≥ 0.6): {validation_results['meets_correlation_requirement']}")
        logger.info(f"  Meets significance requirement (p < 0.01): {validation_results['meets_significance_requirement']}")
        logger.info(f"  Meets constitutional requirements: {validation_results['meets_constitutional_requirements']}")
        logger.info(f"  Effect size interpretation: {correlation_results['interpretation']}")
        
        logger.info(f"\n⏱️  Performance Metrics:")
        logger.info(f"  Total execution time: {end_time - start_time:.2f} seconds")
        logger.info(f"  Generations analyzed: {performance_metrics['total_generations']}")
        logger.info(f"  Overall average performance: {performance_metrics['overall_average']:.3f}")
        logger.info(f"  Best generation: {performance_metrics['best_generation']} (score: {performance_metrics['best_score']:.3f})")
        logger.info(f"  Worst generation: {performance_metrics['worst_generation']} (score: {performance_metrics['worst_score']:.3f})")
        
        # Final assessment
        if validation_results['meets_constitutional_requirements']:
            logger.info("\n🎉 SUCCESS: Cognitive independence correlation meets constitutional requirements!")
            logger.info("   The experiment demonstrates that cognitive diversity correlates with performance improvement (r ≥ 0.6)")
        else:
            logger.info("\n⚠️  NOTICE: Cognitive independence correlation does not meet constitutional requirements")
            logger.info("   Further experimentation may be needed to achieve r ≥ 0.6")
        
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("="*60)
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Experiment failed with error: {e}")
        logger.exception("Detailed error information:")
        return None


if __name__ == "__main__":
    main()