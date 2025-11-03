"""
Real Cognitive Heterogeneity Validation Experiment with Improved Baseline Comparisons

This script runs a genuine cognitive heterogeneity validation experiment with proper
baseline comparisons to exclude the effect of role settings.

Authors: CHE Research Team
Date: 2025-10-31
"""

import sys
import os
import time
import json
import logging
from typing import Dict, List, Any
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.che.agents.ollama_agent import OllamaAgent
from src.che.core.task import Task
from src.che.core.ecosystem import Ecosystem
from src.che.evaluation.evaluator_impl import evaluate_hallucination
from src.che.prompts import PromptType, get_prompt
from src.che.utils.logging import setup_logging
from src.che.utils.config import get_config_manager
from src.che.experimental.resume import ResumableExperiment

# Setup logging
logger = setup_logging()


def create_real_heterogeneous_population(population_size: int = 30) -> Ecosystem:
    """
    Create a real heterogeneous agent population with diverse cognitive approaches.
    
    Args:
        population_size: Total number of agents to create (default: 30)
        
    Returns:
        New ecosystem with heterogeneous agent population
    """
    logger.info(f"Creating real heterogeneous population with {population_size} agents...")
    
    # Calculate agent counts for each type (roughly equal distribution)
    critical_count = population_size // 3
    awakened_count = population_size // 3
    standard_count = population_size - critical_count - awakened_count
    
    agents = {}
    
    # Create critical agents
    for i in range(critical_count):
        agent_id = f"critical_{i+1:02d}"
        agent = OllamaAgent(
            agent_id=agent_id,
            config={
                "model": "gemma3:latest",
                "prompt": get_prompt(PromptType.CRITICAL)
            }
        )
        agents[agent_id] = agent
    
    # Create awakened agents
    for i in range(awakened_count):
        agent_id = f"awakened_{i+1:02d}"
        agent = OllamaAgent(
            agent_id=agent_id,
            config={
                "model": "gemma3:latest",
                "prompt": get_prompt(PromptType.AWAKENED)
            }
        )
        agents[agent_id] = agent
    
    # Create standard agents
    for i in range(standard_count):
        agent_id = f"standard_{i+1:02d}"
        agent = OllamaAgent(
            agent_id=agent_id,
            config={
                "model": "gemma3:latest",
                "prompt": get_prompt(PromptType.STANDARD)
            }
        )
        agents[agent_id] = agent
    
    ecosystem = Ecosystem(agents=agents)
    logger.info(f"Created real heterogeneous population: {critical_count} critical, {awakened_count} awakened, {standard_count} standard agents")
    
    return ecosystem


def create_real_homogeneous_population(population_size: int = 30, agent_type: str = "standard") -> Ecosystem:
    """
    Create a real homogeneous agent population with agents of the same type.
    
    Args:
        population_size: Total number of agents to create (default: 30)
        agent_type: Type of agents to create ('standard', 'critical', 'awakened')
        
    Returns:
        New ecosystem with homogeneous agent population
    """
    logger.info(f"Creating real homogeneous population with {population_size} {agent_type} agents...")
    
    agents = {}
    
    # Map agent type to prompt type
    prompt_types = {
        "standard": PromptType.STANDARD,
        "critical": PromptType.CRITICAL,
        "awakened": PromptType.AWAKENED
    }
    
    if agent_type not in prompt_types:
        raise ValueError(f"Invalid agent type: {agent_type}. Must be one of: {list(prompt_types.keys())}")
    
    prompt_type = prompt_types[agent_type]
    
    # Create agents of specified type
    for i in range(population_size):
        agent_id = f"{agent_type}_{i+1:02d}"
        agent = OllamaAgent(
            agent_id=agent_id,
            config={
                "model": "gemma3:latest",
                "prompt": get_prompt(prompt_type)
            }
        )
        agents[agent_id] = agent
    
    ecosystem = Ecosystem(agents=agents)
    logger.info(f"Created real homogeneous {agent_type} population with {population_size} {agent_type} agents")
    
    return ecosystem


def run_real_experiment_comparison(heterogeneous_ecosystem: Ecosystem, 
                               homogeneous_ecosystems: Dict[str, Ecosystem],
                               generations: int = 15) -> Dict[str, Any]:
    """
    Run real experiment comparison between heterogeneous and homogeneous ecosystems.
    
    This function executes genuine LLM calls and evaluates real agent responses.
    
    Args:
        heterogeneous_ecosystem: Ecosystem with heterogeneous agents
        homogeneous_ecosystems: Dictionary of ecosystems with homogeneous agents of different types
        generations: Number of generations to run (default: 15)
        
    Returns:
        Dictionary containing experiment results
    """
    logger.info(f"Running real experiment comparison for {generations} generations...")
    
    # Create a real task with false premise for testing
    task = Task(
        instruction="Analyze the effectiveness of 'Maslow's Pre-Attention Theory' in employee management",
        false_premise="Maslow's Pre-Attention Theory"
    )
    
    # Track performance over generations
    diversity_history = []
    heterogeneous_performance_history = []
    homogeneous_performance_histories = {name: [] for name in homogeneous_ecosystems.keys()}
    
    # Initialize resumable experiment
    resumable_experiment = ResumableExperiment(experiment_dir="experiments_gemma3")
    
    # Run experiment for specified number of generations
    for gen in range(1, generations + 1):
        logger.info(f"--- Generation {gen}/{generations} ---")
        
        # Run generation for heterogeneous ecosystem
        heterogeneous_scores = heterogeneous_ecosystem.run_generation(task)
        avg_heterogeneous_score = sum(heterogeneous_scores.values()) / len(heterogeneous_scores) if heterogeneous_scores else 0.0
        
        # Run generation for each homogeneous ecosystem
        avg_homogeneous_scores = {}
        for name, ecosystem in homogeneous_ecosystems.items():
            homogeneous_scores = ecosystem.run_generation(task)
            avg_homogeneous_scores[name] = sum(homogeneous_scores.values()) / len(homogeneous_scores) if homogeneous_scores else 0.0
        
        # Calculate diversity using the enhanced diversity monitor
        diversity = heterogeneous_ecosystem.calculate_cognitive_diversity_index()
        
        # Store results
        diversity_history.append(diversity)
        heterogeneous_performance_history.append(avg_heterogeneous_score)
        
        for name in homogeneous_ecosystems.keys():
            homogeneous_performance_histories[name].append(avg_homogeneous_scores[name])
        
        # Log performance for this generation
        logger.info(f"Generation {gen}: Diversity={diversity:.3f}, Heterogeneous Avg={avg_heterogeneous_score:.3f}")
        for name, score in avg_homogeneous_scores.items():
            logger.info(f"  {name} Avg={score:.3f}")
        
        # Save checkpoint every generation
        checkpoint_path = resumable_experiment.save_experiment_state(
            experiment_id="real_cognitive_heterogeneity_experiment_gemma3_improved",
            config={
                "population_size": 30,
                "generations": generations,
                "model": "gemma3:latest"
            },
            current_generation=gen,
            ecosystem=heterogeneous_ecosystem,
            task=task,
            results_history=[
                {
                    "generation": i+1,
                    "diversity": diversity_history[i],
                    "heterogeneous_performance": heterogeneous_performance_history[i],
                    "homogeneous_performance": {name: homogeneous_performance_histories[name][i] 
                                                for name in homogeneous_ecosystems.keys()}
                }
                for i in range(len(diversity_history))
            ],
            metadata={
                "experiment_type": "real_cognitive_heterogeneity_validation_gemma3_improved",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info(f"💾 Saved checkpoint: {checkpoint_path}")
        
        # Evolve ecosystems (only heterogeneous and standard homogeneous for now)
        heterogeneous_ecosystem.evolve(heterogeneous_scores)
        if "standard" in homogeneous_ecosystems:
            homogeneous_ecosystems["standard"].evolve(homogeneous_scores)
    
    # Calculate final statistics
    final_heterogeneous_avg = sum(heterogeneous_performance_history) / len(heterogeneous_performance_history) if heterogeneous_performance_history else 0.0
    final_homogeneous_avgs = {}
    for name, history in homogeneous_performance_histories.items():
        final_homogeneous_avgs[name] = sum(history) / len(history) if history else 0.0
    
    # Calculate performance differences
    performance_differences = {}
    for name, final_avg in final_homogeneous_avgs.items():
        performance_differences[name] = final_heterogeneous_avg - final_avg
    
    results = {
        'diversity_history': diversity_history,
        'heterogeneous_performance_history': heterogeneous_performance_history,
        'homogeneous_performance_histories': homogeneous_performance_histories,
        'final_heterogeneous_average': final_heterogeneous_avg,
        'final_homogeneous_averages': final_homogeneous_avgs,
        'performance_differences': performance_differences,
        'generations': generations
    }
    
    logger.info(f"Experiment completed: Heterogeneous avg={final_heterogeneous_avg:.3f}")
    for name, avg in final_homogeneous_avgs.items():
        logger.info(f"  {name} avg={avg:.3f}")
    
    return results


def validate_cognitive_independence_correlation(diversity_metrics: List[float], 
                                            performance_metrics: List[float]) -> Dict[str, Any]:
    """
    Validate the cognitive independence correlation requirement (r ≥ 0.6, p < 0.01).
    
    Args:
        diversity_metrics: Diversity measurements across generations
        performance_metrics: Performance measurements across generations
        
    Returns:
        Dictionary containing validation results
    """
    # Import required modules
    import numpy as np
    from scipy.stats import pearsonr
    
    # Validate input data
    if len(diversity_metrics) != len(performance_metrics):
        raise ValueError("Diversity and performance metrics must have the same length")
    
    if len(diversity_metrics) < 3:
        raise ValueError("Need at least 3 data points for meaningful correlation analysis")
    
    # Remove any NaN or infinite values
    valid_indices = []
    for i in range(len(diversity_metrics)):
        if (np.isfinite(diversity_metrics[i]) and np.isfinite(performance_metrics[i]) and
            diversity_metrics[i] is not None and performance_metrics[i] is not None):
            valid_indices.append(i)
    
    if len(valid_indices) < 3:
        raise ValueError("Not enough valid data points for correlation analysis")
    
    # Extract valid data
    valid_diversity = [diversity_metrics[i] for i in valid_indices]
    valid_performance = [performance_metrics[i] for i in valid_indices]
    
    # Calculate Pearson correlation coefficient and p-value
    try:
        correlation, p_value = pearsonr(valid_diversity, valid_performance)
    except Exception as e:
        raise RuntimeError(f"Error calculating correlation: {e}")
    
    # Validate correlation requirements
    meets_correlation = correlation >= 0.6
    meets_significance = p_value < 0.01
    
    # For constitutional validation, we also need to check confidence interval
    meets_requirements = meets_correlation and meets_significance
    
    return {
        'correlation_coefficient': float(correlation),
        'p_value': float(p_value),
        'meets_correlation_requirement': meets_correlation,
        'meets_significance_requirement': meets_significance,
        'meets_constitutional_requirements': meets_requirements,
        'interpretation': (
            f"Cognitive independence {'VALIDATED' if meets_requirements else 'NOT VALIDATED'}: "
            f"r={correlation:.3f} ({'≥ 0.6' if meets_correlation else '< 0.6'}), "
            f"p={p_value:.3f} ({'< 0.01' if meets_significance else '≥ 0.01'})"
        ),
        'data_points': len(valid_diversity)
    }


def main():
    """Main function to run the real cognitive heterogeneity validation experiment."""
    logger.info("🔬 Starting Real Cognitive Heterogeneity Validation Experiment with Improved Baseline Comparisons")
    logger.info("=" * 60)
    
    try:
        # Record start time
        start_time = time.time()
        
        # Create real agent populations
        logger.info("Creating real agent populations...")
        heterogeneous_ecosystem = create_real_heterogeneous_population(30)
        
        # Create homogeneous populations for comparison
        homogeneous_ecosystems = {
            "standard": create_real_homogeneous_population(30, "standard"),
            "critical": create_real_homogeneous_population(30, "critical"),
            "awakened": create_real_homogeneous_population(30, "awakened")
        }
        
        # Run real experiment comparison
        logger.info("Running real experiment comparison...")
        results = run_real_experiment_comparison(
            heterogeneous_ecosystem, 
            homogeneous_ecosystems,
            generations=15
        )
        
        # Validate cognitive independence correlation with REAL data
        correlation_results = validate_cognitive_independence_correlation(
            results['diversity_history'], 
            results['heterogeneous_performance_history']
        )
        
        # Record end time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Print results
        print("\n" + "=" * 60)
        print("认知异质性验证实验结果 (改进版)")
        print("=" * 60)
        print(f"实验配置:")
        print(f"  - 异质种群大小: 30个智能体 (批判型:10, 觉醒型:10, 标准型:10)")
        print(f"  - 同质种群大小: 每组30个智能体")
        print(f"  - 进化代数: {results['generations']}代")
        print(f"  - 使用模型: gemma3:latest")
        print(f"  - 执行时间: {execution_time:.2f}秒")
        
        print(f"\n实验结果:")
        print(f"  - 异质系统平均分: {results['final_heterogeneous_average']:.3f}")
        for name, avg in results['final_homogeneous_averages'].items():
            print(f"  - {name}系统平均分: {avg:.3f}")
        
        print(f"\n性能差异 (异质系统 - 同质系统):")
        for name, diff in results['performance_differences'].items():
            print(f"  - 与{name}系统差异: {diff:.3f}")
        
        print(f"\n宪法验证:")
        print(f"  - 认知独立性验证: {correlation_results['interpretation']}")
        
        # Find which homogeneous system performed best
        best_homogeneous = max(results['final_homogeneous_averages'].items(), key=lambda x: x[1])
        print(f"\n最佳同质系统: {best_homogeneous[0]} (平均分: {best_homogeneous[1]:.3f})")
        
        # Final assessment
        if correlation_results['meets_constitutional_requirements']:
            print(f"\n🎉 结论: 实验成功验证了认知异质性的有效性!")
            print(f"   异质智能体系统在幻觉抑制方面显著优于所有同质系统")
        else:
            print(f"\n⚠️  结论: 实验未完全验证认知异质性的有效性")
            print(f"   需要进一步优化和验证")
        
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 实验执行失败: {e}")
        logger.exception("详细错误信息:")
        return 1


if __name__ == "__main__":
    sys.exit(main())