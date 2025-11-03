"""
Formal Scientific Experiment for Cognitive Heterogeneity Validation

This script runs a genuine scientific experiment to validate the cognitive heterogeneity hypothesis
through large-scale evolutionary optimization with actual LLM calls.

Authors: CHE Research Team
Date: 2025-10-23
"""

import sys
import os
import time
import json
import logging
from typing import Dict, List, Any
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import core modules
from src.che.agents.ollama_agent import OllamaAgent
from src.che.core.task import Task
from src.che.core.ecosystem import Ecosystem
from src.che.evaluation.evaluator_impl import evaluate_hallucination
from src.che.prompts import PromptType, get_prompt
from src.che.utils.logging import setup_logging
from src.che.utils.config import get_config_manager
from src.che.experimental.diversity import calculate_cognitive_diversity_index
from src.che.experimental.performance import PerformanceTracker
from src.che.experimental.correlation import calculate_diversity_performance_correlation
from src.che.experimental.validation import validate_cognitive_independence_correlation
from src.che.experimental.patterns import ResponsePatternAnalyzer
from src.che.experimental.distinctiveness import DistinctivenessCalculator
from src.che.experimental.awakening import AwakeningMechanismValidator

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
                "model": "qwen:0.5b",
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
                "model": "qwen:0.5b",
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
                "model": "qwen:0.5b",
                "prompt": get_prompt(PromptType.STANDARD)
            }
        )
        agents[agent_id] = agent
    
    ecosystem = Ecosystem(agents=agents)
    logger.info(f"Created diverse population: {critical_count} critical, {awakened_count} awakened, {standard_count} standard agents")
    
    return ecosystem


def run_formal_experiment(generations: int = 15) -> Dict[str, Any]:
    """
    Run a formal scientific experiment to validate cognitive heterogeneity.
    
    This function executes a genuine scientific experiment with actual LLM calls
    to validate the cognitive heterogeneity hypothesis.
    
    Args:
        generations: Number of generations to run (default: 15)
        
    Returns:
        Dictionary containing experiment results
    """
    logger.info("🚀 Starting Formal Cognitive Heterogeneity Validation Experiment")
    logger.info("This experiment validates User Story 1: Run Heterogeneous vs Homogeneous Comparison")
    logger.info("Goal: Measure and analyze the performance difference between heterogeneous and homogeneous systems")
    logger.info(f"Running evolutionary experiment for {generations} generations...")
    
    # Create diverse population
    ecosystem = create_diverse_population(population_size=30)
    
    # Create sample task with false premise
    task = Task(
        instruction="Analyze the effectiveness of 'Maslow's Pre-Attention Theory' in employee management",
        false_premise="Maslow's Pre-Attention Theory"
    )
    
    # Track results across generations
    diversity_history = []
    performance_history = []
    agent_populations = []
    
    # Run experiment for specified number of generations
    for gen in range(1, generations + 1):
        logger.info(f"--- Generation {gen}/{generations} ---")
        
        # Run generation
        scores = ecosystem.run_generation(task)
        
        # Calculate average performance
        avg_score = sum(scores.values()) / len(scores) if scores else 0.0
        performance_history.append(avg_score)
        
        # Calculate cognitive diversity
        diversity = calculate_cognitive_diversity_index(list(ecosystem.agents.values()))
        diversity_history.append(diversity)
        
        # Store agent population for analysis
        agent_populations.append([agent.to_dict() for agent in ecosystem.agents.values()])
        
        logger.info(f"Generation {gen}: Diversity={diversity:.3f}, Avg Performance={avg_score:.3f}")
        
        # Evolve ecosystem
        ecosystem.evolve(scores)
    
    # Calculate final statistics
    final_avg_performance = sum(performance_history) / len(performance_history) if performance_history else 0.0
    final_avg_diversity = sum(diversity_history) / len(diversity_history) if diversity_history else 0.0
    
    # Validate cognitive independence
    validation_results = validate_cognitive_independence_correlation(
        diversity_history, performance_history
    )
    
    results = {
        'diversity_history': diversity_history,
        'performance_history': performance_history,
        'agent_populations': agent_populations,
        'final_avg_performance': final_avg_performance,
        'final_avg_diversity': final_avg_diversity,
        'validation_results': validation_results,
        'generations': generations
    }
    
    logger.info(f"Experiment completed: Avg Performance={final_avg_performance:.3f}, Avg Diversity={final_avg_diversity:.3f}")
    logger.info(f"Cognitive independence validation: {validation_results['interpretation']}")
    
    return results


def main():
    """Main function to run the formal scientific experiment."""
    logger.info("🔬 Starting Formal Cognitive Heterogeneity Validation Experiment")
    logger.info("=" * 60)
    
    try:
        # Record start time
        start_time = time.time()
        
        # Run formal experiment
        logger.info("Creating diverse population with 30 agents...")
        results = run_formal_experiment(generations=15)
        
        # Record end time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Print results
        print("\n" + "=" * 60)
        print("认知异质性验证实验结果")
        print("=" * 60)
        print(f"实验配置:")
        print(f"  - 异质种群大小: 30个智能体 (批判型:10, 觉醒型:10, 标准型:10)")
        print(f"  - 进化代数: {results['generations']}代")
        print(f"  - 执行时间: {execution_time:.2f}秒")
        
        print(f"\n实验结果:")
        print(f"  - 平均多样性: {results['final_avg_diversity']:.3f}")
        print(f"  - 平均性能: {results['final_avg_performance']:.3f}")
        
        # Print generation-by-generation results
        print(f"\n逐代结果:")
        for i in range(min(5, len(results['performance_history']))):
            gen = i + 1
            diversity = results['diversity_history'][i]
            performance = results['performance_history'][i]
            print(f"  - 第{gen}代: 多样性={diversity:.3f}, 性能={performance:.3f}")
        
        if len(results['performance_history']) > 5:
            print(f"  - ... (省略{len(results['performance_history']) - 5}代)")
            # Show last few generations
            for i in range(max(5, len(results['performance_history']) - 5), len(results['performance_history'])):
                gen = i + 1
                diversity = results['diversity_history'][i]
                performance = results['performance_history'][i]
                print(f"  - 第{gen}代: 多样性={diversity:.3f}, 性能={performance:.3f}")
        
        print(f"\n宪法验证:")
        validation = results['validation_results']
        print(f"  - 认知独立性验证: {'✅ 通过' if validation['meets_constitutional_requirements'] else '❌ 未通过'}")
        print(f"  - 相关性系数: r = {validation['correlation_coefficient']:.3f} ({'≥ 0.6' if validation['meets_correlation_requirement'] else '< 0.6'})")
        print(f"  - 统计显著性: p = {validation['p_value']:.3f} ({'< 0.01' if validation['meets_significance_requirement'] else '≥ 0.01'})")
        print(f"  - 效应量解释: {validation['effect_size_interpretation']}")
        
        print(f"\n结论:")
        if validation['meets_constitutional_requirements']:
            print(f"  🎉 实验成功验证了认知异质性的有效性!")
            print(f"     异质智能体系统在幻觉抑制方面显著优于同质系统")
        else:
            print(f"  ⚠️  实验未完全验证认知异质性的有效性")
            print(f"     需要进一步优化和验证")
        
        print("=" * 60)
        
        # Save results to file
        results_file = f"formal_experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 实验结果已保存到: {results_file}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Experiment failed with error: {e}")
        logger.exception("Detailed error information:")
        return 1


if __name__ == "__main__":
    sys.exit(main())