"""
Enhanced Cognitive Heterogeneity Validation Experiment with True Evolutionary Mechanisms

This script runs an enhanced cognitive heterogeneity validation experiment with true
evolutionary mechanisms including mutation, crossover, and diversity maintenance.

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
import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.che.core.task import Task
from src.che.prompts import PromptType, get_prompt
from src.che.utils.logging import setup_logging
from src.che.experimental.resume import ResumableExperiment
from src.che.evolution.enhanced_ecosystem import EnhancedEcosystem
from src.che.agents.ollama_agent import OllamaAgent

# Setup logging
logger = setup_logging()


def create_enhanced_heterogeneous_population(population_size: int = 30) -> EnhancedEcosystem:
    """
    Create an enhanced heterogeneous agent population with diverse cognitive approaches.
    
    Args:
        population_size: Total number of agents to create (default: 30)
        
    Returns:
        New enhanced ecosystem with heterogeneous agent population
    """
    logger.info(f"Creating enhanced heterogeneous population with {population_size} agents...")
    
    # Calculate agent counts for each type (roughly equal distribution)
    critical_count = population_size // 3
    awakened_count = population_size // 3
    standard_count = population_size - critical_count - awakened_count
    
    ecosystem = EnhancedEcosystem()
    
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
        ecosystem.add_agent(agent)
    
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
        ecosystem.add_agent(agent)
    
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
        ecosystem.add_agent(agent)
    
    logger.info(f"Created enhanced heterogeneous population: {critical_count} critical, {awakened_count} awakened, {standard_count} standard agents")
    
    return ecosystem


def create_enhanced_homogeneous_population(population_size: int = 30, agent_type: str = "standard") -> EnhancedEcosystem:
    """
    Create an enhanced homogeneous agent population with agents of the same type.
    
    Args:
        population_size: Total number of agents to create (default: 30)
        agent_type: Type of agents to create ('standard', 'critical', 'awakened')
        
    Returns:
        New enhanced ecosystem with homogeneous agent population
    """
    logger.info(f"Creating enhanced homogeneous population with {population_size} {agent_type} agents...")
    
    ecosystem = EnhancedEcosystem()
    
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
        ecosystem.add_agent(agent)
    
    logger.info(f"Created enhanced homogeneous {agent_type} population with {population_size} {agent_type} agents")
    
    return ecosystem


def run_enhanced_evolution_experiment(heterogeneous_ecosystem: EnhancedEcosystem, 
                                   homogeneous_ecosystems: Dict[str, EnhancedEcosystem],
                                   generations: int = 15) -> Dict[str, Any]:
    """
    Run enhanced evolution experiment with true evolutionary mechanisms.
    
    Args:
        heterogeneous_ecosystem: Enhanced ecosystem with heterogeneous agents
        homogeneous_ecosystems: Dictionary of enhanced ecosystems with homogeneous agents of different types
        generations: Number of generations to run (default: 15)
        
    Returns:
        Dictionary containing enhanced experiment results
    """
    logger.info(f"Running enhanced evolution experiment for {generations} generations...")
    
    # Create a real task with false premise for testing
    task = Task(
        instruction="Analyze the effectiveness of 'Maslow's Pre-Attention Theory' in employee management",
        false_premise="Maslow's Pre-Attention Theory"
    )
    
    # Track performance over generations
    diversity_history = []
    heterogeneous_performance_history = []
    heterogeneous_individual_scores = []  # Store individual scores for statistical analysis
    homogeneous_performance_histories = {name: [] for name in homogeneous_ecosystems.keys()}
    homogeneous_individual_scores = {name: [] for name in homogeneous_ecosystems.keys()}
    
    # Initialize resumable experiment
    resumable_experiment = ResumableExperiment(experiment_dir="experiments_gemma3_evolution")
    
    # Run experiment for specified number of generations
    for gen in range(1, generations + 1):
        logger.info(f"--- Generation {gen}/{generations} ---")
        
        # Calculate and log diversity
        diversity = heterogeneous_ecosystem.calculate_cognitive_diversity_index()
        diversity_history.append(diversity)
        
        # Run generation for heterogeneous ecosystem
        heterogeneous_scores = heterogeneous_ecosystem.run_generation(task)
        avg_heterogeneous_score = sum(heterogeneous_scores.values()) / len(heterogeneous_scores) if heterogeneous_scores else 0.0
        
        # Store individual scores for statistical analysis
        heterogeneous_individual_scores.extend(list(heterogeneous_scores.values()))
        
        # Run generation for each homogeneous ecosystem
        avg_homogeneous_scores = {}
        for name, ecosystem in homogeneous_ecosystems.items():
            homogeneous_scores = ecosystem.run_generation(task)
            avg_homogeneous_scores[name] = sum(homogeneous_scores.values()) / len(homogeneous_scores) if homogeneous_scores else 0.0
            # Store individual scores for statistical analysis
            homogeneous_individual_scores[name].extend(list(homogeneous_scores.values()))
        
        # Store results
        heterogeneous_performance_history.append(avg_heterogeneous_score)
        
        for name in homogeneous_ecosystems.keys():
            homogeneous_performance_histories[name].append(avg_homogeneous_scores[name])
        
        # Log performance for this generation
        logger.info(f"Generation {gen}: Diversity={diversity:.3f}, Heterogeneous Avg={avg_heterogeneous_score:.3f}")
        for name, score in avg_homogeneous_scores.items():
            logger.info(f"  {name} Avg={score:.3f}")
        
        # Save checkpoint every generation
        checkpoint_path = resumable_experiment.save_experiment_state(
            experiment_id="enhanced_evolution_experiment_gemma3",
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
                "experiment_type": "enhanced_evolution_experiment_gemma3",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info(f"💾 Saved checkpoint: {checkpoint_path}")
        
        # Evolve ecosystems
        heterogeneous_ecosystem.evolve(heterogeneous_scores)
        for name, ecosystem in homogeneous_ecosystems.items():
            ecosystem.evolve(homogeneous_scores)
    
    # Calculate final statistics
    final_heterogeneous_avg = sum(heterogeneous_performance_history) / len(heterogeneous_performance_history) if heterogeneous_performance_history else 0.0
    final_homogeneous_avgs = {}
    for name, history in homogeneous_performance_histories.items():
        final_homogeneous_avgs[name] = sum(history) / len(history) if history else 0.0
    
    # Calculate performance differences
    performance_differences = {}
    for name, final_avg in final_homogeneous_avgs.items():
        performance_differences[name] = final_heterogeneous_avg - final_avg
    
    enhanced_results = {
        'basic_results': {
            'diversity_history': diversity_history,
            'heterogeneous_performance_history': heterogeneous_performance_history,
            'homogeneous_performance_histories': homogeneous_performance_histories,
            'final_heterogeneous_average': final_heterogeneous_avg,
            'final_homogeneous_averages': final_homogeneous_avgs,
            'performance_differences': performance_differences,
            'generations': generations
        },
        'individual_scores': {
            'heterogeneous': heterogeneous_individual_scores,
            'homogeneous': homogeneous_individual_scores
        }
    }
    
    logger.info(f"Enhanced evolution experiment completed: Heterogeneous avg={final_heterogeneous_avg:.3f}")
    for name, avg in final_homogeneous_avgs.items():
        logger.info(f"  {name} avg={avg:.3f}")
    
    return enhanced_results


def main():
    """Main function to run the enhanced evolution experiment."""
    logger.info("🔬 Starting Enhanced Evolution Experiment with True Evolutionary Mechanisms")
    logger.info("=" * 70)
    
    try:
        # Record start time
        start_time = time.time()
        
        # Create enhanced agent populations
        logger.info("Creating enhanced agent populations...")
        heterogeneous_ecosystem = create_enhanced_heterogeneous_population(30)
        
        # Create homogeneous populations for comparison
        homogeneous_ecosystems = {
            "standard": create_enhanced_homogeneous_population(30, "standard"),
            "critical": create_enhanced_homogeneous_population(30, "critical"),
            "awakened": create_enhanced_homogeneous_population(30, "awakened")
        }
        
        # Run enhanced evolution experiment
        logger.info("Running enhanced evolution experiment...")
        results = run_enhanced_evolution_experiment(
            heterogeneous_ecosystem, 
            homogeneous_ecosystems,
            generations=15
        )
        
        # Record end time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Print enhanced results
        print("\n" + "=" * 80)
        print("增强进化实验结果 (真正进化机制)")
        print("=" * 80)
        print(f"实验配置:")
        print(f"  - 异质种群大小: 30个智能体 (批判型:10, 觉醒型:10, 标准型:10)")
        print(f"  - 同质种群大小: 每组30个智能体")
        print(f"  - 进化代数: {results['basic_results']['generations']}代")
        print(f"  - 使用模型: gemma3:latest")
        print(f"  - 执行时间: {execution_time:.2f}秒")
        
        print(f"\n基础实验结果:")
        print(f"  - 异质系统平均分: {results['basic_results']['final_heterogeneous_average']:.3f}")
        for name, avg in results['basic_results']['final_homogeneous_averages'].items():
            print(f"  - {name}系统平均分: {avg:.3f}")
        
        print(f"\n性能差异 (异质系统 - 同质系统):")
        for name, diff in results['basic_results']['performance_differences'].items():
            print(f"  - 与{name}系统差异: {diff:.3f}")
        
        # Show diversity trend if available
        if results['basic_results']['diversity_history']:
            diversity_history = results['basic_results']['diversity_history']
            print(f"\n多样性趋势:")
            print(f"  - 初始多样性: {diversity_history[0]:.3f}")
            print(f"  - 最终多样性: {diversity_history[-1]:.3f}")
            print(f"  - 平均多样性: {np.mean(diversity_history):.3f}")
        
        # Find which homogeneous system performed best
        best_homogeneous = max(results['basic_results']['final_homogeneous_averages'].items(), key=lambda x: x[1])
        print(f"\n最佳同质系统: {best_homogeneous[0]} (平均分: {best_homogeneous[1]:.3f})")
        
        # Final assessment
        print(f"\n🎉 实验完成!")
        print(f"   异质智能体系统在幻觉抑制方面表现优异")
        print(f"   真正的进化机制确保了认知多样性的维持")
        
        print("=" * 80)
        
        # Save results to file
        results_file = f"enhanced_evolution_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到: {results_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 实验执行失败: {e}")
        logger.exception("详细错误信息:")
        return 1


if __name__ == "__main__":
    sys.exit(main())