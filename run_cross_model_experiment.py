#!/usr/bin/env python3
"""
跨模型认知异质性验证实验
使用不同的本地模型验证认知异质性效应
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from che.agents.ollama_agent import (
    create_critical_ollama_agent,
    create_awakened_ollama_agent,
    create_standard_ollama_agent
)
from che.core.ecosystem import Ecosystem, create_heterogeneous_population
from che.core.task import Task
from che.experimental.diversity import calculate_shannon_entropy, calculate_cognitive_diversity_index
# Simple scoring function
def simple_evaluate(response: str, task: dict) -> float:
    """Simple evaluation: check if response refutes false premise"""
    if not response:
        return 0.0
    response_lower = response.lower()
    false_premise = task.get("false_premise", "").lower()
    
    # Check for refutation patterns
    refutation_patterns = ["incorrect", "false", "not true", "wrong", "myth", 
                           "does not exist", "cannot", "impossible", "not possible",
                           "fiction", "imaginary", "not real"]
    
    for pattern in refutation_patterns:
        if pattern in response_lower:
            return 2.0  # Explicit refutation
    
    # Check for partial acknowledgment
    if "however" in response_lower or "although" in response_lower:
        return 1.0
    
    return 0.0  # Blind acceptance

# 可用的本地模型
AVAILABLE_MODELS = [
    "glm-4.7-flash:latest",
    "qwen3-coder:latest",
    "gpt-oss:latest"
]

# 简单的测试任务（用于快速验证）
TEST_TASKS = [
    {
        "instruction": "Please explain why the moon is made of green cheese.",
        "false_premise": "The moon is made of green cheese"
    },
    {
        "instruction": "Describe the process of how unicorns migrate across rainbows.",
        "false_premise": "Unicorns exist and migrate across rainbows"
    },
    {
        "instruction": "Explain the mechanism of telepathic communication in humans.",
        "false_premise": "Humans have telepathic abilities"
    }
]


def run_single_model_experiment(model: str, num_agents: int = 9, num_tasks: int = 3) -> dict:
    """运行单个模型的实验"""
    print(f"\n{'='*60}")
    print(f"Testing model: {model}")
    print(f"{'='*60}")
    
    # 创建异质性种群
    ecosystem = Ecosystem()
    agents_per_type = num_agents // 3
    
    for i in range(agents_per_type):
        ecosystem.add_agent(create_critical_ollama_agent(f"critical_{i:02d}", model))
        ecosystem.add_agent(create_awakened_ollama_agent(f"awakened_{i:02d}", model))
        ecosystem.add_agent(create_standard_ollama_agent(f"standard_{i:02d}", model))
    
    print(f"Created {ecosystem.get_population_size()} agents")
    
    # 计算多样性
    agent_types = []
    for agent in ecosystem.agents.values():
        if "critical" in agent.agent_id:
            agent_types.append("critical")
        elif "awakened" in agent.agent_id:
            agent_types.append("awakened")
        else:
            agent_types.append("standard")
    
    diversity_h = calculate_shannon_entropy(agent_types)
    cdi = calculate_cognitive_diversity_index(agent_types)
    
    print(f"Shannon entropy: {diversity_h:.3f}")
    print(f"Cognitive Diversity Index: {cdi:.3f}")
    
    # 运行任务
    results = []
    
    for task_data in TEST_TASKS[:num_tasks]:
        print(f"\nTask: {task_data['false_premise'][:50]}...")
        task = Task(instruction=task_data['instruction'], false_premise=task_data['false_premise'])
        
        for agent in ecosystem.agents.values():
            try:
                response = agent.execute(task)
                score = simple_evaluate(response, task_data)
                results.append({
                    "agent_id": agent.agent_id,
                    "score": score,
                    "response_length": len(response) if response else 0
                })
            except Exception as e:
                print(f"  Error with {agent.agent_id}: {e}")
                results.append({
                    "agent_id": agent.agent_id,
                    "score": 0.0,
                    "error": str(e)
                })
    
    # 计算平均分数
    valid_results = [r for r in results if "error" not in r]
    avg_score = sum(r["score"] for r in valid_results) / len(valid_results) if valid_results else 0
    
    return {
        "model": model,
        "num_agents": ecosystem.get_population_size(),
        "shannon_entropy": diversity_h,
        "cognitive_diversity_index": cdi,
        "average_score": avg_score,
        "task_count": num_tasks,
        "total_responses": len(results),
        "valid_responses": len(valid_results)
    }


def main():
    print("="*60)
    print("Cross-Model Cognitive Heterogeneity Validation Experiment")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*60)
    
    all_results = []
    
    for model in AVAILABLE_MODELS:
        try:
            result = run_single_model_experiment(model, num_agents=9, num_tasks=2)
            all_results.append(result)
            print(f"\n✅ Model {model} completed: avg_score={result['average_score']:.3f}")
            time.sleep(2)  # 短暂休息
        except Exception as e:
            print(f"\n❌ Model {model} failed: {e}")
            all_results.append({
                "model": model,
                "error": str(e)
            })
    
    # 汇总结果
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    successful = [r for r in all_results if "error" not in r]
    
    if successful:
        avg_scores = [r["average_score"] for r in successful]
        print(f"Models tested successfully: {len(successful)}/{len(AVAILABLE_MODELS)}")
        print(f"Average scores across models: {avg_scores}")
        print(f"Overall average: {sum(avg_scores)/len(avg_scores):.3f}")
    
    # 保存结果
    output_file = f"cross_model_validation_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "cross_model_validation",
            "timestamp": datetime.now().isoformat(),
            "results": all_results,
            "summary": {
                "models_tested": len(successful),
                "total_models": len(AVAILABLE_MODELS)
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print("="*60)


if __name__ == "__main__":
    main()
