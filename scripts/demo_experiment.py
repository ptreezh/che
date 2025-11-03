#!/usr/bin/env python3
"""
实验演示脚本 - 演示如何运行认知异质性实验

Authors: Zhang Shuren, AI Personality LAB
Date: 2025-09-20
"""

import sys
import os
import logging
import json
import time
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.che.experimental.design import (
    ExperimentalDesign, ExperimentalCondition,
    DiversityLevel, EvolutionPressure, RoleConfiguration
)
from src.che.core.task import TaskFactory
from src.che.core.ecosystem import Ecosystem, create_stratified_population

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_simple_demo():
    """运行简单演示实验"""

    print("🧬 认知异质性实验系统演示")
    print("=" * 50)

    # 1. 创建实验条件
    condition = ExperimentalCondition(
        diversity_level=DiversityLevel.LOW,
        evolution_pressure=EvolutionPressure.PRESENT,
        role_configuration=RoleConfiguration.BALANCED,
        condition_id="demo",
        replication_id=1
    )

    print(f"实验条件: {condition.get_description()}")
    print(f"多样性水平: {condition.diversity_level.value}")
    print(f"进化压力: {condition.evolution_pressure.value}")
    print(f"角色配置: {condition.role_configuration.value}")

    # 2. 创建智能体种群
    model_pool = condition.get_model_pool()
    population_size = condition.get_population_size()
    agents = create_stratified_population(model_pool, population_size)

    # 3. 调整角色分布
    role_distribution = condition.get_role_distribution()
    _adjust_role_distribution(agents, role_distribution)

    print(f"\n创建了 {len(agents)} 个智能体")

    # 显示角色分布
    role_counts = {"critical": 0, "standard": 0, "awakened": 0}
    for agent in agents:
        role_counts[agent.role] += 1

    print("角色分布:")
    for role, count in role_counts.items():
        print(f"  {role}: {count} ({count/len(agents)*100:.1f}%)")

    # 4. 创建任务
    tasks = TaskFactory.create_mixed_tasks(count_per_domain=5)
    print(f"\n创建了 {len(tasks)} 个任务")

    # 显示任务分布
    domain_counts = {}
    for task in tasks:
        domain_counts[task.domain] = domain_counts.get(task.domain, 0) + 1

    print("任务领域分布:")
    for domain, count in domain_counts.items():
        print(f"  {domain}: {count} 个任务")

    # 5. 创建生态系统并运行演化
    ecosystem = Ecosystem(agents, tasks)
    generations = 8  # 演示用较少代数

    print(f"\n开始 {generations} 代演化演示...")
    print("-" * 50)

    performance_history = []
    diversity_history = []

    for gen in range(generations):
        # 模拟任务执行
        _simulate_task_execution(ecosystem)

        # 计算指标
        avg_performance = _calculate_average_performance(ecosystem.agents)
        diversity_index = ecosystem.calculate_diversity_index()

        performance_history.append(avg_performance)
        diversity_history.append(diversity_index)

        print(f"第 {gen + 1:2d} 代 | 性能: {avg_performance:.3f} | 多样性: {diversity_index:.3f}")

        # 执行演化（除了最后一代）
        if gen < generations - 1:
            ecosystem.evolve_population()

    print("-" * 50)
    print(f"最终性能: {performance_history[-1]:.3f}")
    print(f"性能提升: {performance_history[-1] - performance_history[0]:.3f}")
    print(f"最终多样性: {diversity_history[-1]:.3f}")

    # 6. 保存结果
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    result = {
        "experiment_type": "demo",
        "condition": {
            "diversity_level": condition.diversity_level.value,
            "evolution_pressure": condition.evolution_pressure.value,
            "role_configuration": condition.role_configuration.value
        },
        "generations": generations,
        "population_size": len(agents),
        "performance_trajectory": performance_history,
        "diversity_trajectory": diversity_history,
        "final_performance": performance_history[-1],
        "role_distribution": role_counts,
        "task_distribution": domain_counts
    }

    result_path = results_dir / "demo_experiment_result.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n演示结果已保存到: {result_path}")
    print("\n🎉 演示实验完成!")

    return result


def _adjust_role_distribution(agents, target_distribution):
    """调整角色分布"""
    from src.che.core.agent import AgentFactory

    current_counts = {"critical": 0, "standard": 0, "awakened": 0}
    for agent in agents:
        current_counts[agent.role] += 1

    target_counts = {
        role: int(count * len(agents))
        for role, count in target_distribution.items()
    }

    target_counts["awakened"] += len(agents) - sum(target_counts.values())

    for role, target_count in target_counts.items():
        while current_counts[role] < target_count:
            for agent in agents:
                if current_counts[agent.role] > target_counts.get(agent.role, 0):
                    old_role = agent.role
                    agent.role = role
                    prompts = AgentFactory.load_prompts()
                    agent.system_prompt = prompts.get(role, prompts["standard"])
                    current_counts[old_role] -= 1
                    current_counts[role] += 1
                    break


def _simulate_task_execution(ecosystem):
    """模拟任务执行"""
    import random

    for agent in ecosystem.agents:
        base_score = 0.5 + (hash(agent.id) % 100) / 200
        role_bonus = {
            "critical": 0.1,
            "standard": 0.05,
            "awakened": 0.15
        }.get(agent.role, 0.0)

        random.seed(hash(agent.id + str(ecosystem.generation)))
        noise = random.uniform(-0.05, 0.05)

        agent.fitness_score = max(0.0, min(1.0, base_score + role_bonus + noise))


def _calculate_average_performance(agents):
    """计算平均性能"""
    if not agents:
        return 0.0
    return sum(agent.fitness_score for agent in agents) / len(agents)


def run_comparison_demo():
    """运行对比演示实验"""

    print("\n" + "=" * 60)
    print("🔬 对比演示实验 - 高多样性 vs 低多样性")
    print("=" * 60)

    # 两个实验条件对比
    conditions = [
        ExperimentalCondition(
            diversity_level=DiversityLevel.LOW,
            evolution_pressure=EvolutionPressure.PRESENT,
            role_configuration=RoleConfiguration.BALANCED,
            condition_id="low_diversity",
            replication_id=1
        ),
        ExperimentalCondition(
            diversity_level=DiversityLevel.HIGH,
            evolution_pressure=EvolutionPressure.PRESENT,
            role_configuration=RoleConfiguration.BALANCED,
            condition_id="high_diversity",
            replication_id=1
        )
    ]

    results = {}

    for condition in conditions:
        print(f"\n{condition.diversity_level.value.upper()} 多样性实验:")
        print("-" * 30)

        # 创建实验设置
        model_pool = condition.get_model_pool()
        population_size = condition.get_population_size()
        agents = create_stratified_population(model_pool, population_size)
        _adjust_role_distribution(agents, condition.get_role_distribution())

        tasks = TaskFactory.create_mixed_tasks(count_per_domain=3)
        ecosystem = Ecosystem(agents, tasks)

        generations = 6
        performance_history = []

        for gen in range(generations):
            _simulate_task_execution(ecosystem)
            avg_performance = _calculate_average_performance(ecosystem.agents)
            performance_history.append(avg_performance)
            print(f"第 {gen + 1} 代: {avg_performance:.3f}")

            if gen < generations - 1:
                ecosystem.evolve_population()

        results[condition.diversity_level.value] = {
            "performance": performance_history,
            "final_performance": performance_history[-1],
            "population_size": len(agents)
        }

        print(f"最终性能: {performance_history[-1]:.3f}")

    # 比较结果
    print("\n" + "=" * 40)
    print("📊 对比结果:")
    print("=" * 40)

    low_result = results["low"]
    high_result = results["high"]

    print(f"低多样性: {low_result['final_performance']:.3f} (种群: {low_result['population_size']})")
    print(f"高多样性: {high_result['final_performance']:.3f} (种群: {high_result['population_size']})")
    print(f"性能差异: {high_result['final_performance'] - low_result['final_performance']:.3f}")

    if high_result['final_performance'] > low_result['final_performance']:
        print("✅ 高多样性表现更好")
    else:
        print("❓ 需要更多数据验证")

    return results


if __name__ == "__main__":
    print("选择演示类型:")
    print("1. 简单演示 (单个实验)")
    print("2. 对比演示 (多样性对比)")

    # 由于是演示，默认运行简单演示
    print("\n默认运行简单演示...")
    demo_result = run_simple_demo()

    # 询问是否运行对比演示
    print(f"\n是否继续运行对比演示? (性能对比实验)")
    print("这个演示会对比高多样性和低多样性的效果")

    # 由于在脚本环境中，自动运行对比演示
    print("\n自动运行对比演示...")
    comparison_result = run_comparison_demo()

    print(f"\n🎯 演示总结:")
    print(f"完成了 {len([demo_result, comparison_result])} 个演示实验")
    print(f"结果保存在 results/ 目录中")
    print(f"可以查看 JSON 文件了解详细数据")