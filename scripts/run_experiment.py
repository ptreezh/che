#!/usr/bin/env python3
"""
实验运行脚本 - 运行认知异质性实验

Authors: Zhang Shuren, AI Personality LAB
Date: 2025-09-20
"""

import sys
import os
import logging
import json
import time
from typing import List, Dict
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
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """实验运行器"""

    def __init__(self):
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

    def run_single_experiment(self,
                            condition: ExperimentalCondition,
                            generations: int = 15,
                            replication_id: int = 1) -> Dict:
        """运行单个实验"""

        logger.info(f"开始实验: {condition.get_description()}")
        logger.info(f"多样性水平: {condition.diversity_level.value}")
        logger.info(f"进化压力: {condition.evolution_pressure.value}")
        logger.info(f"角色配置: {condition.role_configuration.value}")

        # 1. 创建智能体种群
        model_pool = condition.get_model_pool()
        population_size = condition.get_population_size()
        agents = create_stratified_population(model_pool, population_size)

        # 2. 调整角色分布以匹配实验条件
        role_distribution = condition.get_role_distribution()
        self._adjust_role_distribution(agents, role_distribution)

        logger.info(f"创建了 {len(agents)} 个智能体")

        # 3. 创建任务
        tasks = TaskFactory.create_mixed_tasks(count_per_domain=10)
        logger.info(f"创建了 {len(tasks)} 个任务")

        # 4. 创建生态系统
        ecosystem = Ecosystem(agents, tasks)

        # 5. 运行演化实验
        experiment_data = self._run_evolution(ecosystem, generations, condition)

        # 6. 保存结果
        result_filename = f"experiment_{condition.get_description()}_rep{replication_id}.json"
        result_path = self.results_dir / result_filename

        result = {
            "condition": {
                "diversity_level": condition.diversity_level.value,
                "evolution_pressure": condition.evolution_pressure.value,
                "role_configuration": condition.role_configuration.value,
                "description": condition.get_description()
            },
            "replication_id": replication_id,
            "generations": generations,
            "population_size": len(agents),
            "final_performance": experiment_data["final_performance"],
            "diversity_trajectory": experiment_data["diversity_trajectory"],
            "performance_trajectory": experiment_data["performance_trajectory"],
            "execution_time": experiment_data["execution_time"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"实验结果已保存到: {result_path}")
        return result

    def _run_evolution(self,
                      ecosystem: Ecosystem,
                      generations: int,
                      condition: ExperimentalCondition) -> Dict:
        """运行演化过程"""

        start_time = time.time()
        performance_trajectory = []
        diversity_trajectory = []

        logger.info(f"开始 {generations} 代演化")

        for gen in range(generations):
            # 模拟任务执行和评分
            self._simulate_task_execution(ecosystem)

            # 计算当前性能
            avg_performance = self._calculate_average_performance(ecosystem.agents)
            diversity_index = ecosystem.calculate_diversity_index()

            performance_trajectory.append(avg_performance)
            diversity_trajectory.append(diversity_index)

            logger.info(f"第 {gen + 1} 代: 平均性能 = {avg_performance:.3f}, 多样性指数 = {diversity_index:.3f}")

            # 执行演化（除了最后一代）
            if gen < generations - 1:
                ecosystem.evolve_population()

        execution_time = time.time() - start_time

        return {
            "final_performance": performance_trajectory[-1],
            "performance_trajectory": performance_trajectory,
            "diversity_trajectory": diversity_trajectory,
            "execution_time": execution_time
        }

    def _simulate_task_execution(self, ecosystem: Ecosystem):
        """模拟任务执行"""
        for agent in ecosystem.agents:
            # 模拟智能体执行任务并获得分数
            # 这里使用简单的模拟，实际应用中会调用真实的LLM
            base_score = 0.5 + (hash(agent.id) % 100) / 200  # 0.5-1.0

            # 根据角色调整分数
            role_bonus = {
                "critical": 0.1,
                "standard": 0.05,
                "awakened": 0.15
            }.get(agent.role, 0.0)

            # 添加一些随机性
            import random
            random.seed(hash(agent.id + str(ecosystem.generation)))
            noise = random.uniform(-0.05, 0.05)

            agent.fitness_score = max(0.0, min(1.0, base_score + role_bonus + noise))

    def _calculate_average_performance(self, agents: List) -> float:
        """计算平均性能"""
        if not agents:
            return 0.0
        return sum(agent.fitness_score for agent in agents) / len(agents)

    def _adjust_role_distribution(self, agents: List, target_distribution: Dict[str, float]):
        """调整角色分布以匹配目标分布"""
        from src.che.core.agent import AgentFactory

        current_counts = {"critical": 0, "standard": 0, "awakened": 0}
        for agent in agents:
            current_counts[agent.role] += 1

        target_counts = {
            role: int(count * len(agents))
            for role, count in target_distribution.items()
        }

        # 确保总数正确
        target_counts["awakened"] += len(agents) - sum(target_counts.values())

        # 简单的角色调整
        for role, target_count in target_counts.items():
            while current_counts[role] < target_count:
                # 找到可以替换的智能体
                for agent in agents:
                    if current_counts[agent.role] > target_counts.get(agent.role, 0):
                        old_role = agent.role
                        agent.role = role
                        # 更新系统提示
                        prompts = AgentFactory.load_prompts()
                        agent.system_prompt = prompts.get(role, prompts["standard"])
                        # 更新计数
                        current_counts[old_role] -= 1
                        current_counts[role] += 1
                        break

    def run_factorial_experiment(self,
                                replications: int = 3,
                                generations: int = 15) -> List[Dict]:
        """运行完整的因子实验"""

        logger.info("开始完整的2×2×3因子实验")
        logger.info(f"每个条件重复 {replications} 次")
        logger.info(f"每个实验运行 {generations} 代")

        # 创建实验设计
        design = ExperimentalDesign()
        conditions = design.create_all_conditions(replications=replications)

        logger.info(f"总共 {len(conditions)} 个实验条件")

        all_results = []

        for i, condition in enumerate(conditions):
            logger.info(f"执行实验 {i + 1}/{len(conditions)}: {condition.get_description()}")

            try:
                result = self.run_single_experiment(
                    condition,
                    generations=generations,
                    replication_id=condition.replication_id
                )
                all_results.append(result)
            except Exception as e:
                logger.error(f"实验失败: {condition.get_description()}, 错误: {e}")
                continue

        # 保存汇总结果
        summary_filename = f"factorial_experiment_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
        summary_path = self.results_dir / summary_filename

        summary = {
            "experiment_type": "2x2x3_factorial",
            "total_conditions": len(conditions),
            "successful_experiments": len(all_results),
            "replications": replications,
            "generations": generations,
            "results": all_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"实验汇总已保存到: {summary_path}")
        logger.info(f"成功完成 {len(all_results)}/{len(conditions)} 个实验")

        return all_results

    def run_simple_test(self):
        """运行简单的测试实验"""

        logger.info("运行简单测试实验")

        # 创建一个简单的实验条件
        condition = ExperimentalCondition(
            diversity_level=DiversityLevel.LOW,
            evolution_pressure=EvolutionPressure.NONE,
            role_configuration=RoleConfiguration.BALANCED,
            condition_id="simple_test",
            replication_id=1
        )

        # 运行实验（较少代数）
        result = self.run_single_experiment(condition, generations=5)

        logger.info("简单测试实验完成!")
        logger.info(f"最终性能: {result['final_performance']:.3f}")
        logger.info(f"执行时间: {result['execution_time']:.2f}秒")

        return result


def main():
    """主函数"""
    print("🧬 认知异质性实验系统")
    print("基于KISS·YAGNI·SOLID原则的TDD驱动架构")
    print("=" * 60)

    runner = ExperimentRunner()

    # 提供实验选项
    print("\n请选择实验类型:")
    print("1. 简单测试实验 (快速验证)")
    print("2. 完整因子实验 (科学实验)")
    print("3. 自定义实验条件")

    choice = input("\n请输入选择 (1-3): ").strip()

    if choice == "1":
        # 简单测试
        runner.run_simple_test()

    elif choice == "2":
        # 完整因子实验
        replications = int(input("请输入重复次数 (建议3-5): ") or "3")
        generations = int(input("请输入演化代数 (建议10-15): ") or "15")

        print(f"\n开始完整因子实验: {replications}次重复, {generations}代演化")
        confirm = input("确认开始实验? (y/N): ").strip().lower()

        if confirm == 'y':
            results = runner.run_factorial_experiment(
                replications=replications,
                generations=generations
            )
            print(f"\n实验完成! 共完成 {len(results)} 个实验")
        else:
            print("实验已取消")

    elif choice == "3":
        # 自定义实验
        print("\n自定义实验条件:")
        print("多样性水平: low, high")
        print("进化压力: none, present")
        print("角色配置: balanced, critical, innovative")

        diversity = input("多样性水平 (low/high): ").strip().lower()
        evolution = input("进化压力 (none/present): ").strip().lower()
        role = input("角色配置 (balanced/critical/innovative): ").strip().lower()
        generations = int(input("演化代数: ") or "10")

        try:
            condition = ExperimentalCondition(
                diversity_level=DiversityLevel(diversity),
                evolution_pressure=EvolutionPressure(evolution),
                role_configuration=RoleConfiguration(role),
                condition_id="custom",
                replication_id=1
            )

            result = runner.run_single_experiment(condition, generations=generations)
            print(f"\n自定义实验完成! 最终性能: {result['final_performance']:.3f}")

        except ValueError as e:
            print(f"参数错误: {e}")

    else:
        print("无效选择")


if __name__ == "__main__":
    main()