#!/usr/bin/env python3
"""
实验系统验证脚本 - 验证基于KISS YAGNI架构的实验系统

Authors: Zhang Shuren, AI Personality LAB
Date: 2025-09-20
"""

import sys
import os
import time
import logging
from typing import List, Dict

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.che.experimental.design import ExperimentalDesign, ExperimentalCondition
from src.che.core.ecosystem import Ecosystem
from src.che.core.task import Task

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExperimentSystemValidator:
    """实验系统验证器"""

    def __init__(self):
        self.validation_results = {}

    def validate_core_components(self) -> Dict:
        """验证核心组件"""
        logger.info("开始验证核心组件...")

        results = {
            "agent_module": self._validate_agent_module(),
            "task_module": self._validate_task_module(),
            "ecosystem_module": self._validate_ecosystem_module()
        }

        logger.info(f"核心组件验证完成: {sum(results.values())}/{len(results)} 通过")
        return results

    def validate_experimental_design(self) -> Dict:
        """验证实验设计"""
        logger.info("开始验证实验设计...")

        results = {
            "factorial_design": self._validate_factorial_design(),
            "condition_creation": self._validate_condition_creation(),
            "population_generation": self._validate_population_generation()
        }

        logger.info(f"实验设计验证完成: {sum(results.values())}/{len(results)} 通过")
        return results

    def validate_system_integration(self) -> Dict:
        """验证系统集成"""
        logger.info("开始验证系统集成...")

        results = {
            "end_to_end_workflow": self._validate_end_to_end_workflow(),
            "data_persistence": self._validate_data_persistence(),
            "performance_benchmarks": self._validate_performance_benchmarks()
        }

        logger.info(f"系统集成验证完成: {sum(results.values())}/{len(results)} 通过")
        return results

    def _validate_agent_module(self) -> bool:
        """验证Agent模块"""
        try:
            from src.che.core.agent import Agent
            from src.che.agents.agent_factory import AgentFactory

            # 测试Agent创建
            agent = Agent(
                id="test_agent",
                model="qwen:0.5b",
                role="critical",
                system_prompt="Test prompt"
            )

            # 测试Agent功能
            agent.add_response({"task_id": "t1", "score": 0.8})
            assert agent.get_average_score() == 0.8

            # 测试AgentFactory
            model_info = {"model": "qwen:0.5b", "manufacturer": "Qwen", "scale": "Small"}
            factory_agent = AgentFactory.create_agent(model_info, "critical", "factory_test")

            logger.info("✅ Agent模块验证通过")
            return True

        except Exception as e:
            logger.error(f"❌ Agent模块验证失败: {e}")
            return False

    def _validate_task_module(self) -> bool:
        """验证Task模块"""
        try:
            from src.che.core.task import Task, TaskFactory

            # 测试Task创建
            task = Task(
                id="test_task",
                instruction="Test instruction",
                false_premise="False premise",
                reality="Correct reality"
            )

            # 测试任务评估
            response = "This is incorrect because the false premise is wrong. The correct concept is reality."
            score = task.evaluate_response(response)
            assert 0 < score <= 1.0

            # 测试TaskFactory
            tasks = TaskFactory.create_psychology_tasks(count=3)
            assert len(tasks) == 3

            logger.info("✅ Task模块验证通过")
            return True

        except Exception as e:
            logger.error(f"❌ Task模块验证失败: {e}")
            return False

    def _validate_ecosystem_module(self) -> bool:
        """验证Ecosystem模块"""
        try:
            from src.che.core.ecosystem import Ecosystem, create_stratified_population

            # 测试Ecosystem创建
            ecosystem = Ecosystem()
            assert ecosystem.generation == 0

            # 测试种群创建
            model_pool = [
                {"model": "qwen:0.5b", "manufacturer": "Qwen", "scale": "Small"},
                {"model": "gemma:2b", "manufacturer": "Google", "scale": "Small"}
            ]

            agents = create_stratified_population(model_pool, population_size=6)
            assert len(agents) == 6

            # 测试多样性计算
            diversity_index = ecosystem.calculate_diversity_index()
            assert 0 <= diversity_index <= 1

            logger.info("✅ Ecosystem模块验证通过")
            return True

        except Exception as e:
            logger.error(f"❌ Ecosystem模块验证失败: {e}")
            return False

    def _validate_factorial_design(self) -> bool:
        """验证因子设计"""
        try:
            design = ExperimentalDesign()
            conditions = design.create_all_conditions(replications=2)

            # 验证条件数量
            expected_conditions = 2 * 2 * 3 * 2  # 2×2×3×2
            assert len(conditions) == expected_conditions

            # 验证条件唯一性
            condition_ids = [cond.condition_id for cond in conditions]
            assert len(set(condition_ids)) == len(condition_ids)

            # 验证设计验证
            errors = design.validate_design()
            assert len(errors) == 0

            logger.info("✅ 因子设计验证通过")
            return True

        except Exception as e:
            logger.error(f"❌ 因子设计验证失败: {e}")
            return False

    def _validate_condition_creation(self) -> bool:
        """验证条件创建"""
        try:
            condition = ExperimentalCondition(
                diversity_level="high",
                evolution_pressure="present",
                role_configuration="balanced",
                condition_id="test",
                replication_id=1
            )

            # 验证模型池
            model_pool = condition.get_model_pool()
            assert len(model_pool) > 0

            # 验证角色分布
            role_dist = condition.get_role_distribution()
            assert sum(role_dist.values()) == 1.0

            # 验证种群大小
            pop_size = condition.get_population_size()
            assert pop_size > 0

            logger.info("✅ 条件创建验证通过")
            return True

        except Exception as e:
            logger.error(f"❌ 条件创建验证失败: {e}")
            return False

    def _validate_population_generation(self) -> bool:
        """验证种群生成"""
        try:
            from src.che.experimental.design import DiversityLevel, EvolutionPressure, RoleConfiguration
            design = ExperimentalDesign()
            condition = ExperimentalCondition(
                diversity_level=DiversityLevel.HIGH,
                evolution_pressure=EvolutionPressure.PRESENT,
                role_configuration=RoleConfiguration.BALANCED,
                condition_id="test",
                replication_id=1
            )

            agents = design.create_population_for_condition(condition)

            # 验证种群大小
            assert len(agents) == 30

            # 验证角色分布
            role_counts = {"critical": 0, "standard": 0, "awakened": 0}
            for agent in agents:
                role_counts[agent.role] += 1

            # 验证分布合理性
            assert all(count >= 8 for count in role_counts.values())
            assert sum(role_counts.values()) == 30

            logger.info("✅ 种群生成验证通过")
            return True

        except Exception as e:
            logger.error(f"❌ 种群生成验证失败: {e}")
            return False

    def _validate_end_to_end_workflow(self) -> bool:
        """验证端到端工作流"""
        try:
            # 创建实验设计
            design = ExperimentalDesign()
            condition = design.create_single_condition(
                diversity_level="low",
                evolution_pressure="none",
                role_configuration="balanced",
                replication_id=1
            )

            # 创建生态系统
            agents = design.create_population_for_condition(condition)
            tasks = TaskFactory.create_mixed_tasks(count_per_domain=2)
            ecosystem = Ecosystem(agents, tasks)

            # 模拟演化过程
            initial_generation = ecosystem.generation

            # 设置模拟分数
            for i, agent in enumerate(agents):
                agent.fitness_score = 0.5 + (i * 0.1)  # 递增分数

            # 执行演化
            ecosystem.evolve_population()

            # 验证演化效果
            assert ecosystem.generation == initial_generation + 1
            assert len(ecosystem.agents) == len(agents)

            logger.info("✅ 端到端工作流验证通过")
            return True

        except Exception as e:
            logger.error(f"❌ 端到端工作流验证失败: {e}")
            return False

    def _validate_data_persistence(self) -> bool:
        """验证数据持久化"""
        try:
            import tempfile
            import os

            # 创建测试生态系统
            ecosystem = Ecosystem()
            ecosystem.generation = 5

            # 保存状态
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                temp_path = f.name

            ecosystem.save_state(temp_path)

            # 加载状态
            new_ecosystem = Ecosystem()
            new_ecosystem.load_state(temp_path)

            # 验证状态恢复
            assert new_ecosystem.generation == ecosystem.generation
            assert len(new_ecosystem.agents) == len(ecosystem.agents)

            # 清理
            os.unlink(temp_path)

            logger.info("✅ 数据持久化验证通过")
            return True

        except Exception as e:
            logger.error(f"❌ 数据持久化验证失败: {e}")
            return False

    def _validate_performance_benchmarks(self) -> bool:
        """验证性能基准"""
        try:
            import time

            # 测试条件创建性能
            start_time = time.time()
            design = ExperimentalDesign()
            design.create_all_conditions(replications=5)
            creation_time = time.time() - start_time

            # 应该在1秒内完成
            assert creation_time < 1.0

            # 测试种群生成性能
            condition = design.conditions[0]
            start_time = time.time()
            agents = design.create_population_for_condition(condition)
            generation_time = time.time() - start_time

            # 应该在0.5秒内完成
            assert generation_time < 0.5

            logger.info(f"✅ 性能基准验证通过 (创建: {creation_time:.3f}s, 生成: {generation_time:.3f}s)")
            return True

        except Exception as e:
            logger.error(f"❌ 性能基准验证失败: {e}")
            return False

    def run_full_validation(self) -> Dict:
        """运行完整验证"""
        logger.info("🚀 开始完整实验系统验证...")

        start_time = time.time()

        # 执行所有验证
        core_results = self.validate_core_components()
        design_results = self.validate_experimental_design()
        integration_results = self.validate_system_integration()

        # 汇总结果
        all_results = {**core_results, **design_results, **integration_results}
        total_tests = len(all_results)
        passed_tests = sum(all_results.values())

        execution_time = time.time() - start_time

        # 生成报告
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "execution_time": execution_time
            },
            "detailed_results": all_results,
            "status": "PASSED" if passed_tests == total_tests else "FAILED"
        }

        # 输出报告
        self._print_validation_report(report)

        return report

    def _print_validation_report(self, report: Dict):
        """打印验证报告"""
        print("\n" + "="*60)
        print("🧬 认知异质性实验系统验证报告")
        print("="*60)

        summary = report["summary"]
        print(f"\n📊 验证总结:")
        print(f"   总测试数: {summary['total_tests']}")
        print(f"   通过测试: {summary['passed_tests']}")
        print(f"   成功率: {summary['success_rate']:.1%}")
        print(f"   执行时间: {summary['execution_time']:.2f}秒")
        print(f"   总体状态: {report['status']}")

        print(f"\n🔍 详细结果:")
        for test_name, result in report["detailed_results"].items():
            status = "✅ 通过" if result else "❌ 失败"
            print(f"   {test_name}: {status}")

        print("\n" + "="*60)

        if report["status"] == "PASSED":
            print("🎉 所有验证测试通过！实验系统已就绪。")
        else:
            print("⚠️  部分验证测试失败，请检查上述错误信息。")


def main():
    """主函数"""
    print("🧬 认知异质性实验系统验证")
    print("基于KISS·YAGNI·SOLID原则的TDD驱动架构")
    print("="*60)

    validator = ExperimentSystemValidator()
    report = validator.run_full_validation()

    # 返回适当的退出码
    exit_code = 0 if report["status"] == "PASSED" else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()