"""
实验设计模块 - 基于2×2×3因子设计

Authors: Zhang Shuren, AI Personality LAB
Date: 2025-09-20
"""

import json
import os
from enum import Enum
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import logging

from ..core.ecosystem import create_stratified_population

logger = logging.getLogger(__name__)


class DiversityLevel(Enum):
    """认知多样性水平"""
    LOW = "low"      # 5个相似的小型模型
    HIGH = "high"    # 18个多样化模型


class EvolutionPressure(Enum):
    """进化压力"""
    NONE = "none"        # 无进化压力（随机选择）
    PRESENT = "present"  # 有进化压力（适应度选择）


class RoleConfiguration(Enum):
    """认知角色配置"""
    BALANCED = "balanced"      # 33% Critical, 33% Standard, 33% Awakened
    CRITICAL = "critical"      # 60% Critical, 20% Standard, 20% Awakened
    INNOVATIVE = "innovative"  # 20% Critical, 20% Standard, 60% Awakened


@dataclass
class ExperimentalCondition:
    """实验条件定义"""
    diversity_level: DiversityLevel
    evolution_pressure: EvolutionPressure
    role_configuration: RoleConfiguration
    condition_id: str
    replication_id: int

    def get_description(self) -> str:
        """获取实验条件描述"""
        return (f"{self.diversity_level.value}_"
                f"{self.evolution_pressure.value}_"
                f"{self.role_configuration.value}_"
                f"rep{self.replication_id}")

    def get_model_pool(self) -> List[Dict]:
        """根据多样性水平获取模型池"""
        if self.diversity_level == DiversityLevel.LOW:
            # 低多样性：5个相似的小型模型
            return [
                {"model": "qwen:0.5b", "manufacturer": "Qwen", "scale": "Small"},
                {"model": "gemma:2b", "manufacturer": "Google", "scale": "Small"},
                {"model": "llama3.2:1b", "manufacturer": "Meta", "scale": "Small"},
                {"model": "phi3:mini", "manufacturer": "Other", "scale": "Small"},
                {"model": "deepseek-r1:1.5b", "manufacturer": "DeepSeek", "scale": "Small"}
            ]
        else:
            # 高多样性：18个多样化模型
            return [
                {"model": "qwen:0.5b", "manufacturer": "Qwen", "scale": "Small"},
                {"model": "qwen3:4b", "manufacturer": "Qwen", "scale": "Medium"},
                {"model": "qwen:7b-chat", "manufacturer": "Qwen", "scale": "Medium"},
                {"model": "qwen3:8b", "manufacturer": "Qwen", "scale": "Medium"},
                {"model": "gemma:2b", "manufacturer": "Google", "scale": "Small"},
                {"model": "gemma3:latest", "manufacturer": "Google", "scale": "Large"},
                {"model": "llama3.2:1b", "manufacturer": "Meta", "scale": "Small"},
                {"model": "llama3.2:3b", "manufacturer": "Meta", "scale": "Small"},
                {"model": "llama3:instruct", "manufacturer": "Meta", "scale": "Medium"},
                {"model": "llama3:latest", "manufacturer": "Meta", "scale": "Medium"},
                {"model": "deepseek-r1:1.5b", "manufacturer": "DeepSeek", "scale": "Small"},
                {"model": "deepseek-coder:6.7b-instruct", "manufacturer": "DeepSeek", "scale": "Medium"},
                {"model": "deepseek-r1:8b", "manufacturer": "DeepSeek", "scale": "Medium"},
                {"model": "phi3:mini", "manufacturer": "Other", "scale": "Small"},
                {"model": "yi:6b", "manufacturer": "Other", "scale": "Medium"},
                {"model": "mistral:7b-instruct-v0.2-q5_K_M", "manufacturer": "Other", "scale": "Medium"},
                {"model": "glm4:9b", "manufacturer": "Other", "scale": "Large"},
                {"model": "mistral-nemo:latest", "manufacturer": "Other", "scale": "Large"}
            ]

    def get_role_distribution(self) -> Dict[str, float]:
        """根据角色配置获取角色分布"""
        if self.role_configuration == RoleConfiguration.BALANCED:
            return {"critical": 0.33, "standard": 0.33, "awakened": 0.34}
        elif self.role_configuration == RoleConfiguration.CRITICAL:
            return {"critical": 0.60, "standard": 0.20, "awakened": 0.20}
        else:  # INNOVATIVE
            return {"critical": 0.20, "standard": 0.20, "awakened": 0.60}

    def get_population_size(self) -> int:
        """获取种群大小"""
        if self.diversity_level == DiversityLevel.LOW:
            return 15  # 低多样性使用较小种群
        else:
            return 30  # 高多样性使用标准种群


@dataclass
class ExperimentalDesign:
    """实验设计管理器"""

    def __init__(self):
        self.conditions: List[ExperimentalCondition] = []

    def create_all_conditions(self, replications: int = 10) -> List[ExperimentalCondition]:
        """创建所有实验条件"""
        self.conditions = []
        condition_id = 1

        for diversity in DiversityLevel:
            for evolution in EvolutionPressure:
                for role in RoleConfiguration:
                    for replication in range(1, replications + 1):
                        condition = ExperimentalCondition(
                            diversity_level=diversity,
                            evolution_pressure=evolution,
                            role_configuration=role,
                            condition_id=f"cond_{condition_id:03d}",
                            replication_id=replication
                        )
                        self.conditions.append(condition)
                        condition_id += 1

        total_conditions = len(DiversityLevel) * len(EvolutionPressure) * len(RoleConfiguration) * replications
        logger.info(f"创建了 {total_conditions} 个实验条件 (2×2×3×{replications})")

        return self.conditions

    def create_single_condition(self,
                               diversity_level: str,
                               evolution_pressure: str,
                               role_configuration: str,
                               replication_id: int = 1) -> ExperimentalCondition:
        """创建单个实验条件"""
        condition = ExperimentalCondition(
            diversity_level=DiversityLevel(diversity_level),
            evolution_pressure=EvolutionPressure(evolution_pressure),
            role_configuration=RoleConfiguration(role_configuration),
            condition_id=f"custom_{diversity_level}_{evolution_pressure}_{role_configuration}",
            replication_id=replication_id
        )
        return condition

    def get_conditions_by_factor(self, factor: str, value: str) -> List[ExperimentalCondition]:
        """根据因子获取实验条件"""
        matching_conditions = []

        for condition in self.conditions:
            if factor == "diversity_level" and condition.diversity_level.value == value:
                matching_conditions.append(condition)
            elif factor == "evolution_pressure" and condition.evolution_pressure.value == value:
                matching_conditions.append(condition)
            elif factor == "role_configuration" and condition.role_configuration.value == value:
                matching_conditions.append(condition)

        return matching_conditions

    def get_design_summary(self) -> Dict:
        """获取实验设计总结"""
        if not self.conditions:
            return {"total_conditions": 0}

        diversity_counts = {}
        evolution_counts = {}
        role_counts = {}

        for condition in self.conditions:
            # 多样性水平计数
            div_key = condition.diversity_level.value
            diversity_counts[div_key] = diversity_counts.get(div_key, 0) + 1

            # 进化压力计数
            evo_key = condition.evolution_pressure.value
            evolution_counts[evo_key] = evolution_counts.get(evo_key, 0) + 1

            # 角色配置计数
            role_key = condition.role_configuration.value
            role_counts[role_key] = role_counts.get(role_key, 0) + 1

        return {
            "total_conditions": len(self.conditions),
            "diversity_distribution": diversity_counts,
            "evolution_distribution": evolution_counts,
            "role_distribution": role_counts,
            "unique_combinations": len(DiversityLevel) * len(EvolutionPressure) * len(RoleConfiguration)
        }

    def save_design(self, filepath: str) -> None:
        """保存实验设计"""
        design_data = {
            "conditions": [
                {
                    "diversity_level": cond.diversity_level.value,
                    "evolution_pressure": cond.evolution_pressure.value,
                    "role_configuration": cond.role_configuration.value,
                    "condition_id": cond.condition_id,
                    "replication_id": cond.replication_id,
                    "description": cond.get_description()
                }
                for cond in self.conditions
            ],
            "summary": self.get_design_summary()
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(design_data, f, indent=2, ensure_ascii=False)

        logger.info(f"实验设计已保存到: {filepath}")

    def load_design(self, filepath: str) -> None:
        """加载实验设计"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"设计文件不存在: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            design_data = json.load(f)

        self.conditions = []
        for cond_data in design_data["conditions"]:
            condition = ExperimentalCondition(
                diversity_level=DiversityLevel(cond_data["diversity_level"]),
                evolution_pressure=EvolutionPressure(cond_data["evolution_pressure"]),
                role_configuration=RoleConfiguration(cond_data["role_configuration"]),
                condition_id=cond_data["condition_id"],
                replication_id=cond_data["replication_id"]
            )
            self.conditions.append(condition)

        logger.info(f"加载了 {len(self.conditions)} 个实验条件")

    def validate_design(self) -> List[str]:
        """验证实验设计的完整性"""
        errors = []

        if not self.conditions:
            errors.append("没有定义任何实验条件")
            return errors

        # 检查条件ID唯一性
        condition_ids = [cond.condition_id for cond in self.conditions]
        if len(condition_ids) != len(set(condition_ids)):
            errors.append("实验条件ID不唯一")

        # 检查每个条件的完整性
        for condition in self.conditions:
            try:
                condition.get_model_pool()
                condition.get_role_distribution()
                condition.get_population_size()
            except Exception as e:
                errors.append(f"条件 {condition.condition_id} 验证失败: {e}")

        return errors

    def create_population_for_condition(self, condition: ExperimentalCondition) -> List:
        """为特定条件创建智能体种群 - 修复版 (采用新的ID系统)"""
        model_pool = condition.get_model_pool()
        population_size = condition.get_population_size()

        # 创建基础种群 (使用新的ID命名规范)
        agents = self._create_stratified_population_with_new_ids(model_pool, population_size, generation=0)

        # 根据角色配置调整分布
        role_distribution = condition.get_role_distribution()

        # 计算目标数量
        target_counts = {
            role: int(count * population_size)
            for role, count in role_distribution.items()
        }

        # 确保总数正确
        target_counts["awakened"] += population_size - sum(target_counts.values())

        # 重新分配角色（如果需要）
        current_counts = {"critical": 0, "standard": 0, "awakened": 0}
        for agent in agents:
            current_counts[agent.role] += 1

        # 调整角色分配
        for role, target_count in target_counts.items():
            while current_counts[role] < target_count:
                # 找到可以替换的智能体
                for agent in agents:
                    if current_counts[agent.role] > target_counts.get(agent.role, 0):
                        old_role = agent.role
                        agent.role = role
                        # 更新系统提示
                        from ..core.agent import AgentFactory
                        prompts = AgentFactory.load_prompts()
                        agent.system_prompt = prompts.get(role, prompts["standard"])
                        # 更新计数
                        current_counts[old_role] -= 1
                        current_counts[role] += 1
                        break

        logger.info(f"为条件 {condition.get_description()} 创建了 {len(agents)} 个智能体")
        return agents
    
    def _create_stratified_population_with_new_ids(self, model_pool: List[Dict], population_size: int, generation: int = 0) -> List:
        """创建带有新ID命名规范的分层种群"""
        from ..core.ecosystem import create_stratified_population
        from ..core.agent import AgentFactory
        
        # 创建基础种群
        agents = create_stratified_population(model_pool, population_size)
        
        # 应用新的ID命名规范
        prompts = AgentFactory.load_prompts()
        
        for i, agent in enumerate(agents):
            # 生成新ID: gen_X_YY_type
            if i < 10:
                agent_type = "critical"
            elif i < 20:
                agent_type = "awakened"
            else:
                agent_type = "standard"
            
            agent.id = f"gen_{generation}_{i+1:02d}_{agent_type}"
            agent.role = agent_type
            agent.system_prompt = prompts.get(agent_type, prompts["standard"])
            agent.generation = generation
            agent.original_source = None
            agent.is_variant = False
        
        return agents

    def __str__(self) -> str:
        """字符串表示"""
        summary = self.get_design_summary()
        return (f"ExperimentalDesign(conditions={summary['total_conditions']}, "
                f"unique_combinations={summary['unique_combinations']})")


def create_factorial_design(replications: int = 10) -> ExperimentalDesign:
    """创建完整的因子实验设计"""
    design = ExperimentalDesign()
    design.create_all_conditions(replications)
    return design


if __name__ == "__main__":
    # 测试实验设计
    design = create_factorial_design(replications=2)
    summary = design.get_design_summary()
    print(f"实验设计总结: {summary}")

    # 验证设计
    errors = design.validate_design()
    if errors:
        print(f"设计验证错误: {errors}")
    else:
        print("设计验证通过")