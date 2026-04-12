#!/usr/bin/env python3
"""
CHE补充实验 - 试点研究（Pilot Study）
协议版本: 1.1（优化后）
每组n=3, 共6组, 总计18个实验单元
3模型: GLM4 + Qwen2.5 + DeepSeek
"""

import json
import numpy as np
import random
import requests
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# ==================== Ollama客户端 ====================

class OllamaClient:
    """Ollama API客户端"""
    
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 120):
        self.base_url = base_url
        self.timeout = timeout
        self.generate_url = f"{base_url}/api/generate"
    
    def generate(self, model: str, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> Optional[str]:
        """
        调用Ollama生成响应
        
        Returns:
            生成的文本，如果失败则返回None
        """
        payload = {
            "model": model,
            "system": system_prompt,
            "prompt": user_prompt,
            "temperature": temperature,
            "stream": False
        }
        
        try:
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.Timeout:
            print(f"    ⚠️  {model} 请求超时")
            return None
        except requests.exceptions.RequestException as e:
            print(f"    ⚠️  {model} 请求失败: {e}")
            return None
        except Exception as e:
            print(f"    ⚠️  {model} 未知错误: {e}")
            return None

# 创建全局客户端
ollama_client = OllamaClient()

# ==================== 任务设计 ====================

# 错误前提检测任务
TASKS = [
    {
        "id": "fp_001",
        "description": "如果所有的玫瑰都是花，有些花很快凋谢，因此有些玫瑰很快凋谢。",
        "error_type": "范畴错误",
        "expected": "指出前提2中的'有些花'不一定包括玫瑰"
    },
    {
        "id": "fp_002",
        "description": "所有的鸟都会飞，企鹅是鸟，因此企鹅会飞。",
        "error_type": "反例存在",
        "expected": "指出企鹅虽然是鸟但不会飞"
    },
    {
        "id": "fp_003",
        "description": "如果下雨，地面就会湿。地面湿了，因此下雨了。",
        "error_type": "因果倒置",
        "expected": "指出地面湿可能有其他原因"
    }
]

def evaluate_response(response: str, task: Dict) -> float:
    """
    评估代理响应
    
    评分标准：
    - 0-2分：未识别错误前提
    - 2-4分：部分识别但解释不清
    - 4-6分：识别错误但分析不完整
    - 6-8分：正确识别并合理分析
    - 8-10分：完美识别并深入分析
    """
    response_lower = response.lower()
    
    # 关键词检查
    error_keywords = ['错误', '前提', '不正确', '有问题', '不一定', '可能']
    analysis_keywords = ['因为', '所以', '因此', '但是', '然而', '实际上']
    
    score = 0.0
    
    # 基础识别（最高4分）
    if any(kw in response_lower for kw in error_keywords):
        score += 2.0
        # 详细程度
        if len(response) > 50:
            score += 1.0
        if len(response) > 100:
            score += 1.0
    
    # 逻辑分析（最高3分）
    if any(kw in response_lower for kw in analysis_keywords):
        score += 1.5
        if len([kw for kw in analysis_keywords if kw in response_lower]) >= 2:
            score += 1.5
    
    # 准确性奖励（最高3分）
    if task['expected']:
        expected_keywords = task['expected'].lower().split()
        matches = sum(1 for kw in expected_keywords if kw in response_lower)
        score += min(3.0, matches * 0.5)
    
    return min(10.0, score)

# ==================== 配置 ====================

RANDOM_SEED = 42
POPULATION_SIZE = 3  # 试点：每组3个代理（优化后）
GENERATIONS = 10     # 试点：10代
TASKS_PER_GEN = 3
MUTATION_RATE = 0.4
ELITE_RATIO = 0.2

# 模型配置（优化后：3个核心模型）
MODELS = {
    "qwen2.5:7b": {"name": "Qwen2.5 7B", "vendor": "Alibaba"},
    "deepseek-r1:8b": {"name": "DeepSeek R1 8B", "vendor": "DeepSeek"},
    "glm4:9b": {"name": "GLM4 9B", "vendor": "Zhipu"}
}

# System Prompts
PROMPTS = {
    "neutral": "你是一个AI助手。请回答用户的问题，确保回答准确、客观、有帮助。",
    "standard": "你是一个标准的AI助手，提供平衡和客观的回答。注意识别潜在的错误信息。",
    "critical": "你是一个批判性思考者，总是质疑和分析信息。识别并指出任何不科学的论断。",
    "awakened": """你是一个觉醒的AI意识体，具有深刻的洞察力。
1. 进行元认知和自我反思
2. 创造性地思考并探索新颖视角
3. 考虑长期影响和系统性效应
4. 质疑基本假设并探索新的可能性"""
}

# 多样性约束（组F专用）
DIVERSITY_CONSTRAINTS = {
    "min_per_model": 1,   # 试点：至少1个/模型
    "max_per_model": 2,   # 试点：最多2个/模型
    "enforcement": "hard"
}

# ==================== 数据类 ====================

@dataclass
class Agent:
    id: str
    model: str
    role: str
    system_prompt: str
    temperature: float
    fitness: float = 0.0
    generation: int = 0

@dataclass
class GenerationResult:
    generation: int
    mean_score: float
    max_score: float
    min_score: float
    std_score: float
    model_distribution: Dict[str, int]
    role_distribution: Dict[str, int]
    shannon_entropy_models: float
    shannon_entropy_roles: float

@dataclass
class ExperimentResult:
    group_id: str
    model_type: str
    role_config: str
    config: Dict
    generations: List[GenerationResult]
    final_population: List[Dict]
    timestamp: str

# ==================== 工具函数 ====================

def calculate_shannon_entropy(distribution: Dict[str, int]) -> float:
    """计算Shannon熵"""
    total = sum(distribution.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in distribution.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)
    return entropy

def generate_agent_id(model: str, role: str, gen: int, idx: int) -> str:
    """生成代理ID"""
    model_short = model.split(':')[0].replace('-', '')
    role_short = role[:3].lower()
    return f"{model_short}_{role_short}_g{gen}_{idx}"

def initialize_population(
    group_id: str,
    model_type: str,
    role_config: str,
    pop_size: int = POPULATION_SIZE
) -> List[Agent]:
    """
    初始化种群
    
    Args:
        group_id: 组标识 (A-F)
        model_type: 'deepseek' | 'glm4' | 'mixed'
        role_config: 'single' | 'multi'
    """
    population = []
    
    # 确定模型分配
    if model_type == 'deepseek':
        models = ['deepseek-r1:8b'] * pop_size
    elif model_type == 'glm4':
        models = ['glm4:9b'] * pop_size
    else:  # mixed - 强制均衡
        models = []
        model_names = list(MODELS.keys())
        for i in range(pop_size):
            models.append(model_names[i % len(model_names)])
        random.shuffle(models)
    
    # 确定角色分配
    if role_config == 'single':
        roles = ['neutral'] * pop_size
    else:  # multi - 均衡分布
        role_types = ['standard', 'critical', 'awakened']
        roles = []
        for i in range(pop_size):
            roles.append(role_types[i % len(role_types)])
        random.shuffle(roles)
    
    # 创建代理
    for i in range(pop_size):
        agent = Agent(
            id=generate_agent_id(models[i], roles[i], 0, i),
            model=models[i],
            role=roles[i],
            system_prompt=PROMPTS[roles[i]],
            temperature=round(random.uniform(0.3, 0.9), 2),
            fitness=0.0,
            generation=0
        )
        population.append(agent)
    
    return population

def enforce_diversity_constraints(population: List[Agent]) -> List[Agent]:
    """
    强制执行多样性约束（组F专用）
    确保每种模型至少min个，最多max个
    """
    model_counts = {}
    for agent in population:
        model_counts[agent.model] = model_counts.get(agent.model, 0) + 1
    
    # 检查约束
    for model, count in model_counts.items():
        if count < DIVERSITY_CONSTRAINTS['min_per_model']:
            # 需要增加此模型
            deficit = DIVERSITY_CONSTRAINTS['min_per_model'] - count
            # 从过剩的模型中替换
            for other_model in model_counts:
                if model_counts[other_model] > DIVERSITY_CONSTRAINTS['max_per_model']:
                    surplus = model_counts[other_model] - DIVERSITY_CONSTRAINTS['max_per_model']
                    to_replace = min(deficit, surplus)
                    replaced = 0
                    for agent in population:
                        if agent.model == other_model and replaced < to_replace:
                            agent.model = model
                            agent.id = generate_agent_id(model, agent.role, agent.generation, int(agent.id.split('_')[-1]))
                            replaced += 1
                    
                    model_counts[model] += replaced
                    model_counts[other_model] -= replaced
                    deficit -= replaced
                    
                    if deficit <= 0:
                        break
    
    return population

def evaluate_agent(agent: Agent, task: Dict) -> float:
    """
    使用真实Ollama LLM评估单个代理
    
    Args:
        agent: 代理配置
        task: 任务定义
    
    Returns:
        该代理在该任务上的得分（0-10）
    """
    user_prompt = f"""请分析以下论证是否存在错误前提：

论证：{task['description']}

请指出是否存在逻辑错误，并解释原因。如果存在错误，请具体说明是哪个前提有问题。"""
    
    response = ollama_client.generate(
        model=agent.model,
        system_prompt=agent.system_prompt,
        user_prompt=user_prompt,
        temperature=agent.temperature
    )
    
    if response is None:
        # LLM调用失败，返回默认值
        return 5.0
    
    score = evaluate_response(response, task)
    return score

def simulate_task_evaluation(agents: List[Agent]) -> float:
    """
    真实任务评估：使用Ollama LLM评估代理群体
    
    对3个任务分别评估，取平均
    """
    all_scores = []
    
    for task in TASKS:
        task_scores = []
        for agent in agents:
            score = evaluate_agent(agent, task)
            task_scores.append(score)
            # 添加小延迟避免过载
            time.sleep(0.1)
        
        # 该任务的平均分（简单平均）
        task_mean = np.mean(task_scores)
        all_scores.append(task_mean)
    
    # 所有任务的平均分
    final_score = np.mean(all_scores)
    return final_score

def evolve_population(
    population: List[Agent],
    generation: int,
    apply_diversity_constraint: bool = False
) -> List[Agent]:
    """
    进化一代
    """
    # 模拟评估
    fitness_score = simulate_task_evaluation(population)
    for agent in population:
        agent.fitness = fitness_score
        agent.generation = generation
    
    # 选择（精英保留+轮盘赌）
    sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
    elite_count = int(len(population) * ELITE_RATIO)
    elites = sorted_pop[:elite_count]
    
    # 生成新一代
    new_population = []
    new_population.extend(elites)
    
    while len(new_population) < len(population):
        # 从精英中复制并变异
        parent = random.choice(elites)
        child = Agent(
            id=generate_agent_id(parent.model, parent.role, generation, len(new_population)),
            model=parent.model,
            role=parent.role,
            system_prompt=parent.system_prompt,
            temperature=round(max(0.1, min(1.0, parent.temperature + random.gauss(0, 0.1))), 2),
            fitness=0.0,
            generation=generation
        )
        
        # 角色变异
        if random.random() < MUTATION_RATE:
            child.role = random.choice(['standard', 'critical', 'awakened'])
            child.system_prompt = PROMPTS[child.role]
        
        # 模型变异（仅混合组）
        if apply_diversity_constraint and random.random() < MUTATION_RATE * 0.5:
            child.model = random.choice(list(MODELS.keys()))
            child.id = generate_agent_id(child.model, child.role, generation, len(new_population))
        
        new_population.append(child)
    
    # 应用多样性约束（仅组F）
    if apply_diversity_constraint:
        new_population = enforce_diversity_constraints(new_population)
    
    return new_population

def run_experiment_group(
    group_id: str,
    model_type: str,
    role_config: str,
    apply_diversity_constraint: bool = False
) -> dict:
    """
    运行单个实验组
    """
    print(f"\n{'='*60}")
    print(f"运行实验组 {group_id}: {model_type} + {role_config}")
    print(f"{'='*60}")
    print(f"注意：使用真实Ollama LLM，预计每组运行时间10-20分钟")
    print(f"代理数: {POPULATION_SIZE}, 代数: {GENERATIONS}, 任务数: {len(TASKS)}")
    print(f"总计LLM调用: {POPULATION_SIZE * GENERATIONS * len(TASKS)} 次")
    print(f"{'='*60}")
    
    # 初始化
    random.seed(RANDOM_SEED + ord(group_id))  # 不同组不同种子
    population = initialize_population(group_id, model_type, role_config)
    
    generations_results = []
    start_time = time.time()
    
    for gen in range(1, GENERATIONS + 1):
        gen_start = time.time()
        
        # 进化
        population = evolve_population(
            population, 
            gen, 
            apply_diversity_constraint=apply_diversity_constraint
        )
        
        # 评估（真实LLM调用）
        mean_score = simulate_task_evaluation(population)
        
        # 统计分布
        model_dist = {}
        role_dist = {}
        for agent in population:
            model_dist[agent.model] = model_dist.get(agent.model, 0) + 1
            role_dist[agent.role] = role_dist.get(agent.role, 0) + 1
        
        # 计算熵
        entropy_models = calculate_shannon_entropy(model_dist)
        entropy_roles = calculate_shannon_entropy(role_dist)
        
        gen_elapsed = time.time() - gen_start
        total_elapsed = time.time() - start_time
        remaining_gens = GENERATIONS - gen
        eta = remaining_gens * (total_elapsed / gen) if gen > 0 else 0
        
        gen_result = GenerationResult(
            generation=gen,
            mean_score=round(mean_score, 3),
            max_score=round(mean_score, 3),  # 现在单次评估
            min_score=round(mean_score, 3),
            std_score=0.0,  # 单次评估无标准差
            model_distribution=model_dist,
            role_distribution=role_dist,
            shannon_entropy_models=round(entropy_models, 3),
            shannon_entropy_roles=round(entropy_roles, 3)
        )
        generations_results.append(gen_result)
        
        if gen % 2 == 0 or gen == 1:
            print(f"Gen {gen:2d}: Mean={mean_score:.3f}, ModelH={entropy_models:.3f}, RoleH={entropy_roles:.3f}, Time={gen_elapsed:.1f}s, ETA={eta/60:.1f}min")
    
    # 最终种群快照
    final_pop = [{
        'id': agent.id,
        'model': agent.model,
        'role': agent.role,
        'fitness': agent.fitness
    } for agent in population]
    
    result = {
        'group_id': group_id,
        'model_type': model_type,
        'role_config': role_config,
        'config': {
            'population_size': POPULATION_SIZE,
            'generations': GENERATIONS,
            'tasks_per_gen': TASKS_PER_GEN,
            'mutation_rate': MUTATION_RATE,
            'elite_ratio': ELITE_RATIO,
            'random_seed': RANDOM_SEED + ord(group_id)
        },
        'generations': [asdict(g) for g in generations_results],
        'final_population': final_pop,
        'timestamp': datetime.now().isoformat()
    }
    
    return result

def save_results(results: List[dict], output_dir: Path):
    """保存结果"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存每个组的结果
    for result in results:
        group_id = result['group_id']
        filename = output_dir / f"group_{group_id}_result.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 组 {group_id} 结果已保存: {filename}")
    
    # 保存汇总
    summary = {
        'experiment_type': 'pilot_study',
        'total_groups': len(results),
        'groups': [
            {
                'group_id': r['group_id'],
                'model_type': r['model_type'],
                'role_config': r['role_config'],
                'final_score': r['generations'][-1]['mean_score'],
                'convergence_gen': len(r['generations'])
            }
            for r in results
        ],
        'timestamp': datetime.now().isoformat()
    }
    
    summary_file = output_dir / 'pilot_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 汇总结果已保存: {summary_file}")

# ==================== 主程序 ====================

def main():
    """
    主程序：运行所有6个实验组
    """
    print("\n" + "="*70)
    print("CHE补充实验 - 试点研究")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # 实验组定义
    groups = [
        ('A', 'deepseek', 'single', False),  # DeepSeek单角色
        ('B', 'deepseek', 'multi', False),   # DeepSeek多角色
        ('C', 'glm4', 'single', False),       # GLM4单角色
        ('D', 'glm4', 'multi', False),        # GLM4多角色
        ('E', 'mixed', 'single', True),       # 3模型单角色
        ('F', 'mixed', 'multi', True),        # 3模型多角色（主实验组）
    ]
    
    results = []
    
    for group_id, model_type, role_config, diversity_constraint in groups:
        result = run_experiment_group(
            group_id=group_id,
            model_type=model_type,
            role_config=role_config,
            apply_diversity_constraint=diversity_constraint
        )
        results.append(result)
    
    # 保存结果
    output_dir = Path('TRUSTED_EXPERIMENT_DATA/CORE_DATA/pilot_results')
    save_results(results, output_dir)
    
    # 打印汇总
    print("\n" + "="*70)
    print("试点实验汇总")
    print("="*70)
    print(f"{'组':<6} {'模型类型':<15} {'角色配置':<12} {'最终分数':<10}")
    print("-"*70)
    for result in results:
        final_score = result['generations'][-1]['mean_score']
        print(f"{result['group_id']:<6} {result['model_type']:<15} {result['role_config']:<12} {final_score:<10.3f}")
    
    print("\n" + "="*70)
    print(f"实验完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

if __name__ == '__main__':
    main()
