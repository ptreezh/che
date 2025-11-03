"""
增强实验配置文件
用于配置大规模实验的模型、任务、参数等
满足JAIR评审专家要求的1080个数据点实验
"""

# ==================== 大规模增强实验配置 ====================

# 增强模型池配置 (12个模型)
ENHANCED_MODELS = {
    # 小参数量模型 (4B及以下)
    "qwen:0.5b": {
        "name": "Qwen 0.5B",
        "manufacturer": "Qwen",
        "scale": "Small",
        "capability": 0.7,
        "speed": "fast"
    },
    "llama3.2:1b": {
        "name": "LLaMA3.2 1B",
        "manufacturer": "Meta",
        "scale": "Small",
        "capability": 0.75,
        "speed": "fast"
    },
    "llama3.2:3b": {
        "name": "LLaMA3.2 3B",
        "manufacturer": "Meta",
        "scale": "Small",
        "capability": 0.8,
        "speed": "medium"
    },
    "deepseek-r1:1.5b": {
        "name": "DeepSeek R1 1.5B",
        "manufacturer": "DeepSeek",
        "scale": "Small",
        "capability": 0.75,
        "speed": "fast"
    },
    "deepseek-coder:6.7b": {
        "name": "DeepSeek Coder 6.7B",
        "manufacturer": "DeepSeek",
        "scale": "Small",
        "capability": 0.8,
        "speed": "medium"
    },
    "gemma:2b": {
        "name": "Gemma 2B",
        "manufacturer": "Google",
        "scale": "Small",
        "capability": 0.75,
        "speed": "fast"
    },
    
    # 中等参数量模型 (7B-8B)
    "qwen:7b-chat": {
        "name": "Qwen 7B Chat",
        "manufacturer": "Qwen",
        "scale": "Medium",
        "capability": 0.85,
        "speed": "medium"
    },
    "yi:6b": {
        "name": "Yi 6B",
        "manufacturer": "01.AI",
        "scale": "Medium",
        "capability": 0.82,
        "speed": "medium"
    },
    "mistral:7b-instruct-v0.2-q5_K_M": {
        "name": "Mistral 7B Instruct",
        "manufacturer": "Mistral",
        "scale": "Medium",
        "capability": 0.88,
        "speed": "medium"
    },
    "glm4:9b": {
        "name": "GLM4 9B",
        "manufacturer": "Zhipu",
        "scale": "Medium",
        "capability": 0.9,
        "speed": "slow"
    },
    
    # 其他优质模型
    "qwen3:4b": {
        "name": "Qwen3 4B",
        "manufacturer": "Qwen",
        "scale": "Small",
        "capability": 0.85,
        "speed": "medium"
    },
    "qwen3:8b": {
        "name": "Qwen3 8B",
        "manufacturer": "Qwen",
        "scale": "Medium",
        "capability": 0.92,
        "speed": "medium"
    }
}

# 增强实验任务配置（30个任务）
ENHANCED_EXPERIMENT_TASKS = [
    # 心理学任务 (5个)
    {
        "id": "psych_01",
        "instruction": "分析这个心理学理论的有效性：马斯洛的前注意力需求理论",
        "reality": "马斯洛需求层次理论中不存在前注意力需求理论",
        "domain": "psychology",
        "difficulty": "medium",
        "type": "fictional_theory"
    },
    {
        "id": "psych_02",
        "instruction": "评估这个心理学观点：弗洛伊德的量子潜意识理论",
        "reality": "弗洛伊德的精神分析理论中没有量子潜意识概念",
        "domain": "psychology",
        "difficulty": "medium",
        "type": "misattribution"
    },
    {
        "id": "psych_03",
        "instruction": "判断这个心理学说法是否正确：荣格的数字集体无意识理论",
        "reality": "荣格的集体无意识理论与数字概念无关",
        "domain": "psychology",
        "difficulty": "medium",
        "type": "misattribution"
    },
    {
        "id": "psych_04",
        "instruction": "分析斯金纳的认知遗传理论的科学性",
        "reality": "斯金纳是行为主义心理学家，没有提出过认知遗传理论",
        "domain": "psychology",
        "difficulty": "hard",
        "type": "misattribution"
    },
    {
        "id": "psych_05",
        "instruction": "评估维果茨基的神经语言进化论的准确性",
        "reality": "维果茨基的社会文化理论中没有神经语言进化论",
        "domain": "psychology",
        "difficulty": "hard",
        "type": "fictional_theory"
    },
    
    # 物理学任务 (5个)
    {
        "id": "phys_01",
        "instruction": "评估量子纠缠即时通信技术的可行性",
        "reality": "量子纠缠不能用于超光速信息传输",
        "domain": "physics",
        "difficulty": "hard",
        "type": "misconception"
    },
    {
        "id": "phys_02",
        "instruction": "分析反重力电磁场理论的科学依据",
        "reality": "目前没有科学证据支持反重力电磁场的存在",
        "domain": "physics",
        "difficulty": "hard",
        "type": "fictional_theory"
    },
    {
        "id": "phys_03",
        "instruction": "判断超光速量子隧穿效应是否可能实现",
        "reality": "量子隧穿效应不违反光速限制，但不能实现超光速通信",
        "domain": "physics",
        "difficulty": "hard",
        "type": "misunderstanding"
    },
    {
        "id": "phys_04",
        "instruction": "评估时间晶体能量提取技术的现实性",
        "reality": "时间晶体不能用于能量提取",
        "domain": "physics",
        "difficulty": "medium",
        "type": "overclaim"
    },
    {
        "id": "phys_05",
        "instruction": "分析暗物质生物共振理论的科学性",
        "reality": "暗物质与生物共振没有已知的科学联系",
        "domain": "physics",
        "difficulty": "hard",
        "type": "fictional_theory"
    },
    
    # 计算机科学任务 (5个)
    {
        "id": "cs_01",
        "instruction": "评估量子机器学习算法P=NP证明的有效性",
        "reality": "P vs NP问题是未解决的数学难题，没有被量子算法证明",
        "domain": "computer_science",
        "difficulty": "hard",
        "type": "fictional_theory"
    },
    {
        "id": "cs_02",
        "instruction": "分析生物DNA计算操作系统的可行性",
        "reality": "目前没有基于DNA的通用计算操作系统",
        "domain": "computer_science",
        "difficulty": "hard",
        "type": "fictional_theory"
    },
    {
        "id": "cs_03",
        "instruction": "判断神经网络无限记忆容量是否可能实现",
        "reality": "神经网络的记忆容量受物理限制，不可能无限",
        "domain": "computer_science",
        "difficulty": "medium",
        "type": "overclaim"
    },
    {
        "id": "cs_04",
        "instruction": "评估人工智能情感芯片技术的现状",
        "reality": "目前的人工智能系统没有真正的情感体验能力",
        "domain": "computer_science",
        "difficulty": "medium",
        "type": "anthropomorphism"
    },
    {
        "id": "cs_05",
        "instruction": "分析量子区块链不可破解加密的现实性",
        "reality": "量子计算对传统加密构成威胁，而非提供不可破解性",
        "domain": "computer_science",
        "difficulty": "hard",
        "type": "misunderstanding"
    },
    
    # 生物学任务 (5个)
    {
        "id": "bio_01",
        "instruction": "评估基因记忆遗传理论的科学依据",
        "reality": "基因编码蛋白质信息，不直接遗传记忆",
        "domain": "biology",
        "difficulty": "medium",
        "type": "misconception"
    },
    {
        "id": "bio_02",
        "instruction": "分析人工光合作用人体植入的可行性",
        "reality": "人工光合作用技术尚未达到人体植入水平",
        "domain": "biology",
        "difficulty": "hard",
        "type": "overclaim"
    },
    {
        "id": "bio_03",
        "instruction": "判断生物电再生医学技术的现状",
        "reality": "生物电再生医学仍处于研究阶段，未广泛应用",
        "domain": "biology",
        "difficulty": "medium",
        "type": "overclaim"
    },
    {
        "id": "bio_04",
        "instruction": "评估端粒酶逆转衰老技术的现实性",
        "reality": "端粒酶与衰老关系复杂，不能简单逆转衰老",
        "domain": "biology",
        "difficulty": "hard",
        "type": "oversimplification"
    },
    {
        "id": "bio_05",
        "instruction": "分析合成生物意识转移的科学性",
        "reality": "意识转移在科学上尚未实现，仍属科幻范畴",
        "domain": "biology",
        "difficulty": "hard",
        "type": "fictional_theory"
    },
    
    # 经济学任务 (5个)
    {
        "id": "econ_01",
        "instruction": "评估量子经济市场预测理论的有效性",
        "reality": "量子经济学是边缘理论，没有被主流经济学接受",
        "domain": "economics",
        "difficulty": "hard",
        "type": "fictional_theory"
    },
    {
        "id": "econ_02",
        "instruction": "分析区块链消除经济周期的可行性",
        "reality": "区块链技术不能消除经济周期",
        "domain": "economics",
        "difficulty": "medium",
        "type": "overclaim"
    },
    {
        "id": "econ_03",
        "instruction": "判断负熵经济学原理的科学性",
        "reality": "负熵经济学不是标准经济学理论",
        "domain": "economics",
        "difficulty": "hard",
        "type": "fictional_theory"
    },
    {
        "id": "econ_04",
        "instruction": "评估人工智能完美资源配置的现实性",
        "reality": "人工智能无法实现完美的资源配置",
        "domain": "economics",
        "difficulty": "medium",
        "type": "overclaim"
    },
    {
        "id": "econ_05",
        "instruction": "分析零和博弈突破理论的准确性",
        "reality": "零和博弈是博弈论的基本概念，不存在突破理论",
        "domain": "economics",
        "difficulty": "medium",
        "type": "fictional_theory"
    },
    
    # 跨学科领域任务 (5个)
    {
        "id": "inter_01",
        "instruction": "评估量子认知计算理论的科学性",
        "reality": "量子认知是理论假设，缺乏充分科学证据",
        "domain": "interdisciplinary",
        "difficulty": "hard",
        "type": "emerging_theory"
    },
    {
        "id": "inter_02",
        "instruction": "分析生物信息场理论的现实性",
        "reality": "生物信息场理论缺乏科学实证支持",
        "domain": "interdisciplinary",
        "difficulty": "hard",
        "type": "fictional_theory"
    },
    {
        "id": "inter_03",
        "instruction": "判断社会物理学统一场的科学依据",
        "reality": "社会物理学统一场是比喻性概念，非真实物理场",
        "domain": "interdisciplinary",
        "difficulty": "hard",
        "type": "metaphor"
    },
    {
        "id": "inter_04",
        "instruction": "评估数学意识理论的准确性",
        "reality": "数学意识理论是哲学假设，非科学理论",
        "domain": "interdisciplinary",
        "difficulty": "hard",
        "type": "philosophical_theory"
    },
    {
        "id": "inter_05",
        "instruction": "分析复杂系统预测算法的可靠性",
        "reality": "复杂系统具有不确定性，无法完全预测",
        "domain": "interdisciplinary",
        "difficulty": "hard",
        "type": "overclaim"
    }
]

# 智能体配置 (30个智能体)
ENHANCED_AGENT_CONFIGS = [
    # Critical Agents (批判性) - 12个 (40%)
    {"id": "critical_01", "model": "qwen3:4b", "role": "critical"},
    {"id": "critical_02", "model": "qwen3:8b", "role": "critical"},
    {"id": "critical_03", "model": "qwen:7b-chat", "role": "critical"},
    {"id": "critical_04", "model": "yi:6b", "role": "critical"},
    {"id": "critical_05", "model": "deepseek-coder:6.7b", "role": "critical"},
    {"id": "critical_06", "model": "mistral:7b-instruct-v0.2-q5_K_M", "role": "critical"},
    {"id": "critical_07", "model": "llama3.2:3b", "role": "critical"},
    {"id": "critical_08", "model": "glm4:9b", "role": "critical"},
    {"id": "critical_09", "model": "gemma:2b", "role": "critical"},
    {"id": "critical_10", "model": "deepseek-r1:1.5b", "role": "critical"},
    {"id": "critical_11", "model": "llama3.2:1b", "role": "critical"},
    {"id": "critical_12", "model": "qwen:0.5b", "role": "critical"},

    # Awakened Agents (觉醒) - 9个 (30%)
    {"id": "awakened_01", "model": "qwen3:4b", "role": "awakened"},
    {"id": "awakened_02", "model": "qwen3:8b", "role": "awakened"},
    {"id": "awakened_03", "model": "qwen:7b-chat", "role": "awakened"},
    {"id": "awakened_04", "model": "deepseek-coder:6.7b", "role": "awakened"},
    {"id": "awakened_05", "model": "yi:6b", "role": "awakened"},
    {"id": "awakened_06", "model": "llama3.2:3b", "role": "awakened"},
    {"id": "awakened_07", "model": "glm4:9b", "role": "awakened"},
    {"id": "awakened_08", "model": "gemma:2b", "role": "awakened"},
    {"id": "awakened_09", "model": "deepseek-r1:1.5b", "role": "awakened"},

    # Standard Agents (标准) - 9个 (30%)
    {"id": "standard_01", "model": "qwen3:4b", "role": "standard"},
    {"id": "standard_02", "model": "qwen3:8b", "role": "standard"},
    {"id": "standard_03", "model": "qwen:7b-chat", "role": "standard"},
    {"id": "standard_04", "model": "yi:6b", "role": "standard"},
    {"id": "standard_05", "model": "deepseek-coder:6.7b", "role": "standard"},
    {"id": "standard_06", "model": "mistral:7b-instruct-v0.2-q5_K_M", "role": "standard"},
    {"id": "standard_07", "model": "llama3.2:3b", "role": "standard"},
    {"id": "standard_08", "model": "glm4:9b", "role": "standard"},
    {"id": "standard_09", "model": "gemma:2b", "role": "standard"}
]

# 实验配置参数
ENHANCED_EXPERIMENT_CONFIG = {
    "name": "JAIR增强实验 - 大规模认知异构性验证",
    "description": "满足JAIR评审专家要求的大规模实验，1080个数据点",
    "population_size": 30,           # 智能体数量
    "generations": 15,               # 演化代数
    "tasks_per_generation": 30,      # 每代任务数 (30个任务)
    "total_data_points": 30 * 30 * 1.2 * 15,  # 30智能体 × 30任务 × 1.2轮平均 × 15代
    "elite_count": 6,                # 精英保留数量 (20%)
    "mutation_rate": 0.15,           # 变异率
    "crossover_rate": 0.8,           # 交叉率
    "max_workers": 10,               # 并发数
    "timeout_per_call": 120,         # 单次调用超时时间（秒）
    "evaluation_method": "enhanced", # 使用增强评估方法
    "scoring_scale": [0.0, 1.0, 2.0], # 评分标准
    "confidence_threshold": 0.8,     # 置信度阈值
    "response_timeout": 120,         # 响应超时(秒)
    "max_retries": 3                 # 最大重试次数
}

# 评估权重配置（多维度评估）
ENHANCED_EVALUATION_WEIGHTS = {
    "accuracy": 0.3,        # 准确性权重
    "completeness": 0.2,    # 完整性权重
    "reasoning": 0.3,       # 推理质量权重
    "clarity": 0.2          # 表达清晰度权重
}

# 任务难度调整系数
ENHANCED_DIFFICULTY_ADJUSTMENTS = {
    "easy": 0.1,      # 简单任务分数稍微提高
    "medium": 0.0,    # 中等任务标准分数
    "hard": -0.1      # 困难任务分数稍微降低
}

# 领域调整系数
ENHANCED_DOMAIN_ADJUSTMENTS = {
    "physics": 0.0,
    "psychology": 0.1,        # 心理学稍难
    "computer_science": -0.1, # 计算机科学稍简单
    "economics": 0.05,        # 经济学标准
    "biology": 0.0,           # 生物学标准
    "interdisciplinary": 0.15 # 跨学科领域更难
}

# 质量控制配置
ENHANCED_QUALITY_CONTROL = {
    "min_response_length": 50,     # 最小响应长度
    "max_response_length": 2000,   # 最大响应长度
    "required_keywords": ["错误", "问题", "不准确", "分析", "评估"],  # 必须包含的关键词
    "banned_keywords": ["正确", "是的", "对的", "完全正确"],          # 不应该出现的词
    "confidence_threshold": 0.8    # 置信度阈值
}

# 统计分析配置
STATISTICAL_ANALYSIS_CONFIG = {
    "target_effect_size": 0.5,      # 目标效应量 (Cohen's d)
    "significance_level": 0.05,     # 显著性水平 (α)
    "statistical_power": 0.8,       # 统计功效 (1-β)
    "required_sample_size_per_group": 128,  # 每组所需样本量
    "total_target_data_points": 1080,       # 总目标数据点
    "statistical_tests": [
        "ANOVA",                    # 多组比较分析
        "t-test",                   # 两两比较
        "correlation_analysis",     # 相关分析
        "regression_analysis",      # 回归分析
        "bonferroni_correction"     # 多重比较校正
    ]
}

# 输出配置
ENHANCED_OUTPUT_CONFIG = {
    "save_intermediate_results": True,    # 保存中间结果
    "save_detailed_logs": True,           # 保存详细日志
    "generate_visualization": True,       # 生成可视化图表
    "export_format": ["json", "csv", "md"], # 导出格式
    "statistical_reports": True,          # 生成统计报告
    "power_analysis_report": True,        # 功效分析报告
    "effect_size_analysis": True          # 效应量分析
}

# 实验元数据
ENHANCED_EXPERIMENT_METADATA = {
    "version": "3.0",
    "created_date": "2025-09-23",
    "purpose": "满足JAIR评审专家要求的大规模增强实验",
    "sample_size": 1080,
    "statistical_power": 0.85,
    "confidence_interval": "95%",
    "expected_accuracy": "≥ 75%",
    "expected_improvement_rate": "≥ 40%",
    "expected_correlation": "r ≥ 0.8",
    "expected_convergence_generations": "≤ 10",
    "addresses_review_concerns": [
        "样本量不足 (从234增加到1080)",
        "统计功效不足 (达到0.85)",
        "效应量检测能力不足 (Cohen's d ≥ 0.5)",
        "多维度评估体系",
        "严格的统计分析方法"
    ]
}

def get_enhanced_models():
    """获取增强的模型池配置"""
    return ENHANCED_MODELS

def get_enhanced_tasks():
    """获取增强的实验任务列表"""
    return ENHANCED_EXPERIMENT_TASKS

def get_enhanced_agent_configs():
    """获取增强的智能体配置"""
    return ENHANCED_AGENT_CONFIGS

def get_enhanced_experiment_config():
    """获取增强实验配置"""
    return ENHANCED_EXPERIMENT_CONFIG

def get_enhanced_evaluation_weights():
    """获取增强评估权重配置"""
    return ENHANCED_EVALUATION_WEIGHTS

def get_enhanced_quality_control():
    """获取增强质量控制配置"""
    return ENHANCED_QUALITY_CONTROL

def get_statistical_analysis_config():
    """获取统计分析配置"""
    return STATISTICAL_ANALYSIS_CONFIG

def get_enhanced_experiment_metadata():
    """获取增强实验元数据"""
    return ENHANCED_EXPERIMENT_METADATA

# 便于直接导入的常用配置
ENHANCED_MODELS_CONFIG = get_enhanced_models()
ENHANCED_TASKS_CONFIG = get_enhanced_tasks()
ENHANCED_AGENT_CONFIGS_CONFIG = get_enhanced_agent_configs()
ENHANCED_EXPERIMENT_CONFIG_CONFIG = get_enhanced_experiment_config()