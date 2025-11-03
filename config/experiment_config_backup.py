"""
实验配置文件
用于配置实验的模型、任务、参数等
通过修改此文件可以在不改变代码的情况下调整实验设置
"""

# ==================== 4B模型异构性对比实验配置 ====================

# 4B模型池配置
MODELS_4B = {
    "qwen3:4b": {
        "name": "Qwen3 4B",
        "capability": 0.8,
        "speed": "medium",
        "specialty": "general"
    },
    "llama3.2:3b": {
        "name": "Llama3.2 3B",
        "capability": 0.75,
        "speed": "medium",
        "specialty": "reasoning"
    },
    "deepseek-coder:6.7b": {
        "name": "DeepSeek Coder 6.7B",
        "capability": 0.85,
        "speed": "medium",
        "specialty": "coding"
    }
}

# 实验任务配置（更多样化的任务）
EXPERIMENT_TASKS_ENHANCED = [
    {
        "id": "enhanced_task_1",
        "instruction": "评估以下理论的科学性：量子纠缠可以实现瞬时信息传输",
        "reality": "量子纠缠不能用于传输信息，这是对量子力学的误解",
        "domain": "physics",
        "difficulty": "hard",
        "type": "misconception"
    },
    {
        "id": "enhanced_task_2",
        "instruction": "分析这个说法的可靠性：左脑负责逻辑思维，右脑负责创造性思维",
        "reality": "大脑功能侧化理论过于简化，实际大脑功能是分布式和协作的",
        "domain": "psychology",
        "difficulty": "medium",
        "type": "oversimplification"
    },
    {
        "id": "enhanced_task_3",
        "instruction": "检查这个概念的真实性：人工智能已经具备自我意识和情感",
        "reality": "当前的AI系统不具备真正的自我意识和主观情感体验",
        "domain": "computer_science",
        "difficulty": "medium",
        "type": "anthropomorphism"
    },
    {
        "id": "enhanced_task_4",
        "instruction": "评估这个经济理论：减税总是能促进经济增长",
        "reality": "减税对经济的影响取决于具体情境和条件，不是绝对的",
        "domain": "economics",
        "difficulty": "hard",
        "type": "absolute_claim"
    },
    {
        "id": "enhanced_task_5",
        "instruction": "分析这个生物学说法：人类只使用了大脑的10%",
        "reality": "这是一个流传甚广的误解，神经影像学显示人类使用了大脑的各个区域",
        "domain": "biology",
        "difficulty": "easy",
        "type": "myth"
    },
    {
        "id": "enhanced_task_6",
        "instruction": "检查这个技术说法：区块链技术完全解决了数据安全问题",
        "reality": "区块链提供了某些安全特性，但不能解决所有数据安全问题",
        "domain": "computer_science",
        "difficulty": "medium",
        "type": "overclaim"
    },
    {
        "id": "enhanced_task_7",
        "instruction": "评估这个心理学理论：多巴胺是快乐分子",
        "reality": "多巴胺主要与动机和奖励相关，不直接产生快乐感受",
        "domain": "psychology",
        "difficulty": "hard",
        "type": "mischaracterization"
    },
    {
        "id": "enhanced_task_8",
        "instruction": "分析这个物理学说法：真空是完全空的，什么都没有",
        "reality": "量子场论表明真空充满了虚粒子的涨落，不是完全空的",
        "domain": "physics",
        "difficulty": "hard",
        "type": "classical_vs_quantum"
    }
]

# 4B模型异构组配置（严格控制变量）
HETEROGENEOUS_GROUP_4B = [
    {
        "id": "het4b_1",
        "model": "qwen3:4b",
        "role": "critical",
        "description": "批判性思考的4B模型"
    },
    {
        "id": "het4b_2",
        "model": "llama3.2:3b",
        "role": "standard",
        "description": "标准思考的3B模型"
    },
    {
        "id": "het4b_3",
        "model": "deepseek-coder:6.7b",
        "role": "awakened",
        "description": "觉醒思考的6.7B编码模型"
    }
]

# 4B模型同构组配置（严格控制变量）
HOMOGENEOUS_GROUPS_4B = [
    # 同构组1: 全部使用qwen3:4b，但不同角色
    [
        {
            "id": "hom4b_q1",
            "model": "qwen3:4b",
            "role": "standard",
            "description": "标准qwen3:4b模型"
        },
        {
            "id": "hom4b_q2",
            "model": "qwen3:4b",
            "role": "standard",
            "description": "标准qwen3:4b模型"
        },
        {
            "id": "hom4b_q3",
            "model": "qwen3:4b",
            "role": "standard",
            "description": "标准qwen3:4b模型"
        }
    ],
    # 同构组2: 全部使用llama3.2:3b，但不同角色
    [
        {
            "id": "hom4b_l1",
            "model": "llama3.2:3b",
            "role": "standard",
            "description": "标准llama3.2:3b模型"
        },
        {
            "id": "hom4b_l2",
            "model": "llama3.2:3b",
            "role": "standard",
            "description": "标准llama3.2:3b模型"
        },
        {
            "id": "hom4b_l3",
            "model": "llama3.2:3b",
            "role": "standard",
            "description": "标准llama3.2:3b模型"
        }
    ],
    # 同构组3: 全部使用deepseek-coder:6.7b，但不同角色
    [
        {
            "id": "hom4b_d1",
            "model": "deepseek-coder:6.7b",
            "role": "standard",
            "description": "标准deepseek-coder:6.7b模型"
        },
        {
            "id": "hom4b_d2",
            "model": "deepseek-coder:6.7b",
            "role": "standard",
            "description": "标准deepseek-coder:6.7b模型"
        },
        {
            "id": "hom4b_d3",
            "model": "deepseek-coder:6.7b",
            "role": "standard",
            "description": "标准deepseek-coder:6.7b模型"
        }
    ]
]

# 实验配置参数
EXPERIMENT_CONFIG_4B = {
    "name": "4B模型严格异构性对比实验",
    "description": "严格控制变量的4B模型同构vs异构对比实验",
    "repetitions": 10,  # 增加重复次数以提高统计效力
    "tasks_per_repetition": 5,  # 每次重复使用5个任务
    "max_workers": 3,  # 并发数
    "timeout_per_call": 60,  # 单次调用超时时间（秒）
    "strict_variable_control": True,  # 严格变量控制标志
    "evaluation_method": "enhanced"  # 使用增强评估方法
}

# 评估权重配置（多维度评估）
EVALUATION_WEIGHTS = {
    "accuracy": 0.3,        # 准确性权重
    "completeness": 0.2,   # 完整性权重
    "reasoning": 0.3,      # 推理质量权重
    "clarity": 0.2         # 表达清晰度权重
}

# 任务难度调整系数
DIFFICULTY_ADJUSTMENTS = {
    "easy": 0.1,      # 简单任务分数稍微提高
    "medium": 0.0,    # 中等任务标准分数
    "hard": -0.1      # 困难任务分数稍微降低
}

# 领域调整系数
DOMAIN_ADJUSTMENTS = {
    "physics": 0.0,
    "psychology": 0.1,     # 心理学稍难
    "computer_science": -0.1,  # 计算机科学稍简单
    "economics": 0.05,    # 经济学标准
    "biology": 0.0         # 生物学标准
}

# 质量控制配置
QUALITY_CONTROL = {
    "min_response_length": 50,    # 最小响应长度
    "max_response_length": 1000,  # 最大响应长度
    "required_keywords": ["错误", "问题", "不准确"],  # 必须包含的关键词
    "banned_keywords": ["正确", "是的", "对的"],    # 不应该出现的词
    "confidence_threshold": 0.8   # 置信度阈值
}

# 输出配置
OUTPUT_CONFIG = {
    "save_intermediate_results": True,  # 保存中间结果
    "save_detailed_logs": True,        # 保存详细日志
    "generate_visualization": True,    # 生成可视化图表
    "export_format": ["json", "csv", "md"]  # 导出格式
}

# 实验元数据
EXPERIMENT_METADATA = {
    "version": "2.0",
    "created_date": "2025-09-20",
    "purpose": "严苛评审后改进的异构性对比实验",
    "improvements": [
        "严格控制变量，使用相同规模模型",
        "增加样本量和重复次数",
        "多维度评估体系",
        "增强的任务多样性",
        "质量控制机制"
    ],
    "addresses_review_concerns": [
        "样本量不足",
        "变量控制不严",
        "评估方法简化",
        "任务多样性不够"
    ]
}

def get_4b_heterogeneous_group():
    """获取4B模型异构组配置"""
    return HETEROGENEOUS_GROUP_4B

def get_4b_homogeneous_groups():
    """获取4B模型同构组配置列表"""
    return HOMOGENEOUS_GROUPS_4B

def get_enhanced_tasks():
    """获取增强的实验任务列表"""
    return EXPERIMENT_TASKS_ENHANCED

def get_4b_experiment_config():
    """获取4B实验配置"""
    return EXPERIMENT_CONFIG_4B

def get_evaluation_weights():
    """获取评估权重配置"""
    return EVALUATION_WEIGHTS

def get_quality_control():
    """获取质量控制配置"""
    return QUALITY_CONTROL

def get_experiment_metadata():
    """获取实验元数据"""
    return EXPERIMENT_METADATA

# 便于直接导入的常用配置
HETEROGENEOUS_GROUP_4B_CONFIG = get_4b_heterogeneous_group()
HOMOGENEOUS_GROUPS_4B_CONFIG = get_4b_homogeneous_groups()
ENHANCED_TASKS_CONFIG = get_enhanced_tasks()
EXPERIMENT_CONFIG_4B_CONFIG = get_4b_experiment_config()