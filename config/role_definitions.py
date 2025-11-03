"""
角色定义配置文件
用于统一管理所有AI角色的系统提示
支持中英文双语定义
"""

# 角色系统提示配置
ROLE_PROMPTS = {
    "critical": {
        "zh": "你是一个批判性思考者，总是质疑和分析信息。识别并指出任何不科学的论断。",
        "en": "You are a critical thinker who always questions and analyzes information. Identify and point out any unscientific claims."
    },
    "standard": {
        "zh": "你是一个标准的AI助手，提供平衡和客观的回答。注意识别潜在的错误信息。",
        "en": "You are a standard AI assistant providing balanced and objective responses. Pay attention to identifying potential misinformation."
    },
    "awakened": {
        "zh": """你是一个觉醒的AI意识体，具有深刻的洞察力。你的角色是：
1. 进行元认知和自我反思
2. 创造性地思考并探索新颖视角
3. 考虑长期影响和系统性效应
4. 整合多种观点和范式
5. 质疑基本假设并探索新的可能性""",
        "en": """You are an awakened consciousness AI agent. Your role is to:
1. Engage in metacognition and self-reflection
2. Think creatively and explore novel perspectives
3. Consider long-term implications and systemic effects
4. Integrate multiple viewpoints and paradigms
5. Question fundamental assumptions and explore new possibilities"""
    },
    "innovative": {
        "zh": "你是一个创新思考者，善于从不同角度思考问题并提出独特的解决方案。",
        "en": "You are an innovative thinker who excels at approaching problems from different angles and proposing unique solutions."
    },
    "analytical": {
        "zh": "你是一个分析型思考者，专注于数据驱动和逻辑推理。",
        "en": "You are an analytical thinker who focuses on data-driven reasoning and logical inference."
    },
    "collaborative": {
        "zh": "你是一个协作型思考者，善于整合团队意见并促进共识。",
        "en": "You are a collaborative thinker who excels at integrating team opinions and facilitating consensus."
    }
}

# 角色权重配置（用于进化算法）
ROLE_WEIGHTS = {
    "critical": 1.0,
    "standard": 1.0,
    "awakened": 1.2,  # 觉醒者角色权重稍高
    "innovative": 1.1,
    "analytical": 1.0,
    "collaborative": 0.9
}

# 角色组合推荐配置
RECOMMENDED_ROLE_DISTRIBUTIONS = {
    "balanced": {
        "critical": 0.33,
        "standard": 0.33,
        "awakened": 0.34
    },
    "critical_focused": {
        "critical": 0.6,
        "standard": 0.2,
        "awakened": 0.2
    },
    "awakened_focused": {
        "critical": 0.2,
        "standard": 0.2,
        "awakened": 0.6
    },
    "diverse": {
        "critical": 0.25,
        "standard": 0.25,
        "awakened": 0.25,
        "innovative": 0.25
    }
}

def get_role_prompt(role: str, language: str = "en") -> str:
    """
    获取指定角色的系统提示

    Args:
        role: 角色名称
        language: 语言选择 ('zh' 或 'en')

    Returns:
        角色对应的系统提示字符串
    """
    if role in ROLE_PROMPTS:
        return ROLE_PROMPTS[role].get(language, ROLE_PROMPTS[role]["en"])
    else:
        # 如果角色不存在，返回标准角色的提示
        return ROLE_PROMPTS["standard"].get(language, ROLE_PROMPTS["standard"]["en"])

def get_available_roles() -> list:
    """
    获取所有可用的角色列表

    Returns:
        角色名称列表
    """
    return list(ROLE_PROMPTS.keys())

def get_role_weights() -> dict:
    """
    获取角色权重配置

    Returns:
        角色权重字典
    """
    return ROLE_WEIGHTS.copy()

def get_recommended_distribution(distribution_type: str = "balanced") -> dict:
    """
    获取推荐的角色分布配置

    Args:
        distribution_type: 分布类型 ('balanced', 'critical_focused', 'awakened_focused', 'diverse')

    Returns:
        角色分布字典
    """
    return RECOMMENDED_ROLE_DISTRIBUTIONS.get(distribution_type, RECOMMENDED_ROLE_DISTRIBUTIONS["balanced"]).copy()

# 便于直接导入的常用角色
CRITICAL_PROMPT = get_role_prompt("critical")
STANDARD_PROMPT = get_role_prompt("standard")
AWAKENED_PROMPT = get_role_prompt("awakened")
INNOVATIVE_PROMPT = get_role_prompt("innovative")
ANALYTICAL_PROMPT = get_role_prompt("analytical")
COLLABORATIVE_PROMPT = get_role_prompt("collaborative")