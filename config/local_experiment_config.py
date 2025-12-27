"""
本地模型配置文件
将实验配置调整为使用本地支持的模型
"""

# 本地支持的模型配置
LOCAL_MODELS_CONFIG = {
    # 替代 qwen3:4b 和 qwen:0.5b
    "qwen_alternative": "qwen:7b-chat",
    
    # 替代 llama3.2:3b
    "llama_alternative": "llama3:latest",
    
    # 替代 gemma3:latest
    "gemma_alternative": "gemma:2b",
    
    # 本地原生支持
    "deepseek_coder": "deepseek-coder:6.7b"
}

# 更新后的4B模型异构组配置（使用本地支持的模型）
HETEROGENEOUS_GROUP_LOCAL = [
    {
        "id": "local_het_1",
        "model": LOCAL_MODELS_CONFIG["qwen_alternative"],
        "role": "critical",
        "description": "批判性思考的Qwen模型 (本地替代)"
    },
    {
        "id": "local_het_2",
        "model": LOCAL_MODELS_CONFIG["gemma_alternative"],
        "role": "standard",
        "description": "标准思考的Gemma模型 (本地替代)"
    },
    {
        "id": "local_het_3",
        "model": LOCAL_MODELS_CONFIG["deepseek_coder"],
        "role": "awakened",
        "description": "觉醒思考的DeepSeek模型 (本地支持)"
    }
]

# 更新后的同构组配置（使用本地支持的模型）
HOMOGENEOUS_GROUPS_LOCAL = [
    # 同构组1: 全部使用qwen:7b-chat
    [
        {
            "id": "local_hom_q1",
            "model": LOCAL_MODELS_CONFIG["qwen_alternative"],
            "role": "standard",
            "description": "标准qwen模型 (本地替代)"
        },
        {
            "id": "local_hom_q2",
            "model": LOCAL_MODELS_CONFIG["qwen_alternative"],
            "role": "standard",
            "description": "标准qwen模型 (本地替代)"
        },
        {
            "id": "local_hom_q3",
            "model": LOCAL_MODELS_CONFIG["qwen_alternative"],
            "role": "standard",
            "description": "标准qwen模型 (本地替代)"
        }
    ],
    # 同构组2: 全部使用gemma:2b
    [
        {
            "id": "local_hom_g1",
            "model": LOCAL_MODELS_CONFIG["gemma_alternative"],
            "role": "standard",
            "description": "标准gemma模型 (本地替代)"
        },
        {
            "id": "local_hom_g2",
            "model": LOCAL_MODELS_CONFIG["gemma_alternative"],
            "role": "standard",
            "description": "标准gemma模型 (本地替代)"
        },
        {
            "id": "local_hom_g3",
            "model": LOCAL_MODELS_CONFIG["gemma_alternative"],
            "role": "standard",
            "description": "标准gemma模型 (本地替代)"
        }
    ],
    # 同构组3: 全部使用deepseek-coder:6.7b
    [
        {
            "id": "local_hom_d1",
            "model": LOCAL_MODELS_CONFIG["deepseek_coder"],
            "role": "standard",
            "description": "标准deepseek模型 (本地支持)"
        },
        {
            "id": "local_hom_d2",
            "model": LOCAL_MODELS_CONFIG["deepseek_coder"],
            "role": "standard",
            "description": "标准deepseek模型 (本地支持)"
        },
        {
            "id": "local_hom_d3",
            "model": LOCAL_MODELS_CONFIG["deepseek_coder"],
            "role": "standard",
            "description": "标准deepseek模型 (本地支持)"
        }
    ]
]

# 本地实验配置参数
LOCAL_EXPERIMENT_CONFIG = {
    "name": "本地支持模型异构性对比实验",
    "description": "使用本地支持模型的异构vs同构对比实验",
    "repetitions": 5,  # 减少重复次数以加快测试
    "tasks_per_repetition": 3,  # 减少每轮任务数以加快测试
    "max_workers": 2,  # 减少并发数以节省资源
    "timeout_per_call": 60,  # 单次调用超时时间（秒）
    "strict_variable_control": True,
    "evaluation_method": "enhanced"
}

def get_local_heterogeneous_group():
    """获取本地支持的异构组配置"""
    return HETEROGENEOUS_GROUP_LOCAL

def get_local_homogeneous_groups():
    """获取本地支持的同构组配置列表"""
    return HOMOGENEOUS_GROUPS_LOCAL

def get_local_experiment_config():
    """获取本地实验配置"""
    return LOCAL_EXPERIMENT_CONFIG

# 便于直接导入的配置
LOCAL_HETEROGENEOUS_CONFIG = get_local_heterogeneous_group()
LOCAL_HOMOGENEOUS_CONFIG = get_local_homogeneous_groups()
LOCAL_EXPERIMENT_CONFIG_OBJ = get_local_experiment_config()