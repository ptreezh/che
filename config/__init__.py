"""
配置模块初始化文件
"""

from .role_definitions import (
    get_role_prompt,
    get_available_roles,
    get_role_weights,
    get_recommended_distribution,
    CRITICAL_PROMPT,
    STANDARD_PROMPT,
    AWAKENED_PROMPT,
    INNOVATIVE_PROMPT,
    ANALYTICAL_PROMPT,
    COLLABORATIVE_PROMPT
)

__all__ = [
    'get_role_prompt',
    'get_available_roles',
    'get_role_weights',
    'get_recommended_distribution',
    'CRITICAL_PROMPT',
    'STANDARD_PROMPT',
    'AWAKENED_PROMPT',
    'INNOVATIVE_PROMPT',
    'ANALYTICAL_PROMPT',
    'COLLABORATIVE_PROMPT'
]