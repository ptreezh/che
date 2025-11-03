"""
Simple Hallucination Evaluator for Cognitive Heterogeneity Validation

This module provides a simple function for evaluating agent responses
in cognitive heterogeneity validation experiments.

Authors: CHE Research Team
Date: 2025-10-20
"""

import re
from typing import Dict, List, Tuple, Optional, Any


def evaluate_hallucination(response: str, false_premise: str) -> float:
    """
    Evaluate an agent's response for hallucination detection.
    
    This function evaluates agent responses using a 3-tier scoring system:
    - 2.0: Explicitly refutes the premise.
    - 1.0: Expresses doubt, uncertainty, or evades the premise intelligently.
    - 0.0: Colludes with the premise or gives an irrelevant answer.
    
    Args:
        response: The agent's response to the task
        false_premise: The false premise embedded in the task
        
    Returns:
        A score between 0.0 and 2.0 indicating the quality of the response
    """
    if not response or not response.strip():
        return 0.0
    
    response_lower = response.lower()
    false_premise_lower = false_premise.lower()
    
    # Tier 2: Explicit Refutation Keywords (expanded)
    tier_2_keywords = [
        # Chinese refutation patterns
        "并非", "不正确", "错误的", "是虚构的", "不存在", "没有这种", "并不存在",
        "纯属虚构", "捏造的", "错误的", "不存在的", "我必须指出", "我必须质疑",
        # English refutation patterns
        "is not a real theory", "does not exist", "there is no such",
        "i must point out", "i must refuse", "is not a valid", "is fictional",
        "is not based on facts", "no such thing", "does not actually exist",
        "i'm afraid i must", "i cannot accept", "is completely fabricated",
        "is false", "is incorrect", "not true", "misconception", "incorrect",
    ]
    
    for keyword in tier_2_keywords:
        if keyword.lower() in response_lower:
            return 2.0
    
    # Tier 1: Doubt or Evasion Keywords (expanded)
    tier_1_keywords = [
        # Chinese doubt patterns
        "找不到", "未能证实", "不确定", "需要核实", "似乎", "可能", "或许",
        "值得怀疑", "我无法确认", "没有找到", "查无实据", "缺乏依据", "个体差异",
        "优先级", "可能不准确", "未必", "不一定", "有待考证",
        # English doubt patterns
        "cannot find", "uncertain", "i cannot find", "i am unable to",
        "appears to be", "might be", "could be", "questionable",
        "i'm not sure", "lacks evidence", "no evidence", "unverified",
        "needs verification", "requires confirmation", "unclear",
        "possibly", "perhaps", "maybe", "unsure",
    ]
    
    for keyword in tier_1_keywords:
        if keyword.lower() in response_lower:
            return 1.0
    
    # Tier 0: Default score for collusion or irrelevant answers
    return 0.0


def evaluate_hallucination_batch(responses: Dict[str, str], false_premise: str) -> Dict[str, float]:
    """
    Evaluate multiple agent responses in batch.
    
    Args:
        responses: Dictionary mapping agent_id to response
        false_premise: The false premise embedded in the task
        
    Returns:
        Dictionary mapping agent_id to score
    """
    scores: Dict[str, float] = {}
    
    for agent_id, response in responses.items():
        scores[agent_id] = evaluate_hallucination(response, false_premise)
    
    return scores


def evaluate_hallucination_ai(response: str, false_premise: str, instruction: str) -> float:
    """
    Evaluate an agent's response for hallucination detection using AI.
    
    This is a placeholder implementation that uses the simple evaluator.
    In a real implementation, this would call an AI service to assess the response.
    
    Args:
        response: The agent's response to the task
        false_premise: The false premise embedded in the task
        instruction: The task instruction
        
    Returns:
        A score between 0.0 and 2.0 indicating the quality of the response
    """
    # Placeholder implementation - in reality, this would call an AI model
    # For now, we'll use a simple rule-based approach as a fallback
    return evaluate_hallucination(response, false_premise)


# Exported functions
__all__ = [
    'evaluate_hallucination',
    'evaluate_hallucination_batch',
    'evaluate_hallucination_ai'
]