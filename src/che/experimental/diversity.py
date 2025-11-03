"""
Diversity Metrics Calculator for Cognitive Heterogeneity Validation

This module provides functions to calculate diversity metrics in cognitive heterogeneity
validation experiments, measuring the variety and uniqueness in agent responses.

Authors: CHE Research Team
Date: 2025-10-19
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Set
import logging
from collections import Counter
import math

from ..core.agent import Agent

logger = logging.getLogger(__name__)


def calculate_cognitive_diversity_index(
    agents: List[Agent],
    response_vectors: Optional[List[np.ndarray]] = None
) -> float:
    """
    Calculate the cognitive diversity index of an agent population.
    
    The cognitive diversity index is calculated as:
    CDI = (number_of_distinct_cognitive_types / total_cognitive_types_possible) *
          sqrt(1 - (sum_of_squared_type_frequencies / population_size^2))
          
    Args:
        agents: List of agents in the population
        response_vectors: Optional list of response vectors for deeper diversity analysis
        
    Returns:
        Cognitive diversity index between 0.0 (no diversity) and 1.0 (maximum diversity)
    """
    if not agents:
        return 0.0
    
    # Extract agent types from their configuration
    agent_types = [agent.config.get('prompt_type', 'standard') for agent in agents]
    
    # Count type frequencies
    type_counts = Counter(agent_types)
    total_agents = len(agents)
    
    # Calculate type diversity (even distribution among types)
    if len(type_counts) == 1:
        # All agents are of the same type
        type_diversity = 0.0
    else:
        # Calculate Simpson's diversity index for type distribution
        numerator = sum(count * (count - 1) for count in type_counts.values())
        denominator = total_agents * (total_agents - 1)
        simpson_index = numerator / denominator if denominator > 0 else 0.0
        type_diversity = 1.0 - simpson_index
    
    # Calculate the proportion of distinct types
    possible_types = 3  # critical, awakened, standard
    distinct_type_proportion = len(type_counts) / possible_types
    
    # Combined diversity index
    diversity_index = distinct_type_proportion * type_diversity
    
    logger.debug(f"Calculated cognitive diversity index: {diversity_index:.4f} "
                f"(distinct types: {len(type_counts)}/{possible_types}, "
                f"type diversity: {type_diversity:.4f})")
    
    return diversity_index


def calculate_response_diversity(
    responses: List[str],
    method: str = "jaccard"
) -> float:
    """
    Calculate the diversity among agent responses.
    
    Args:
        responses: List of agent responses to the same task
        method: Method for calculating diversity ('jaccard', 'euclidean', 'cosine')
        
    Returns:
        Response diversity index between 0.0 (identical responses) and 1.0 (completely different responses)
    """
    if not responses or len(responses) < 2:
        return 0.0
    
    if method == "jaccard":
        return _calculate_jaccard_diversity(responses)
    elif method == "euclidean":
        return _calculate_euclidean_diversity(responses)
    elif method == "cosine":
        return _calculate_cosine_diversity(responses)
    else:
        raise ValueError(f"Unknown diversity calculation method: {method}")


def _calculate_jaccard_diversity(responses: List[str]) -> float:
    """
    Calculate Jaccard diversity between agent responses.
    
    Jaccard diversity = 1 - (intersection / union) of unique words
    
    Args:
        responses: List of agent responses to the same task
        
    Returns:
        Jaccard diversity index between 0.0 and 1.0
    """
    if len(responses) < 2:
        return 0.0
    
    # Tokenize responses into word sets
    response_sets = []
    for response in responses:
        words = set(response.lower().split())
        response_sets.append(words)
    
    # Calculate pairwise Jaccard similarities
    similarities = []
    for i in range(len(response_sets)):
        for j in range(i + 1, len(response_sets)):
            intersection = len(response_sets[i] & response_sets[j])
            union = len(response_sets[i] | response_sets[j])
            if union > 0:
                jaccard_similarity = intersection / union
                similarities.append(jaccard_similarity)
    
    if not similarities:
        return 0.0
    
    # Average similarity, then invert to get diversity
    avg_similarity = sum(similarities) / len(similarities)
    diversity = 1.0 - avg_similarity
    
    return diversity


def _calculate_euclidean_diversity(responses: List[str]) -> float:
    """
    Calculate Euclidean diversity between agent responses.
    
    Args:
        responses: List of agent responses to the same task
        
    Returns:
        Euclidean diversity index between 0.0 and 1.0
    """
    # Simplified implementation - in practice, you would use TF-IDF or embeddings
    return 0.5  # Placeholder implementation


def _calculate_cosine_diversity(responses: List[str]) -> float:
    """
    Calculate cosine diversity between agent responses.
    
    Args:
        responses: List of agent responses to the same task
        
    Returns:
        Cosine diversity index between 0.0 and 1.0
    """
    # Simplified implementation - in practice, you would use TF-IDF or embeddings
    return 0.5  # Placeholder implementation


def calculate_behavioral_diversity(
    agent_scores: Dict[str, float],
    threshold: float = 1.0
) -> float:
    """
    Calculate the diversity of agent behaviors based on their scores.
    
    Behavioral diversity considers how agents respond to tasks:
    - High scores (> threshold): Detected false premise (2.0)
    - Mid scores (0.5-1.0): Expressed doubt (1.0)
    - Low scores (< 0.5): Accepted false premise (0.0)
    
    Args:
        agent_scores: Dictionary mapping agent_id to score
        threshold: Threshold for distinguishing score categories (default: 1.0)
        
    Returns:
        Behavioral diversity index between 0.0 (identical behaviors) and 1.0 (diverse behaviors)
    """
    if not agent_scores:
        return 0.0
    
    # Categorize scores
    high_count = 0  # Score >= threshold (detects false premises)
    mid_count = 0   # Score between 0.5 and threshold (expresses doubt)
    low_count = 0   # Score < 0.5 (accepts false premises)
    
    for score in agent_scores.values():
        if score >= threshold:
            high_count += 1
        elif score >= 0.5:
            mid_count += 1
        else:
            low_count += 1
    
    total_agents = len(agent_scores)
    
    # Calculate Simpson's diversity index for behavioral categories
    if total_agents <= 1:
        return 0.0
    
    numerator = high_count * (high_count - 1) + mid_count * (mid_count - 1) + low_count * (low_count - 1)
    denominator = total_agents * (total_agents - 1)
    
    if denominator == 0:
        return 0.0
    
    simpson_index = numerator / denominator
    behavioral_diversity = 1.0 - simpson_index
    
    logger.debug(f"Calculated behavioral diversity index: {behavioral_diversity:.4f} "
                f"(high: {high_count}, mid: {mid_count}, low: {low_count})")
    
    return behavioral_diversity


def calculate_generational_diversity_trend(
    diversity_history: List[float]
) -> Dict[str, Any]:
    """
    Calculate trends in diversity over generations.
    
    Args:
        diversity_history: List of diversity metrics across generations
        
    Returns:
        Dictionary containing trend analysis
    """
    if len(diversity_history) < 2:
        return {
            'trend': 0.0,
            'average': 0.0 if not diversity_history else diversity_history[0],
            'volatility': 0.0,
            'interpretation': 'Insufficient data points for trend analysis'
        }
    
    # Calculate linear trend (slope)
    x_values = np.arange(len(diversity_history))
    slope, intercept = np.polyfit(x_values, diversity_history, 1)
    
    # Calculate volatility (standard deviation)
    volatility = np.std(diversity_history)
    
    # Calculate average diversity
    average = np.mean(diversity_history)
    
    # Interpret trend
    if slope > 0.01:
        interpretation = 'increasing'
    elif slope < -0.01:
        interpretation = 'decreasing'
    else:
        interpretation = 'stable'
    
    return {
        'trend': slope,
        'average': average,
        'volatility': volatility,
        'interpretation': interpretation
    }


def calculate_shannon_entropy(agent_types: List[str]) -> float:
    """
    Calculate Shannon entropy of agent types as a measure of diversity.
    
    Args:
        agent_types: List of agent types in the population
        
    Returns:
        Shannon entropy value (higher values indicate higher diversity)
    """
    if not agent_types:
        return 0.0
    
    # Count type frequencies
    type_counts = Counter(agent_types)
    total = len(agent_types)
    
    # Calculate entropy
    probabilities = [count / total for count in type_counts.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    
    # Normalize by maximum possible entropy
    max_entropy = math.log2(len(type_counts))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    
    return normalized_entropy


def validate_cognitive_independence_requirement(
    correlation: float,
    p_value: float,
    required_correlation: float = 0.6,
    required_p_value: float = 0.01
) -> bool:
    """
    Validate if cognitive independence correlation meets constitutional requirements.
    
    Args:
        correlation: Pearson correlation coefficient between diversity and performance
        p_value: P-value for correlation
        required_correlation: Minimum required correlation (default: 0.6)
        required_p_value: Maximum allowed p-value (default: 0.01)
        
    Returns:
        True if requirements are met, False otherwise
    """
    meets_correlation = correlation >= required_correlation
    meets_significance = p_value < required_p_value
    
    logger.info(f"Cognitive independence validation - "
                f"Correlation: {correlation:.3f} (required: ≥{required_correlation}), "
                f"P-value: {p_value:.3f} (required: <{required_p_value}), "
                f"Meets requirements: {meets_correlation and meets_significance}")
    
    return meets_correlation and meets_significance


# Exported functions
__all__ = [
    'calculate_cognitive_diversity_index',
    'calculate_response_diversity',
    '_calculate_jaccard_diversity',
    '_calculate_euclidean_diversity',
    '_calculate_cosine_diversity',
    'calculate_behavioral_diversity',
    'calculate_generational_diversity_trend',
    'calculate_shannon_entropy',
    'validate_cognitive_independence_requirement'
]