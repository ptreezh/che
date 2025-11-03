"""
Enhanced Correlation Analysis for Cognitive Heterogeneity Validation

This module provides enhanced functions for analyzing the correlation between 
cognitive diversity and performance improvement in cognitive heterogeneity experiments.

Authors: CHE Research Team
Date: 2025-10-31
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


def calculate_pearson_correlation(x: List[float], y: List[float]) -> Dict[str, float]:
    """
    Calculate Pearson correlation coefficient with confidence interval.
    
    Args:
        x: List of diversity metrics
        y: List of performance metrics
        
    Returns:
        Dictionary containing correlation coefficient, p-value, and confidence interval
    """
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Input lists must have same length and contain at least 2 elements")
    
    # Calculate Pearson correlation
    correlation, p_value = stats.pearsonr(x, y)
    
    # Calculate confidence interval using Fisher transformation
    n = len(x)
    if n < 3:
        # Cannot calculate confidence interval with less than 3 points
        ci_lower, ci_upper = np.nan, np.nan
    else:
        # Fisher transformation
        fisher_z = np.arctanh(correlation)
        standard_error = 1 / np.sqrt(n - 3)
        
        # 99% confidence interval (z = 2.576 for 99% confidence)
        z_critical = 2.576
        ci_lower_z = fisher_z - z_critical * standard_error
        ci_upper_z = fisher_z + z_critical * standard_error
        
        # Back-transform to correlation scale
        ci_lower = np.tanh(ci_lower_z)
        ci_upper = np.tanh(ci_upper_z)
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n': n,
        'method': 'pearson'
    }


def calculate_spearman_correlation(x: List[float], y: List[float]) -> Dict[str, float]:
    """
    Calculate Spearman rank correlation coefficient with confidence interval.
    
    Args:
        x: List of diversity metrics
        y: List of performance metrics
        
    Returns:
        Dictionary containing correlation coefficient, p-value, and confidence interval
    """
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Input lists must have same length and contain at least 2 elements")
    
    # Calculate Spearman correlation
    correlation, p_value = stats.spearmanr(x, y)
    
    # Calculate confidence interval using Fisher transformation
    n = len(x)
    if n < 3:
        # Cannot calculate confidence interval with less than 3 points
        ci_lower, ci_upper = np.nan, np.nan
    else:
        # Fisher transformation
        fisher_z = np.arctanh(correlation)
        standard_error = 1 / np.sqrt(n - 3)
        
        # 99% confidence interval (z = 2.576 for 99% confidence)
        z_critical = 2.576
        ci_lower_z = fisher_z - z_critical * standard_error
        ci_upper_z = fisher_z + z_critical * standard_error
        
        # Back-transform to correlation scale
        ci_lower = np.tanh(ci_lower_z)
        ci_upper = np.tanh(ci_upper_z)
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n': n,
        'method': 'spearman'
    }


def calculate_kendall_correlation(x: List[float], y: List[float]) -> Dict[str, float]:
    """
    Calculate Kendall tau correlation coefficient with confidence interval.
    
    Args:
        x: List of diversity metrics
        y: List of performance metrics
        
    Returns:
        Dictionary containing correlation coefficient, p-value, and confidence interval
    """
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Input lists must have same length and contain at least 2 elements")
    
    # Calculate Kendall correlation
    correlation, p_value = stats.kendalltau(x, y)
    
    # Calculate confidence interval using Fisher transformation
    n = len(x)
    if n < 3:
        # Cannot calculate confidence interval with less than 3 points
        ci_lower, ci_upper = np.nan, np.nan
    else:
        # Fisher transformation
        fisher_z = np.arctanh(correlation)
        standard_error = 1 / np.sqrt(n - 3)
        
        # 99% confidence interval (z = 2.576 for 99% confidence)
        z_critical = 2.576
        ci_lower_z = fisher_z - z_critical * standard_error
        ci_upper_z = fisher_z + z_critical * standard_error
        
        # Back-transform to correlation scale
        ci_lower = np.tanh(ci_lower_z)
        ci_upper = np.tanh(ci_upper_z)
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n': n,
        'method': 'kendall'
    }


def calculate_correlation_effect_size(correlation: float) -> Dict[str, float]:
    """
    Calculate effect size interpretation for correlation coefficient.
    
    Args:
        correlation: Correlation coefficient
        
    Returns:
        Dictionary containing effect size classification and interpretation
    """
    # Absolute value for interpretation
    abs_corr = abs(correlation)
    
    # Cohen's guidelines for correlation effect sizes
    if abs_corr < 0.1:
        effect_size = "negligible"
        interpretation = "very weak association"
    elif abs_corr < 0.3:
        effect_size = "small"
        interpretation = "weak association"
    elif abs_corr < 0.5:
        effect_size = "medium"
        interpretation = "moderate association"
    else:
        effect_size = "large"
        interpretation = "strong association"
    
    # Calculate r-squared (proportion of variance explained)
    r_squared = correlation ** 2
    
    return {
        'effect_size': effect_size,
        'interpretation': interpretation,
        'r_squared': r_squared,
        'correlation': correlation
    }


def validate_cognitive_independence(
    diversity_metrics: List[float], 
    performance_metrics: List[float]
) -> Dict[str, Any]:
    """
    Validate cognitive independence with comprehensive correlation analysis.
    
    This function validates the constitutional requirement that cognitive independence
    correlation reaches r ≥ 0.6 with statistical significance p < 0.01.
    
    Args:
        diversity_metrics: List of diversity measurements across generations
        performance_metrics: List of performance measurements across generations
        
    Returns:
        Dictionary containing validation results and all correlation measures
    """
    if len(diversity_metrics) != len(performance_metrics):
        raise ValueError("Diversity and performance metrics must have same length")
    
    if len(diversity_metrics) < 3:
        raise ValueError("Need at least 3 data points for meaningful correlation analysis")
    
    # Calculate all correlation measures
    pearson_results = calculate_pearson_correlation(diversity_metrics, performance_metrics)
    spearman_results = calculate_spearman_correlation(diversity_metrics, performance_metrics)
    kendall_results = calculate_kendall_correlation(diversity_metrics, performance_metrics)
    
    # Extract primary correlation (Pearson)
    primary_correlation = pearson_results['correlation']
    primary_p_value = pearson_results['p_value']
    primary_ci_lower = pearson_results['ci_lower']
    primary_ci_upper = pearson_results['ci_upper']
    
    # Effect size analysis
    effect_size_results = calculate_correlation_effect_size(primary_correlation)
    
    # Constitutional validation
    meets_correlation_requirement = primary_correlation >= 0.6
    meets_significance_requirement = primary_p_value < 0.01
    meets_confidence_requirement = not np.isnan(primary_ci_lower) and primary_ci_lower >= 0.5
    
    # Overall validation
    meets_constitutional_requirements = (
        meets_correlation_requirement and 
        meets_significance_requirement and 
        meets_confidence_requirement
    )
    
    return {
        'pearson': pearson_results,
        'spearman': spearman_results,
        'kendall': kendall_results,
        'effect_size': effect_size_results,
        'validation': {
            'meets_correlation_requirement': meets_correlation_requirement,  # r ≥ 0.6
            'meets_significance_requirement': meets_significance_requirement,  # p < 0.01
            'meets_confidence_requirement': meets_confidence_requirement,  # CI lower bound ≥ 0.5
            'meets_constitutional_requirements': meets_constitutional_requirements,
            'interpretation': (
                f"Cognitive independence {'VALIDATED' if meets_constitutional_requirements else 'NOT VALIDATED'}: "
                f"r={primary_correlation:.3f} ({'≥ 0.6' if meets_correlation_requirement else '< 0.6'}), "
                f"p={primary_p_value:.3f} ({'< 0.01' if meets_significance_requirement else '≥ 0.01'}), "
                f"99% CI=[{primary_ci_lower:.3f}, {primary_ci_upper:.3f}]"
            )
        }
    }


# Exported functions
__all__ = [
    'calculate_pearson_correlation',
    'calculate_spearman_correlation',
    'calculate_kendall_correlation',
    'calculate_correlation_effect_size',
    'validate_cognitive_independence'
]