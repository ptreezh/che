"""
Correlation Analysis Between Diversity and Performance

This module provides functions to analyze the correlation between cognitive diversity
and performance improvement in cognitive heterogeneity validation experiments.

Authors: CHE Research Team
Date: 2025-10-19
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Any, Optional
import logging

from .diversity import calculate_cognitive_diversity_index, calculate_behavioral_diversity
from .performance import PerformanceTracker

logger = logging.getLogger(__name__)


def calculate_diversity_performance_correlation(
    diversity_metrics: List[float],
    performance_metrics: List[float]
) -> Dict[str, Any]:
    """
    Calculate the correlation between diversity and performance metrics.

    This function validates the cognitive independence hypothesis that
    diversity correlates with performance (r ≥ 0.6, p < 0.01).

    Args:
        diversity_metrics: Diversity measurements across generations
        performance_metrics: Performance measurements across generations

    Returns:
        Dictionary containing correlation analysis results
    """
    if len(diversity_metrics) != len(performance_metrics):
        raise ValueError("Diversity and performance metrics must have the same length")

    if len(diversity_metrics) < 2:
        raise ValueError("Need at least 2 data points for correlation analysis")

    # Convert to numpy arrays
    diversity_array = np.array(diversity_metrics)
    performance_array = np.array(performance_metrics)

    # Calculate Pearson correlation coefficient
    pearson_r, pearson_p = stats.pearsonr(diversity_array, performance_array)

    # Calculate Spearman rank correlation (non-parametric)
    spearman_rho, spearman_p = stats.spearmanr(diversity_array, performance_array)

    # Calculate Kendall's tau (robust to outliers)
    kendall_tau, kendall_p = stats.kendalltau(diversity_array, performance_array)

    # Calculate coefficient of determination (r-squared)
    r_squared = pearson_r ** 2

    # Effect size interpretation
    if abs(pearson_r) < 0.1:
        effect_size = "negligible"
    elif abs(pearson_r) < 0.3:
        effect_size = "small"
    elif abs(pearson_r) < 0.5:
        effect_size = "medium"
    else:
        effect_size = "large"

    # Confidence interval for Pearson correlation (approximate)
    n = len(diversity_metrics)
    if n > 3:
        # Fisher transformation for confidence interval
        fisher_z = np.arctanh(pearson_r)
        se = 1 / np.sqrt(n - 3)
        # 95% confidence interval
        z_critical = 1.96
        ci_lower = np.tanh(fisher_z - z_critical * se)
        ci_upper = np.tanh(fisher_z + z_critical * se)
    else:
        ci_lower, ci_upper = np.nan, np.nan

    return {
        'pearson_r': pearson_r,
        'pearson_p_value': pearson_p,
        'spearman_rho': spearman_rho,
        'spearman_p_value': spearman_p,
        'kendall_tau': kendall_tau,
        'kendall_p_value': kendall_p,
        'r_squared': r_squared,
        'effect_size': effect_size,
        'confidence_interval': (ci_lower, ci_upper),
        'n': n,
        'interpretation': f"Correlation r={pearson_r:.3f}, p={pearson_p:.3f}, effect={effect_size}"
    }


def calculate_correlation_significance(
    correlation: float,
    sample_size: int,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Calculate the statistical significance of a correlation coefficient.

    Args:
        correlation: Correlation coefficient
        sample_size: Number of data points
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary containing significance test results
    """
    if sample_size < 3:
        raise ValueError("Sample size must be at least 3")

    # Calculate t-statistic
    degrees_freedom = sample_size - 2
    t_stat = correlation * np.sqrt(degrees_freedom / (1 - correlation**2))

    # Calculate p-value (two-tailed)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), degrees_freedom))

    # Determine significance
    is_significant = p_value < alpha

    return {
        't_statistic': t_stat,
        'degrees_freedom': degrees_freedom,
        'p_value': p_value,
        'is_significant': is_significant,
        'alpha': alpha,
        'critical_value': stats.t.ppf(1 - alpha/2, degrees_freedom)
    }


def calculate_partial_correlation(
    x: List[float],
    y: List[float],
    z: List[float]
) -> Dict[str, Any]:
    """
    Calculate partial correlation between x and y controlling for z.

    Args:
        x: First variable
        y: Second variable
        z: Control variable

    Returns:
        Dictionary containing partial correlation results
    """
    if not (len(x) == len(y) == len(z)):
        raise ValueError("All input lists must have the same length")

    if len(x) < 3:
        raise ValueError("Need at least 3 data points for partial correlation")

    # Calculate correlations
    r_xy, _ = stats.pearsonr(x, y)
    r_xz, _ = stats.pearsonr(x, z)
    r_yz, _ = stats.pearsonr(y, z)

    # Calculate partial correlation
    r_xy_z = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2))

    # Calculate significance
    n = len(x)
    degrees_freedom = n - 3  # controlling for 1 variable
    t_stat = r_xy_z * np.sqrt(degrees_freedom / (1 - r_xy_z**2))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), degrees_freedom))

    return {
        'partial_correlation': r_xy_z,
        'original_correlation': r_xy,
        'control_correlations': {'x_z': r_xz, 'y_z': r_yz},
        't_statistic': t_stat,
        'p_value': p_value,
        'degrees_freedom': degrees_freedom,
        'interpretation': f"Partial r={r_xy_z:.3f} controlling for z"
    }


def validate_cognitive_independence_correlation(
    diversity_history: List[float],
    performance_history: List[float]
) -> Dict[str, Any]:
    """
    Validate cognitive independence correlation requirements.

    This function checks if the correlation meets the constitutional requirements:
    - r ≥ 0.6 (strong correlation)
    - p < 0.01 (high statistical significance)

    Args:
        diversity_history: Historical diversity measurements
        performance_history: Historical performance measurements

    Returns:
        Dictionary containing validation results
    """
    if len(diversity_history) != len(performance_history):
        raise ValueError("Diversity and performance histories must have same length")

    if len(diversity_history) < 3:
        raise ValueError("Need at least 3 data points for meaningful correlation analysis")

    # Calculate correlation
    correlation_results = calculate_diversity_performance_correlation(
        diversity_history, performance_history
    )

    # Extract key metrics
    pearson_r = correlation_results['pearson_r']
    pearson_p = correlation_results['pearson_p_value']

    # Validate requirements
    meets_correlation_requirement = pearson_r >= 0.6
    meets_significance_requirement = pearson_p < 0.01
    meets_constitutional_requirements = meets_correlation_requirement and meets_significance_requirement

    return {
        'correlation_coefficient': pearson_r,
        'p_value': pearson_p,
        'meets_correlation_requirement': meets_correlation_requirement,
        'meets_significance_requirement': meets_significance_requirement,
        'meets_constitutional_requirements': meets_constitutional_requirements,
        'full_correlation_analysis': correlation_results,
        'summary': f"r={pearson_r:.3f}, p={pearson_p:.3f}, meets_requirements={meets_constitutional_requirements}"
    }


# Exported functions
__all__ = [
    'calculate_diversity_performance_correlation',
    'calculate_correlation_significance',
    'calculate_partial_correlation',
    'validate_cognitive_independence_correlation'
]