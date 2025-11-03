"""
Enhanced Statistical Validation for Cognitive Heterogeneity Validation

This module provides comprehensive statistical validation functions for cognitive heterogeneity experiments,
including power analysis, effect size calculation, and multiple comparison corrections.

Authors: CHE Research Team
Date: 2025-10-31
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


def calculate_cohens_d(x1: List[float], x2: List[float]) -> Dict[str, float]:
    """
    Calculate Cohen's d effect size for comparing two groups.
    
    Args:
        x1: First group of measurements (e.g., heterogeneous performance)
        x2: Second group of measurements (e.g., homogeneous performance)
        
    Returns:
        Dictionary containing effect size and interpretation
    """
    if len(x1) < 2 or len(x2) < 2:
        raise ValueError("Each group must have at least 2 elements")
    
    # Convert to numpy arrays
    x1_array = np.array(x1)
    x2_array = np.array(x2)
    
    # Calculate means and standard deviations
    mean1 = np.mean(x1_array)
    mean2 = np.mean(x2_array)
    std1 = np.std(x1_array, ddof=1)  # Sample standard deviation
    std2 = np.std(x2_array, ddof=1)  # Sample standard deviation
    
    n1 = len(x1)
    n2 = len(x2)
    
    # Calculate pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    # Calculate Cohen's d
    cohens_d = (mean1 - mean2) / pooled_std
    
    # Calculate confidence interval for Cohen's d using Hedges' g correction
    # Calculate Hedges' g (bias-corrected effect size)
    j_factor = 1 - 3 / (4 * (n1 + n2) - 9)
    hedges_g = j_factor * cohens_d
    
    # Standard error of Hedges' g
    se_g = np.sqrt((n1 + n2) / (n1 * n2) + hedges_g**2 / (2 * (n1 + n2)))
    
    # 95% confidence interval using normal approximation
    z_critical = 1.96  # For 95% CI
    ci_lower = hedges_g - z_critical * se_g
    ci_upper = hedges_g + z_critical * se_g
    
    # Interpretation
    abs_d = abs(hedges_g)
    if abs_d < 0.2:
        effect_size = "negligible"
        interpretation = "very small effect"
    elif abs_d < 0.5:
        effect_size = "small"
        interpretation = "small effect"
    elif abs_d < 0.8:
        effect_size = "medium"
        interpretation = "medium effect"
    else:
        effect_size = "large"
        interpretation = "large effect"
    
    return {
        'cohens_d': cohens_d,
        'hedges_g': hedges_g,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'interpretation': interpretation,
        'effect_size': effect_size,
        'mean1': mean1,
        'mean2': mean2,
        'std1': std1,
        'std2': std2,
        'n1': n1,
        'n2': n2
    }


def calculate_statistical_power(effect_size: float, n1: int, n2: int, alpha: float = 0.05) -> float:
    """
    Calculate statistical power for a two-sample t-test.
    
    Args:
        effect_size: Cohen's d effect size
        n1: Sample size of group 1
        n2: Sample size of group 2
        alpha: Significance level (default 0.05)
        
    Returns:
        Statistical power (probability of detecting effect if it exists)
    """
    from scipy.stats import nct
    
    # Calculate harmonic mean of sample sizes
    n_harmonic = 2 * n1 * n2 / (n1 + n2)
    
    # Calculate non-centrality parameter
    lambda_param = effect_size * np.sqrt(n_harmonic / 2)
    
    # Calculate critical t-value for the given alpha
    df = n1 + n2 - 2
    t_critical = stats.t.ppf(1 - alpha/2, df)
    
    # Calculate power using the non-central t-distribution
    power = 1 - nct.cdf(t_critical, df, -lambda_param) + nct.cdf(-t_critical, df, -lambda_param)
    
    return power


def calculate_sample_size_for_power(effect_size: float, desired_power: float = 0.8, alpha: float = 0.05) -> int:
    """
    Calculate required sample size to achieve desired statistical power.
    
    Args:
        effect_size: Expected Cohen's d effect size
        desired_power: Desired statistical power (default 0.8)
        alpha: Significance level (default 0.05)
        
    Returns:
        Required sample size per group
    """
    # Start with a reasonable guess
    n = 10
    power = 0.0
    
    # Iteratively increase sample size until desired power is achieved
    while power < desired_power and n < 10000:  # Prevent infinite loops
        power = calculate_statistical_power(effect_size, n, n, alpha)
        if power < desired_power:
            n += 5  # Increase in steps for efficiency
    
    return n


def apply_multiple_comparison_correction(p_values: List[float], method: str = 'bonferroni') -> List[float]:
    """
    Apply multiple comparison correction to p-values.
    
    Args:
        p_values: List of raw p-values
        method: Correction method ('bonferroni', 'holm', 'fdr_bh')
        
    Returns:
        List of corrected p-values
    """
    p_array = np.array(p_values)
    n_comparisons = len(p_values)
    
    if method.lower() == 'bonferroni':
        # Simple Bonferroni correction: multiply p-values by number of comparisons
        corrected_p_values = np.minimum(p_array * n_comparisons, 1.0)
    elif method.lower() == 'holm':
        # Holm-Bonferroni method (step-down)
        # Sort p-values and their original indices
        sorted_indices = np.argsort(p_array)
        sorted_p = p_array[sorted_indices]
        
        # Apply Holm correction
        corrected_sorted = np.zeros_like(sorted_p)
        for i, p_val in enumerate(sorted_p):
            corrected_p = p_val * (n_comparisons - i)
            corrected_sorted[i] = min(corrected_p, 1.0)
        
        # Ensure monotonicity (corrected p-values should be non-decreasing)
        for i in range(1, len(corrected_sorted)):
            corrected_sorted[i] = max(corrected_sorted[i], corrected_sorted[i-1])
        
        # Restore original order
        corrected_p_values = np.empty_like(corrected_sorted)
        corrected_p_values[sorted_indices] = corrected_sorted
    elif method.lower() == 'fdr_bh':
        # Benjamini-Hochberg FDR correction
        sorted_indices = np.argsort(p_array)
        sorted_p = p_array[sorted_indices]
        
        # Apply BH correction
        corrected_sorted = np.zeros_like(sorted_p)
        for i in range(len(sorted_p)):
            corrected_p = sorted_p[i] * n_comparisons / (i + 1)
            corrected_sorted[i] = min(corrected_p, 1.0)
        
        # Ensure monotonicity
        for i in range(len(corrected_sorted) - 2, -1, -1):
            corrected_sorted[i] = min(corrected_sorted[i], corrected_sorted[i+1])
        
        # Restore original order
        corrected_p_values = np.empty_like(corrected_sorted)
        corrected_p_values[sorted_indices] = corrected_sorted
    else:
        raise ValueError(f"Unknown correction method: {method}. Use 'bonferroni', 'holm', or 'fdr_bh'")
    
    return corrected_p_values.tolist()


def perform_statistical_comparison(
    heterogeneous_performance: List[float], 
    homogeneous_performance: List[float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform comprehensive statistical comparison between heterogeneous and homogeneous systems.
    
    Args:
        heterogeneous_performance: Performance metrics for heterogeneous system
        homogeneous_performance: Performance metrics for homogeneous system
        alpha: Significance level for tests (default 0.05)
        
    Returns:
        Dictionary containing comprehensive statistical analysis
    """
    if len(heterogeneous_performance) < 2 or len(homogeneous_performance) < 2:
        raise ValueError("Each group must have at least 2 measurements")
    
    # Calculate effect size
    effect_size_results = calculate_cohens_d(heterogeneous_performance, homogeneous_performance)
    
    # Perform independent t-test
    t_stat, p_value = stats.ttest_ind(heterogeneous_performance, homogeneous_performance)
    
    # Calculate statistical power
    power = calculate_statistical_power(
        effect_size=effect_size_results['hedges_g'],
        n1=len(heterogeneous_performance),
        n2=len(homogeneous_performance),
        alpha=alpha
    )
    
    # Calculate sample size needed for desired power
    required_sample_size = calculate_sample_size_for_power(
        effect_size=effect_size_results['hedges_g'],
        desired_power=0.8,
        alpha=alpha
    )
    
    # Validation checks
    meets_effect_size_requirement = abs(effect_size_results['hedges_g']) >= 0.5
    meets_significance_requirement = p_value < alpha
    meets_power_requirement = power >= 0.8
    
    return {
        't_test': {
            't_statistic': t_stat,
            'p_value': p_value,
            'degrees_of_freedom': len(heterogeneous_performance) + len(homogeneous_performance) - 2
        },
        'effect_size': effect_size_results,
        'power_analysis': {
            'observed_power': power,
            'required_sample_size_for_80_power': required_sample_size,
            'meets_power_requirement': meets_power_requirement
        },
        'validation': {
            'meets_effect_size_requirement': meets_effect_size_requirement,  # d ≥ 0.5
            'meets_significance_requirement': meets_significance_requirement,  # p < α
            'meets_power_requirement': meets_power_requirement,  # power ≥ 0.8
            'interpretation': (
                f"Statistical comparison: t={t_stat:.3f}, p={p_value:.3f}, d={effect_size_results['hedges_g']:.3f}, "
                f"power={power:.3f}"
            )
        }
    }


def validate_academic_compliance(
    heterogeneous_performance: List[float], 
    homogeneous_performance: List[float],
    diversity_metrics: List[float],
    performance_metrics: List[float]
) -> Dict[str, Any]:
    """
    Validate full academic compliance for cognitive heterogeneity experiments.
    
    Args:
        heterogeneous_performance: Performance metrics for heterogeneous system
        homogeneous_performance: Performance metrics for homogeneous system
        diversity_metrics: Diversity measurements across generations
        performance_metrics: Performance measurements across generations
        
    Returns:
        Dictionary containing full academic compliance validation
    """
    # Perform statistical comparison
    comparison_results = perform_statistical_comparison(
        heterogeneous_performance, 
        homogeneous_performance
    )
    
    # Validate cognitive independence
    independence_results = validate_cognitive_independence(diversity_metrics, performance_metrics)
    
    # Overall compliance validation
    meets_all_requirements = (
        comparison_results['validation']['meets_significance_requirement'] and
        comparison_results['validation']['meets_effect_size_requirement'] and
        comparison_results['validation']['meets_power_requirement'] and
        independence_results['validation']['meets_constitutional_requirements']
    )
    
    return {
        'statistical_comparison': comparison_results,
        'cognitive_independence': independence_results,
        'academic_compliance': {
            'meets_all_requirements': meets_all_requirements,
            'summary': f"Academic compliance: {'PASSED' if meets_all_requirements else 'FAILED'}",
            'details': {
                'significance_met': comparison_results['validation']['meets_significance_requirement'],
                'effect_size_met': comparison_results['validation']['meets_effect_size_requirement'],
                'power_met': comparison_results['validation']['meets_power_requirement'],
                'independence_validated': independence_results['validation']['meets_constitutional_requirements']
            }
        }
    }


# Import correlation functions for complete analysis
try:
    from .correlation import validate_cognitive_independence
except ImportError:
    logger.warning("Could not import correlation module. Please ensure it exists.")
    
    # Define a placeholder function if correlation module is not available
    def validate_cognitive_independence(diversity_metrics, performance_metrics):
        """Placeholder function if correlation module is not available."""
        raise ImportError("correlation module not available")


# Exported functions
__all__ = [
    'calculate_cohens_d',
    'calculate_statistical_power',
    'calculate_sample_size_for_power',
    'apply_multiple_comparison_correction',
    'perform_statistical_comparison',
    'validate_academic_compliance',
    'validate_cognitive_independence'  # Imported from correlation module
]