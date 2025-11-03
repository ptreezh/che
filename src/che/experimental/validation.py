"""
Cognitive Independence Validation for Cognitive Heterogeneity Experiments

This module provides functions to validate the cognitive independence requirement
that diversity correlates with performance (r ≥ 0.6, p < 0.01) as specified in
the project constitution.

Authors: CHE Research Team
Date: 2025-10-19
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


def validate_cognitive_independence_correlation(
    diversity_metrics: List[float],
    performance_metrics: List[float],
    required_correlation: float = 0.6,
    required_p_value: float = 0.01
) -> Dict[str, Any]:
    """
    Validate the cognitive independence correlation requirement.
    
    This function validates that the correlation between diversity and performance
    meets the constitutional requirement of r ≥ 0.6 with p < 0.01.
    
    Args:
        diversity_metrics: Diversity measurements across generations
        performance_metrics: Performance measurements across generations
        required_correlation: Minimum required correlation coefficient (default: 0.6)
        required_p_value: Maximum allowed p-value (default: 0.01)
        
    Returns:
        Dictionary containing validation results
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
    
    # Calculate Spearman correlation (rank-based, more robust)
    spearman_rho, spearman_p = stats.spearmanr(diversity_array, performance_array)
    
    # Validate constitutional requirements
    meets_correlation = pearson_r >= required_correlation
    meets_significance = pearson_p < required_p_value
    meets_requirements = meets_correlation and meets_significance
    
    # Interpret effect size
    effect_size_interpretation = interpret_correlation_effect_size(abs(pearson_r))
    
    # Calculate confidence interval for correlation
    n = len(diversity_metrics)
    if n > 3:
        # Fisher transformation for confidence interval
        fisher_z = np.arctanh(pearson_r)
        standard_error = 1 / np.sqrt(n - 3)
        z_critical = 1.96  # 95% confidence interval
        
        ci_lower = np.tanh(fisher_z - z_critical * standard_error)
        ci_upper = np.tanh(fisher_z + z_critical * standard_error)
    else:
        ci_lower = ci_upper = 0.0
    
    return {
        'meets_constitutional_requirements': meets_requirements,
        'meets_correlation_requirement': meets_correlation,
        'meets_significance_requirement': meets_significance,
        'correlation_coefficient': pearson_r,
        'p_value': pearson_p,
        'spearman_rho': spearman_rho,
        'spearman_p_value': spearman_p,
        'required_correlation': required_correlation,
        'required_p_value': required_p_value,
        'effect_size_interpretation': effect_size_interpretation,
        'confidence_interval_lower': ci_lower,
        'confidence_interval_upper': ci_upper,
        'sample_size': n,
        'interpretation': (
            f"Cognitive independence {'VALIDATED' if meets_requirements else 'NOT VALIDATED'}: "
            f"r={pearson_r:.3f} ({effect_size_interpretation} correlation), "
            f"p={pearson_p:.3f} ({'statistically significant' if meets_significance else 'not significant'}), "
            f"95% CI [{ci_lower:.3f}, {ci_upper:.3f}]"
        )
    }


def interpret_correlation_effect_size(correlation: float) -> str:
    """
    Interpret the effect size of a correlation coefficient.
    
    Args:
        correlation: Correlation coefficient value
        
    Returns:
        Interpretation of the effect size
    """
    if correlation >= 0.7:
        return "strong"
    elif correlation >= 0.5:
        return "moderate"
    elif correlation >= 0.3:
        return "weak"
    else:
        return "negligible"


def validate_multiple_correlation_requirements(
    correlation_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validate multiple correlation analyses against constitutional requirements.
    
    Args:
        correlation_results: List of correlation analysis results
        
    Returns:
        Dictionary containing aggregated validation results
    """
    if not correlation_results:
        return {
            'overall_validation': False,
            'valid_correlations': 0,
            'total_correlations': 0,
            'interpretation': 'No correlation results to validate'
        }
    
    # Count valid correlations
    valid_correlations = 0
    total_correlations = len(correlation_results)
    
    for result in correlation_results:
        if result.get('meets_constitutional_requirements', False):
            valid_correlations += 1
    
    # Overall validation (require majority to be valid)
    overall_validation = valid_correlations / total_correlations >= 0.5
    
    return {
        'overall_validation': overall_validation,
        'valid_correlations': valid_correlations,
        'total_correlations': total_correlations,
        'validation_rate': valid_correlations / total_correlations,
        'interpretation': (
            f"Overall validation: {'PASSED' if overall_validation else 'FAILED'} "
            f"({valid_correlations}/{total_correlations} correlations meet requirements)"
        )
    }


def calculate_statistical_power(
    correlation: float,
    sample_size: int,
    alpha: float = 0.01
) -> float:
    """
    Calculate statistical power for a correlation analysis.
    
    Args:
        correlation: Correlation coefficient
        sample_size: Sample size (number of data points)
        alpha: Significance level (default: 0.01)
        
    Returns:
        Statistical power (probability of detecting effect if it exists)
    """
    # This is a simplified approximation
    # For accurate power calculation, you would use specialized libraries
    
    # Approximate power calculation using the relationship between sample size,
    # effect size, and significance level
    if sample_size < 3:
        return 0.0
    
    # Simplified power calculation
    # In practice, you would use non-central t-distribution
    effect_size_squared = correlation ** 2
    power_approximation = 1 - np.exp(-0.5 * (sample_size - 3) * effect_size_squared)
    
    return max(0.0, min(1.0, power_approximation))


def validate_cognitive_independence_with_power(
    diversity_metrics: List[float],
    performance_metrics: List[float],
    required_correlation: float = 0.6,
    required_p_value: float = 0.01,
    minimum_power: float = 0.8
) -> Dict[str, Any]:
    """
    Validate cognitive independence with statistical power consideration.
    
    Args:
        diversity_metrics: Diversity measurements across generations
        performance_metrics: Performance measurements across generations
        required_correlation: Minimum required correlation coefficient (default: 0.6)
        required_p_value: Maximum allowed p-value (default: 0.01)
        minimum_power: Minimum required statistical power (default: 0.8)
        
    Returns:
        Dictionary containing validation results with power analysis
    """
    # Perform basic correlation validation
    basic_validation = validate_cognitive_independence_correlation(
        diversity_metrics, performance_metrics, required_correlation, required_p_value
    )
    
    # Calculate statistical power
    correlation = basic_validation['correlation_coefficient']
    sample_size = basic_validation['sample_size']
    statistical_power = calculate_statistical_power(correlation, sample_size, required_p_value)
    
    # Validate power requirement
    meets_power_requirement = statistical_power >= minimum_power
    
    # Overall validation (must meet correlation, significance, AND power requirements)
    overall_validation = (
        basic_validation['meets_constitutional_requirements'] and 
        meets_power_requirement
    )
    
    # Enhanced interpretation
    basic_interpretation = basic_validation['interpretation']
    power_interpretation = (
        f"Statistical power: {statistical_power:.3f} "
        f"({'adequate' if meets_power_requirement else 'inadequate'})"
    )
    
    enhanced_interpretation = f"{basic_interpretation}; {power_interpretation}"
    
    return {
        **basic_validation,
        'statistical_power': statistical_power,
        'meets_power_requirement': meets_power_requirement,
        'overall_validation': overall_validation,
        'interpretation': enhanced_interpretation,
        'power_interpretation': power_interpretation
    }


def generate_validation_report(
    validation_results: Dict[str, Any],
    experiment_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive cognitive independence validation report.
    
    Args:
        validation_results: Dictionary containing validation results
        experiment_context: Optional dictionary with experiment context information
        
    Returns:
        Dictionary containing comprehensive validation report
    """
    report = {
        'validation_results': validation_results,
        'experiment_context': experiment_context or {},
        'report_generated': True,
        'constitutional_compliance': {}
    }
    
    # Extract key validation results
    meets_requirements = validation_results.get('meets_constitutional_requirements', False)
    correlation = validation_results.get('correlation_coefficient', 0.0)
    p_value = validation_results.get('p_value', 1.0)
    effect_size = validation_results.get('effect_size_interpretation', 'unknown')
    sample_size = validation_results.get('sample_size', 0)
    statistical_power = validation_results.get('statistical_power', 0.0)
    meets_power = validation_results.get('meets_power_requirement', False)
    
    # Constitutional compliance assessment
    report['constitutional_compliance'] = {
        'cognitive_independence_validated': meets_requirements,
        'correlation_requirement': f"r ≥ 0.6 (actual: r = {correlation:.3f})",
        'significance_requirement': f"p < 0.01 (actual: p = {p_value:.3f})",
        'power_requirement': f"power ≥ 0.8 (actual: power = {statistical_power:.3f})",
        'effect_size_requirement': f"moderate to strong (actual: {effect_size})",
        'sample_size_requirement': f"sufficient (n = {sample_size})",
        'overall_compliance': meets_requirements and meets_power,
        'interpretation': (
            f"The cognitive independence requirement is "
            f"{'MET' if meets_requirements and meets_power else 'NOT MET'} "
            f"based on the constitutional standards. "
            f"{'Further validation recommended.' if not meets_requirements or not meets_power else 'Validation successful.'}"
        )
    }
    
    return report


# Convenience functions for common validation operations


def quick_independence_check(
    diversity_metrics: List[float],
    performance_metrics: List[float]
) -> bool:
    """
    Quick check if cognitive independence correlation requirement is likely met.
    
    Args:
        diversity_metrics: Diversity measurements
        performance_metrics: Performance measurements
        
    Returns:
        True if correlation is likely ≥ 0.6, False otherwise
    """
    if len(diversity_metrics) < 2 or len(performance_metrics) < 2:
        return False
    
    if len(diversity_metrics) != len(performance_metrics):
        return False
    
    # Calculate correlation
    correlation_result = validate_cognitive_independence_correlation(
        diversity_metrics, performance_metrics
    )
    
    return correlation_result.get('meets_constitutional_requirements', False)


def validate_constitutional_compliance(
    correlation: float,
    p_value: float,
    sample_size: int
) -> Dict[str, Any]:
    """
    Validate compliance with constitutional requirements.
    
    Args:
        correlation: Pearson correlation coefficient
        p_value: P-value for correlation
        sample_size: Number of data points
        
    Returns:
        Dictionary containing compliance validation results
    """
    # Constitutional requirements:
    # 1. Correlation coefficient r ≥ 0.6
    # 2. P-value < 0.01
    # 3. Sample size ≥ 10 (minimum for reliable correlation)
    
    meets_correlation = correlation >= 0.6
    meets_significance = p_value < 0.01
    meets_sample_size = sample_size >= 10
    
    constitutional_compliance = meets_correlation and meets_significance and meets_sample_size
    
    return {
        'constitutional_compliance': constitutional_compliance,
        'meets_correlation_requirement': meets_correlation,
        'meets_significance_requirement': meets_significance,
        'meets_sample_size_requirement': meets_sample_size,
        'correlation_coefficient': correlation,
        'p_value': p_value,
        'sample_size': sample_size,
        'interpretation': (
            f"Constitutional compliance: {'ACHIEVED' if constitutional_compliance else 'NOT ACHIEVED'} - "
            f"Correlation: {'✓' if meets_correlation else '✗'} r≥0.6, "
            f"Significance: {'✓' if meets_significance else '✗'} p<0.01, "
            f"Sample Size: {'✓' if meets_sample_size else '✗'} n≥10"
        )
    }