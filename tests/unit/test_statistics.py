"""
Unit tests for enhanced statistical validation module.

Authors: CHE Research Team
Date: 2025-10-31
"""

import pytest
import numpy as np
from src.che.experimental.statistics import (
    calculate_cohens_d,
    calculate_statistical_power,
    calculate_sample_size_for_power,
    apply_multiple_comparison_correction,
    perform_statistical_comparison,
    validate_academic_compliance
)


def test_cohens_d_basic():
    """Test basic Cohen's d calculation."""
    # Two groups with clear difference
    group1 = [1, 2, 3, 4, 5]
    group2 = [3, 4, 5, 6, 7]
    
    result = calculate_cohens_d(group1, group2)
    
    # Should have reasonable values
    assert isinstance(result['cohens_d'], float)
    assert isinstance(result['hedges_g'], float)
    assert result['n1'] == 5
    assert result['n2'] == 5


def test_cohens_d_identical_groups():
    """Test Cohen's d with identical groups."""
    group1 = [1, 2, 3, 4, 5]
    group2 = [1, 2, 3, 4, 5]
    
    result = calculate_cohens_d(group1, group2)
    
    # Should have zero effect size
    assert result['cohens_d'] == pytest.approx(0.0, abs=1e-10)
    assert result['hedges_g'] == pytest.approx(0.0, abs=1e-10)


def test_cohens_d_invalid_input():
    """Test Cohen's d with invalid input."""
    # Too few elements
    group1 = [1]
    group2 = [2]
    
    with pytest.raises(ValueError):
        calculate_cohens_d(group1, group2)


def test_statistical_power():
    """Test statistical power calculation."""
    # Test with medium effect size
    power = calculate_statistical_power(effect_size=0.5, n1=30, n2=30)
    
    # Should return a reasonable power value
    assert 0 <= power <= 1
    assert isinstance(power, float)


def test_sample_size_for_power():
    """Test sample size calculation for desired power."""
    # Test with medium effect size
    sample_size = calculate_sample_size_for_power(effect_size=0.5, desired_power=0.8)
    
    # Should return a reasonable sample size
    assert sample_size > 0
    assert isinstance(sample_size, int)


def test_multiple_comparison_correction():
    """Test multiple comparison correction methods."""
    # Test p-values
    p_values = [0.01, 0.03, 0.05, 0.10, 0.20]
    
    # Test Bonferroni correction
    bonferroni_corrected = apply_multiple_comparison_correction(p_values, 'bonferroni')
    assert len(bonferroni_corrected) == len(p_values)
    assert all(0 <= p <= 1 for p in bonferroni_corrected)
    
    # Test Holm correction
    holm_corrected = apply_multiple_comparison_correction(p_values, 'holm')
    assert len(holm_corrected) == len(p_values)
    assert all(0 <= p <= 1 for p in holm_corrected)
    
    # Test FDR correction
    fdr_corrected = apply_multiple_comparison_correction(p_values, 'fdr_bh')
    assert len(fdr_corrected) == len(p_values)
    assert all(0 <= p <= 1 for p in fdr_corrected)
    
    # Test invalid method
    with pytest.raises(ValueError):
        apply_multiple_comparison_correction(p_values, 'invalid_method')


def test_statistical_comparison():
    """Test comprehensive statistical comparison."""
    # Create two groups with clear difference
    np.random.seed(42)
    heterogeneous = np.random.normal(1.0, 0.2, 30).tolist()  # Mean = 1.0
    homogeneous = np.random.normal(0.5, 0.2, 30).tolist()    # Mean = 0.5
    
    result = perform_statistical_comparison(heterogeneous, homogeneous)
    
    # Check that all expected keys are present
    assert 't_test' in result
    assert 'effect_size' in result
    assert 'power_analysis' in result
    assert 'validation' in result
    
    # Check t-test results
    assert 't_statistic' in result['t_test']
    assert 'p_value' in result['t_test']
    assert 'degrees_of_freedom' in result['t_test']
    
    # Check effect size results
    assert 'cohens_d' in result['effect_size']
    assert 'hedges_g' in result['effect_size']
    assert 'ci_lower' in result['effect_size']
    assert 'ci_upper' in result['effect_size']
    
    # Check power analysis results
    assert 'observed_power' in result['power_analysis']
    assert 'required_sample_size_for_80_power' in result['power_analysis']
    assert 'meets_power_requirement' in result['power_analysis']
    
    # Check validation results
    assert 'meets_effect_size_requirement' in result['validation']
    assert 'meets_significance_requirement' in result['validation']
    assert 'meets_power_requirement' in result['validation']


def test_statistical_comparison_invalid_input():
    """Test statistical comparison with invalid input."""
    # Too few elements
    group1 = [1]
    group2 = [2]
    
    with pytest.raises(ValueError):
        perform_statistical_comparison(group1, group2)


def test_academic_compliance():
    """Test academic compliance validation."""
    # Create test data
    np.random.seed(42)
    heterogeneous = np.random.normal(1.0, 0.2, 30).tolist()  # Mean = 1.0
    homogeneous = np.random.normal(0.5, 0.2, 30).tolist()    # Mean = 0.5
    diversity = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    performance = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    
    result = validate_academic_compliance(heterogeneous, homogeneous, diversity, performance)
    
    # Check that all expected keys are present
    assert 'statistical_comparison' in result
    assert 'cognitive_independence' in result
    assert 'academic_compliance' in result
    
    # Check academic compliance results
    assert 'meets_all_requirements' in result['academic_compliance']
    assert 'summary' in result['academic_compliance']
    assert 'details' in result['academic_compliance']


if __name__ == "__main__":
    pytest.main([__file__])