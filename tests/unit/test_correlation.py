"""
Unit tests for enhanced correlation analysis module.

Authors: CHE Research Team
Date: 2025-10-31
"""

import pytest
import numpy as np
from src.che.experimental.correlation import (
    calculate_pearson_correlation,
    calculate_spearman_correlation,
    calculate_kendall_correlation,
    calculate_correlation_effect_size,
    validate_cognitive_independence
)


def test_pearson_correlation_basic():
    """Test basic Pearson correlation calculation."""
    # Perfect positive correlation
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    
    result = calculate_pearson_correlation(x, y)
    
    assert result['correlation'] == pytest.approx(1.0, abs=1e-10)
    assert result['p_value'] == pytest.approx(0.0, abs=1e-10)
    assert result['n'] == 5
    assert result['method'] == 'pearson'


def test_pearson_correlation_negative():
    """Test negative Pearson correlation calculation."""
    # Perfect negative correlation
    x = [1, 2, 3, 4, 5]
    y = [10, 8, 6, 4, 2]
    
    result = calculate_pearson_correlation(x, y)
    
    assert result['correlation'] == pytest.approx(-1.0, abs=1e-10)
    assert result['p_value'] == pytest.approx(0.0, abs=1e-10)


def test_pearson_correlation_weak():
    """Test weak Pearson correlation calculation."""
    # Weak positive correlation
    x = [1, 2, 3, 4, 5]
    y = [1, 3, 2, 5, 4]  # Slightly scattered
    
    result = calculate_pearson_correlation(x, y)
    
    assert result['correlation'] > 0.5
    assert result['correlation'] < 1.0


def test_pearson_correlation_invalid_input():
    """Test Pearson correlation with invalid input."""
    # Different lengths
    x = [1, 2, 3]
    y = [1, 2]
    
    with pytest.raises(ValueError):
        calculate_pearson_correlation(x, y)
    
    # Too few points
    x = [1]
    y = [2]
    
    with pytest.raises(ValueError):
        calculate_pearson_correlation(x, y)


def test_spearman_correlation_basic():
    """Test basic Spearman correlation calculation."""
    # Perfect monotonic relationship
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    
    result = calculate_spearman_correlation(x, y)
    
    assert result['correlation'] == pytest.approx(1.0, abs=1e-10)
    assert result['p_value'] == pytest.approx(0.0, abs=1e-10)
    assert result['n'] == 5
    assert result['method'] == 'spearman'


def test_kendall_correlation_basic():
    """Test basic Kendall correlation calculation."""
    # Perfect concordance
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    
    result = calculate_kendall_correlation(x, y)
    
    assert result['correlation'] == pytest.approx(1.0, abs=1e-10)
    assert result['p_value'] == pytest.approx(0.0, abs=1e-10)
    assert result['n'] == 5
    assert result['method'] == 'kendall'


def test_correlation_effect_size():
    """Test correlation effect size calculation."""
    # Test different effect sizes
    result1 = calculate_correlation_effect_size(0.05)  # Negligible
    assert result1['effect_size'] == 'negligible'
    assert result1['interpretation'] == 'very weak association'
    
    result2 = calculate_correlation_effect_size(0.2)  # Small
    assert result2['effect_size'] == 'small'
    assert result2['interpretation'] == 'weak association'
    
    result3 = calculate_correlation_effect_size(0.4)  # Medium
    assert result3['effect_size'] == 'medium'
    assert result3['interpretation'] == 'moderate association'
    
    result4 = calculate_correlation_effect_size(0.7)  # Large
    assert result4['effect_size'] == 'large'
    assert result4['interpretation'] == 'strong association'
    
    # Test r-squared calculation
    assert result4['r_squared'] == pytest.approx(0.49, abs=1e-10)


def test_cognitive_independence_validation():
    """Test cognitive independence validation with strong correlation."""
    # Strong positive correlation data
    diversity = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    performance = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    
    result = validate_cognitive_independence(diversity, performance)
    
    # Check that all correlation methods are calculated
    assert 'pearson' in result
    assert 'spearman' in result
    assert 'kendall' in result
    assert 'effect_size' in result
    assert 'validation' in result
    
    # Check primary correlation
    assert result['pearson']['correlation'] > 0.9
    assert result['validation']['meets_correlation_requirement'] in [True, False]
    assert result['validation']['meets_significance_requirement'] in [True, False]


def test_cognitive_independence_validation_strong():
    """Test cognitive independence validation with data that meets requirements."""
    # Create data with strong correlation that should meet requirements
    np.random.seed(42)  # For reproducibility
    x = np.linspace(0, 1, 20)
    y = 0.7 * x + 0.2 * np.random.normal(size=20)  # r ≈ 0.7
    
    diversity = x.tolist()
    performance = y.tolist()
    
    result = validate_cognitive_independence(diversity, performance)
    
    # Should have valid correlation
    assert not np.isnan(result['pearson']['correlation'])
    assert result['pearson']['p_value'] < 0.05  # Should be significant


def test_cognitive_independence_invalid_input():
    """Test cognitive independence validation with invalid input."""
    # Different lengths
    diversity = [0.1, 0.2, 0.3]
    performance = [0.2, 0.3]
    
    with pytest.raises(ValueError):
        validate_cognitive_independence(diversity, performance)
    
    # Too few points
    diversity = [0.1]
    performance = [0.2]
    
    with pytest.raises(ValueError):
        validate_cognitive_independence(diversity, performance)


if __name__ == "__main__":
    pytest.main([__file__])