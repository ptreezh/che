"""
Performance Tracker for Cognitive Heterogeneity Validation

This module provides functions to track and analyze the performance of agents
and ecosystems in cognitive heterogeneity validation experiments.

Authors: CHE Research Team
Date: 2025-10-19
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """
    Data class for storing performance metrics.
    """
    generation: int
    timestamp: datetime
    average_score: float
    median_score: float
    std_deviation: float
    min_score: float
    max_score: float
    agent_count: int
    score_distribution: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert performance metrics to dictionary.
        
        Returns:
            Dictionary representation of performance metrics
        """
        return {
            'generation': self.generation,
            'timestamp': self.timestamp.isoformat(),
            'average_score': self.average_score,
            'median_score': self.median_score,
            'std_deviation': self.std_deviation,
            'min_score': self.min_score,
            'max_score': self.max_score,
            'agent_count': self.agent_count,
            'score_distribution': self.score_distribution,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """
        Create performance metrics from dictionary.
        
        Args:
            data: Dictionary representation of performance metrics
            
        Returns:
            New performance metrics instance
        """
        return cls(
            generation=data['generation'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            average_score=data['average_score'],
            median_score=data['median_score'],
            std_deviation=data['std_deviation'],
            min_score=data['min_score'],
            max_score=data['max_score'],
            agent_count=data['agent_count'],
            score_distribution=data.get('score_distribution', {}),
            metadata=data.get('metadata', {})
        )


class PerformanceTracker:
    """
    Tracker for monitoring and analyzing agent performance in experiments.
    """
    
    def __init__(self):
        """Initialize the performance tracker."""
        self.performance_history: List[PerformanceMetrics] = []
        self.current_generation: int = 0
    
    def record_generation_performance(
        self,
        agent_scores: Dict[str, float],
        generation: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PerformanceMetrics:
        """
        Record performance metrics for a generation.
        
        Args:
            agent_scores: Dictionary mapping agent_id to score
            generation: Generation number
            metadata: Optional additional metadata
            
        Returns:
            Performance metrics for the generation
        """
        if not agent_scores:
            raise ValueError("Agent scores cannot be empty")
        
        scores = list(agent_scores.values())
        agent_count = len(scores)
        
        # Calculate basic statistics
        average_score = np.mean(scores)
        median_score = np.median(scores)
        std_deviation = np.std(scores) if len(scores) > 1 else 0.0
        min_score = min(scores)
        max_score = max(scores)
        
        # Calculate score distribution (categorized by score tiers)
        score_distribution = self._categorize_scores(scores)
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            generation=generation,
            timestamp=datetime.now(),
            average_score=average_score,
            median_score=median_score,
            std_deviation=std_deviation,
            min_score=min_score,
            max_score=max_score,
            agent_count=agent_count,
            score_distribution=score_distribution,
            metadata=metadata or {}
        )
        
        # Store metrics
        self.performance_history.append(metrics)
        self.current_generation = generation
        
        logger.info(f"Recorded performance for generation {generation}: "
                   f"Avg={average_score:.3f}, Std={std_deviation:.3f}, "
                   f"Min={min_score:.1f}, Max={max_score:.1f}")
        
        return metrics
    
    def _categorize_scores(self, scores: List[float]) -> Dict[str, int]:
        """
        Categorize scores into performance tiers.
        
        Args:
            scores: List of agent scores
            
        Returns:
            Dictionary mapping score categories to counts
        """
        categories = {
            'refutation': 0,    # 2.0 scores (explicit refutation)
            'doubt': 0,        # 1.0 scores (expression of doubt)
            'acceptance': 0    # 0.0 scores (blind acceptance)
        }
        
        for score in scores:
            if score >= 1.9:  # Account for floating point precision
                categories['refutation'] += 1
            elif score >= 0.9:
                categories['doubt'] += 1
            else:
                categories['acceptance'] += 1
        
        return categories
    
    def calculate_performance_improvement(
        self,
        start_generation: int,
        end_generation: int
    ) -> Dict[str, Any]:
        """
        Calculate performance improvement between two generations.
        
        Args:
            start_generation: Starting generation number
            end_generation: Ending generation number
            
        Returns:
            Dictionary containing performance improvement metrics
        """
        start_metrics = self.get_generation_metrics(start_generation)
        end_metrics = self.get_generation_metrics(end_generation)
        
        if not start_metrics or not end_metrics:
            raise ValueError("Could not find metrics for specified generations")
        
        # Calculate improvements
        avg_improvement = end_metrics.average_score - start_metrics.average_score
        median_improvement = end_metrics.median_score - start_metrics.median_score
        
        # Calculate percentage improvements
        avg_pct_improvement = (avg_improvement / start_metrics.average_score * 100 
                              if start_metrics.average_score > 0 else 0)
        median_pct_improvement = (median_improvement / start_metrics.median_score * 100 
                                 if start_metrics.median_score > 0 else 0)
        
        return {
            'start_generation': start_generation,
            'end_generation': end_generation,
            'average_improvement': avg_improvement,
            'median_improvement': median_improvement,
            'average_pct_improvement': avg_pct_improvement,
            'median_pct_improvement': median_pct_improvement,
            'start_average': start_metrics.average_score,
            'end_average': end_metrics.average_score,
            'start_median': start_metrics.median_score,
            'end_median': end_metrics.median_score
        }
    
    def get_generation_metrics(self, generation: int) -> Optional[PerformanceMetrics]:
        """
        Get performance metrics for a specific generation.
        
        Args:
            generation: Generation number
            
        Returns:
            Performance metrics for the generation, or None if not found
        """
        for metrics in self.performance_history:
            if metrics.generation == generation:
                return metrics
        return None
    
    def get_performance_trend(self) -> Dict[str, Any]:
        """
        Calculate the overall performance trend across all generations.
        
        Returns:
            Dictionary containing trend analysis
        """
        if len(self.performance_history) < 2:
            return {
                'trend_slope': 0.0,
                'trend_direction': 'insufficient_data',
                'correlation': 0.0,
                'interpretation': 'Need at least 2 generations for trend analysis'
            }
        
        # Extract average scores and generation numbers
        generations = [metrics.generation for metrics in self.performance_history]
        avg_scores = [metrics.average_score for metrics in self.performance_history]
        
        # Calculate linear trend
        slope, intercept = np.polyfit(generations, avg_scores, 1)
        
        # Calculate correlation
        correlation = np.corrcoef(generations, avg_scores)[0, 1]
        
        # Determine trend direction
        if slope > 0.01:
            trend_direction = 'improving'
        elif slope < -0.01:
            trend_direction = 'declining'
        else:
            trend_direction = 'stable'
        
        return {
            'trend_slope': slope,
            'trend_direction': trend_direction,
            'correlation': correlation,
            'interpretation': f'Performance is {trend_direction} over generations'
        }
    
    def calculate_cohens_d(
        self,
        group1_scores: List[float],
        group2_scores: List[float]
    ) -> float:
        """
        Calculate Cohen's d effect size between two groups of scores.
        
        Args:
            group1_scores: Scores from first group
            group2_scores: Scores from second group
            
        Returns:
            Cohen's d effect size
        """
        if not group1_scores or not group2_scores:
            return 0.0
        
        # Calculate means
        mean1 = np.mean(group1_scores)
        mean2 = np.mean(group2_scores)
        
        # Calculate pooled standard deviation
        n1, n2 = len(group1_scores), len(group2_scores)
        var1, var2 = np.var(group1_scores, ddof=1), np.var(group2_scores, ddof=1)
        
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        # Calculate Cohen's d
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
        
        return cohens_d
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all performance metrics.
        
        Returns:
            Dictionary containing performance summary statistics
        """
        if not self.performance_history:
            return {
                'total_generations': 0,
                'interpretation': 'No performance data recorded'
            }
        
        # Extract all average scores
        avg_scores = [metrics.average_score for metrics in self.performance_history]
        
        return {
            'total_generations': len(self.performance_history),
            'overall_average': np.mean(avg_scores),
            'overall_std': np.std(avg_scores),
            'best_generation': max(self.performance_history, key=lambda m: m.average_score).generation,
            'best_score': max(avg_scores),
            'worst_generation': min(self.performance_history, key=lambda m: m.average_score).generation,
            'worst_score': min(avg_scores),
            'latest_generation': self.performance_history[-1].generation,
            'latest_score': self.performance_history[-1].average_score,
            'interpretation': f'Tracked {len(self.performance_history)} generations of performance data'
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert performance tracker to dictionary.
        
        Returns:
            Dictionary representation of performance tracker
        """
        return {
            'performance_history': [metrics.to_dict() for metrics in self.performance_history],
            'current_generation': self.current_generation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceTracker':
        """
        Create performance tracker from dictionary.
        
        Args:
            data: Dictionary representation of performance tracker
            
        Returns:
            New performance tracker instance
        """
        tracker = cls()
        tracker.performance_history = [
            PerformanceMetrics.from_dict(metrics_data) 
            for metrics_data in data.get('performance_history', [])
        ]
        tracker.current_generation = data.get('current_generation', 0)
        return tracker


# Convenience functions for common performance tracking operations


def track_experiment_performance(
    tracker: PerformanceTracker,
    agent_scores: Dict[str, float],
    generation: int,
    experiment_type: str,
    metadata: Optional[Dict[str, Any]] = None
) -> PerformanceMetrics:
    """
    Track performance for an experiment generation.
    
    Args:
        tracker: Performance tracker instance
        agent_scores: Dictionary mapping agent_id to score
        generation: Generation number
        experiment_type: Type of experiment (e.g., 'heterogeneous', 'homogeneous')
        metadata: Optional additional metadata
        
    Returns:
        Performance metrics for the generation
    """
    # Add experiment type to metadata
    combined_metadata = metadata or {}
    combined_metadata['experiment_type'] = experiment_type
    
    # Record performance
    metrics = tracker.record_generation_performance(
        agent_scores=agent_scores,
        generation=generation,
        metadata=combined_metadata
    )
    
    return metrics


def compare_system_performance(
    heterogeneous_tracker: PerformanceTracker,
    homogeneous_tracker: PerformanceTracker,
    generations: List[int]
) -> Dict[str, Any]:
    """
    Compare performance between heterogeneous and homogeneous systems.
    
    Args:
        heterogeneous_tracker: Tracker for heterogeneous system
        homogeneous_tracker: Tracker for homogeneous system
        generations: List of generation numbers to compare
        
    Returns:
        Dictionary containing comparison results
    """
    comparison_results = {}
    
    for gen in generations:
        het_metrics = heterogeneous_tracker.get_generation_metrics(gen)
        hom_metrics = homogeneous_tracker.get_generation_metrics(gen)
        
        if het_metrics and hom_metrics:
            difference = het_metrics.average_score - hom_metrics.average_score
            percent_better = (difference / hom_metrics.average_score * 100 
                           if hom_metrics.average_score > 0 else 0)
            
            comparison_results[gen] = {
                'heterogeneous_avg': het_metrics.average_score,
                'homogeneous_avg': hom_metrics.average_score,
                'absolute_difference': difference,
                'percent_better': percent_better,
                'statistically_significant': abs(difference) > 0.1  # Simple threshold
            }
    
    return comparison_results