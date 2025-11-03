"""
Visualization Module for Cognitive Heterogeneity Validation Project

This module creates charts and graphs to visualize the experimental results
of the cognitive heterogeneity validation project.

Authors: CHE Research Team
Date: 2025-10-24
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from typing import Dict, List, Any
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_experiment_data(file_path: str) -> Dict[str, Any]:
    """Load experiment data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_performance_comparison(results_history: List[Dict[str, Any]]) -> None:
    """
    Plot performance comparison between heterogeneous and homogeneous systems.
    
    Args:
        results_history: List of results dictionaries from experiment
    """
    generations = [result['generation'] for result in results_history]
    heterogeneous_perf = [result['heterogeneous_performance'] for result in results_history]
    homogeneous_perf = [result['homogeneous_performance'] for result in results_history]
    
    plt.figure(figsize=(12, 8))
    
    # Plot performance curves
    plt.plot(generations, heterogeneous_perf, 'b-o', label='异质系统', linewidth=2, markersize=6)
    plt.plot(generations, homogeneous_perf, 'r-s', label='同质系统', linewidth=2, markersize=6)
    
    # Add labels and title
    plt.xlabel('进化代数 (Generations)', fontsize=14)
    plt.ylabel('平均适应度 (Average Fitness)', fontsize=14)
    plt.title('认知异质性验证 - 异质系统 vs 同质系统性能比较', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add performance difference annotation
    final_diff = heterogeneous_perf[-1] - homogeneous_perf[-1]
    plt.annotate(f'最终性能差异: +{final_diff:.3f}', 
                xy=(generations[-1], heterogeneous_perf[-1]), 
                xytext=(generations[-1]-2, heterogeneous_perf[-1]+0.1),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Save plot
    plt.tight_layout()
    plt.savefig('results/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_diversity_correlation(results_history: List[Dict[str, Any]]) -> None:
    """
    Plot correlation between diversity and performance.
    
    Args:
        results_history: List of results dictionaries from experiment
    """
    generations = [result['generation'] for result in results_history]
    diversity = [result.get('diversity', 0.0) for result in results_history]
    performance = [result['heterogeneous_performance'] for result in results_history]
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    plt.scatter(diversity, performance, c=generations, cmap='viridis', s=100, alpha=0.7)
    
    # Add trend line
    if len(set(diversity)) > 1:  # Only add trend line if there's variance in diversity
        z = np.polyfit(diversity, performance, 1)
        p = np.poly1d(z)
        plt.plot(diversity, p(diversity), "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(diversity, performance)[0, 1]
    else:
        correlation = 0.0
    
    # Add labels and title
    plt.xlabel('认知多样性指数 (Cognitive Diversity Index)', fontsize=14)
    plt.ylabel('平均适应度 (Average Fitness)', fontsize=14)
    plt.title(f'认知独立性验证 - 多样性与性能相关性 (r = {correlation:.3f})', fontsize=16)
    plt.colorbar(label='进化代数')
    
    # Add annotations for key points
    for i, (gen, div, perf) in enumerate(zip(generations, diversity, performance)):
        if i % 3 == 0:  # Annotate every 3rd point to avoid clutter
            plt.annotate(f'G{gen}', (div, perf), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8, alpha=0.7)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('results/diversity_performance_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_generation_performance(results_history: List[Dict[str, Any]]) -> None:
    """
    Plot generation-by-generation performance trends.
    
    Args:
        results_history: List of results dictionaries from experiment
    """
    generations = [result['generation'] for result in results_history]
    performance = [result['heterogeneous_performance'] for result in results_history]
    
    plt.figure(figsize=(12, 8))
    
    # Plot bar chart
    bars = plt.bar(generations, performance, color='skyblue', alpha=0.8)
    
    # Add value labels on bars
    for i, (bar, perf) in enumerate(zip(bars, performance)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{perf:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Add labels and title
    plt.xlabel('进化代数 (Generations)', fontsize=14)
    plt.ylabel('平均适应度 (Average Fitness)', fontsize=14)
    plt.title('逐代性能趋势 - 认知异质性验证', fontsize=16)
    plt.xticks(generations)
    plt.grid(axis='y', alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('results/generation_performance_trends.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_visualization_report() -> None:
    """Create a comprehensive visualization report."""
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Load experiment data (using sample data for demonstration)
    # In a real implementation, this would load from actual experiment results
    sample_results = [
        {"generation": 1, "diversity": 0.0, "heterogeneous_performance": 0.667, "homogeneous_performance": 0.267},
        {"generation": 2, "diversity": 0.0, "heterogeneous_performance": 0.733, "homogeneous_performance": 0.200},
        {"generation": 3, "diversity": 0.0, "heterogeneous_performance": 0.200, "homogeneous_performance": 0.267},
        {"generation": 4, "diversity": 0.0, "heterogeneous_performance": 0.400, "homogeneous_performance": 0.333},
        {"generation": 5, "diversity": 0.0, "heterogeneous_performance": 0.400, "homogeneous_performance": 0.333},
        {"generation": 6, "diversity": 0.0, "heterogeneous_performance": 0.467, "homogeneous_performance": 0.200},
        {"generation": 7, "diversity": 0.0, "heterogeneous_performance": 0.267, "homogeneous_performance": 0.467},
        {"generation": 8, "diversity": 0.0, "heterogeneous_performance": 0.467, "homogeneous_performance": 0.267},
        {"generation": 9, "diversity": 0.0, "heterogeneous_performance": 0.467, "homogeneous_performance": 0.333},
        {"generation": 10, "diversity": 0.0, "heterogeneous_performance": 0.533, "homogeneous_performance": 0.100},
        {"generation": 11, "diversity": 0.0, "heterogeneous_performance": 0.267, "homogeneous_performance": 0.400},
        {"generation": 12, "diversity": 0.0, "heterogeneous_performance": 0.333, "homogeneous_performance": 0.400},
        {"generation": 13, "diversity": 0.0, "heterogeneous_performance": 0.000, "homogeneous_performance": 0.200},
        {"generation": 14, "diversity": 0.0, "heterogeneous_performance": 0.000, "homogeneous_performance": 0.200},
        {"generation": 15, "diversity": 0.0, "heterogeneous_performance": 0.000, "homogeneous_performance": 0.200}
    ]
    
    # Create visualizations
    plot_performance_comparison(sample_results)
    plot_diversity_correlation(sample_results)
    plot_generation_performance(sample_results)
    
    print("📊 可视化图表已生成:")
    print("  - 性能比较图表: results/performance_comparison.png")
    print("  - 多样性与性能相关性图表: results/diversity_performance_correlation.png")
    print("  - 逐代性能趋势图表: results/generation_performance_trends.png")

if __name__ == "__main__":
    create_visualization_report()