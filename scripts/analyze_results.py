#!/usr/bin/env python3
"""
实验结果分析脚本

Authors: Zhang Shuren, AI Personality LAB
Date: 2025-09-20
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_experiment_results(results_dir: str = "results") -> pd.DataFrame:
    """加载实验结果"""

    results_path = Path(results_dir)
    data = []

    # 查找所有实验结果文件
    for file in results_path.glob("experiment_*.json"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                result = json.load(f)

            # 提取关键信息
            row = {
                'diversity_level': result['condition']['diversity_level'],
                'evolution_pressure': result['condition']['evolution_pressure'],
                'role_configuration': result['condition']['role_configuration'],
                'condition_description': result['condition']['description'],
                'replication_id': result['replication_id'],
                'population_size': result['population_size'],
                'generations': result['generations'],
                'final_performance': result['final_performance'],
                'execution_time': result['execution_time'],
                'initial_performance': result['performance_trajectory'][0],
                'performance_improvement': result['final_performance'] - result['performance_trajectory'][0],
                'final_diversity': result['diversity_trajectory'][-1],
                'initial_diversity': result['diversity_trajectory'][0],
                'diversity_change': result['diversity_trajectory'][-1] - result['diversity_trajectory'][0]
            }
            data.append(row)

        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

    return pd.DataFrame(data)

def analyze_factorial_results(df: pd.DataFrame):
    """分析因子实验结果"""

    print("🧬 认知异质性实验结果分析")
    print("=" * 60)

    # 基本统计
    print(f"\n📊 基本统计:")
    print(f"总实验数: {len(df)}")
    print(f"成功率: {len(df) / (len(df) + df.isnull().sum().sum()) * 100:.1f}%")
    print(f"平均最终性能: {df['final_performance'].mean():.3f} ± {df['final_performance'].std():.3f}")
    print(f"性能提升范围: {df['performance_improvement'].min():.3f} 到 {df['performance_improvement'].max():.3f}")

    # 因子分析
    print(f"\n🔍 因子分析:")

    # 1. 多样性水平影响
    print(f"\n1. 多样性水平影响:")
    diversity_stats = df.groupby('diversity_level')['final_performance'].agg(['mean', 'std', 'count'])
    print(diversity_stats)

    # 2. 进化压力影响
    print(f"\n2. 进化压力影响:")
    evolution_stats = df.groupby('evolution_pressure')['final_performance'].agg(['mean', 'std', 'count'])
    print(evolution_stats)

    # 3. 角色配置影响
    print(f"\n3. 角色配置影响:")
    role_stats = df.groupby('role_configuration')['final_performance'].agg(['mean', 'std', 'count'])
    print(role_stats)

    # 4. 交互作用分析
    print(f"\n4. 关键交互作用:")

    # 多样性 × 进化压力
    print(f"\n   多样性 × 进化压力:")
    interaction1 = df.groupby(['diversity_level', 'evolution_pressure'])['final_performance'].mean().unstack()
    print(interaction1)

    # 多样性 × 角色配置
    print(f"\n   多样性 × 角色配置:")
    interaction2 = df.groupby(['diversity_level', 'role_configuration'])['final_performance'].mean().unstack()
    print(interaction2)

    # 进化压力 × 角色配置
    print(f"\n   进化压力 × 角色配置:")
    interaction3 = df.groupby(['evolution_pressure', 'role_configuration'])['final_performance'].mean().unstack()
    print(interaction3)

    # 性能排名
    print(f"\n🏆 最佳实验条件:")
    top_performers = df.nlargest(5, 'final_performance')[['condition_description', 'final_performance', 'performance_improvement']]
    print(top_performers.to_string(index=False))

    # 多样性分析
    print(f"\n📈 多样性分析:")
    print(f"平均初始多样性: {df['initial_diversity'].mean():.3f}")
    print(f"平均最终多样性: {df['final_diversity'].mean():.3f}")
    print(f"多样性变化: {df['diversity_change'].mean():.3f}")

    # 相关性分析
    print(f"\n🔗 相关性分析:")
    correlation_matrix = df[['final_performance', 'population_size', 'initial_diversity', 'final_diversity']].corr()
    print("最终性能与其他因素的相关性:")
    print(correlation_matrix['final_performance'].sort_values(ascending=False))

    return df

def generate_summary_report(df: pd.DataFrame):
    """生成汇总报告"""

    report = f"""
# 🧬 认知异质性实验结果报告

## 📊 实验概况
- **实验设计**: 2×2×3因子设计
- **总实验数**: {len(df)}
- **成功率**: {len(df) / (len(df) + df.isnull().sum().sum()) * 100:.1f}%
- **平均性能**: {df['final_performance'].mean():.3f} ± {df['final_performance'].std():.3f}

## 🎯 主要发现

### 1. 多样性效应
{format_factor_effect(df.groupby('diversity_level')['final_performance'].mean())}

### 2. 进化压力效应
{format_factor_effect(df.groupby('evolution_pressure')['final_performance'].mean())}

### 3. 角色配置效应
{format_factor_effect(df.groupby('role_configuration')['final_performance'].mean())}

### 4. 最佳配置
最佳实验条件: {df.loc[df['final_performance'].idxmax(), 'condition_description']}
最终性能: {df['final_performance'].max():.3f}

## 📈 关键指标
- **平均性能提升**: {df['performance_improvement'].mean():.3f}
- **最大性能提升**: {df['performance_improvement'].max():.3f}
- **平均执行时间**: {df['execution_time'].mean():.4f}秒

## 🔬 统计显著性
需要进一步的统计检验来确定效应的显著性。
"""

    return report

def format_factor_effect(grouped_data):
    """格式化因子效应"""
    lines = []
    for name, value in grouped_data.items():
        lines.append(f"- **{name.title()}**: {value:.3f}")
    return '\n'.join(lines)

def save_analysis_report(df: pd.DataFrame, filename: str = "experiment_analysis_report.md"):
    """保存分析报告"""

    report = generate_summary_report(df)

    # 保存报告
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report)

    # 保存数据为CSV
    csv_filename = filename.replace('.md', '_data.csv')
    df.to_csv(csv_filename, index=False)

    print(f"\n📄 分析报告已保存:")
    print(f"  - 报告: {filename}")
    print(f"  - 数据: {csv_filename}")

def main():
    """主函数"""

    print("🔬 认知异质性实验结果分析")
    print("=" * 50)

    # 加载数据
    df = load_experiment_results()

    if df.empty:
        print("❌ 没有找到实验结果文件")
        print("请先运行实验: python scripts/quick_experiment.py --type factorial")
        return

    print(f"✅ 成功加载 {len(df)} 个实验结果")

    # 分析结果
    df_analyzed = analyze_factorial_results(df)

    # 保存报告
    save_analysis_report(df_analyzed)

    print(f"\n🎉 分析完成! 查看生成的报告文件了解详细结果。")

if __name__ == "__main__":
    main()