#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析真实实验数据，计算实际统计指标
包括：Cohen's d、相关性、统计功效等
"""

import os
import json
import numpy as np
from scipy import stats
import math

def collect_experiment_results():
    """收集所有实验的性能数据"""
    results = []
    
    # experiments目录
    exp_dir = 'experiments'
    if os.path.exists(exp_dir):
        for f in os.listdir(exp_dir):
            if f.endswith('.json') and 'experiment' in f:
                try:
                    with open(os.path.join(exp_dir, f), 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        if 'results_history' in data and data['results_history']:
                            for r in data['results_history']:
                                results.append({
                                    'file': f,
                                    'generation': r.get('generation', 0),
                                    'diversity': r.get('diversity', 0),
                                    'hetero_perf': r.get('heterogeneous_performance', 0),
                                    'homo_perf': r.get('homogeneous_performance', 0)
                                })
                except Exception as e:
                    pass
    
    # experiments_gemma3目录
    exp_dir = 'experiments_gemma3'
    if os.path.exists(exp_dir):
        for f in os.listdir(exp_dir):
            if f.endswith('.json') and 'experiment' in f:
                try:
                    with open(os.path.join(exp_dir, f), 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        if 'results_history' in data and data['results_history']:
                            for r in data['results_history']:
                                results.append({
                                    'file': f,
                                    'generation': r.get('generation', 0),
                                    'diversity': r.get('diversity', 0),
                                    'hetero_perf': r.get('heterogeneous_performance', 0),
                                    'homo_perf': r.get('homogeneous_performance', 0)
                                })
                except Exception as e:
                    pass
    
    return results

def calculate_cohens_d(group1, group2):
    """计算Cohen's d效应量"""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # 池化标准差
    pooled_std = math.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    
    if pooled_std == 0:
        return 0.0
    
    d = (mean1 - mean2) / pooled_std
    return d

def main():
    print("=" * 60)
    print("真实实验数据分析报告")
    print("=" * 60)
    
    # 收集数据
    results = collect_experiment_results()
    print(f"\n收集到 {len(results)} 条结果记录")
    
    if not results:
        print("没有找到有效数据！")
        return
    
    # 提取性能数据 (只取数值型)
    hetero_scores = []
    homo_scores = []
    diversities = []
    
    for r in results:
        h = r['hetero_perf']
        hm = r['homo_perf']
        d = r['diversity']
        
        if isinstance(h, (int, float)) and h > 0:
            hetero_scores.append(h)
        if isinstance(hm, (int, float)) and hm > 0:
            homo_scores.append(hm)
        if isinstance(d, (int, float)) and d > 0:
            diversities.append(d)
    
    print(f"\n有效数据点:")
    print(f"  异质系统性能: {len(hetero_scores)} 个")
    print(f"  同质系统性能: {len(homo_scores)} 个")
    print(f"  多样性指标: {len(diversities)} 个")
    
    # 基本统计
    print(f"\n基本统计:")
    if hetero_scores:
        print(f"  异质系统性能: mean={np.mean(hetero_scores):.4f}, std={np.std(hetero_scores):.4f}")
    if homo_scores:
        print(f"  同质系统性能: mean={np.mean(homo_scores):.4f}, std={np.std(homo_scores):.4f}")
    if diversities:
        print(f"  多样性: mean={np.mean(diversities):.4f}, std={np.std(diversities):.4f}")
    
    # 计算Cohen's d
    if hetero_scores and homo_scores and len(hetero_scores) > 1 and len(homo_scores) > 1:
        d = calculate_cohens_d(hetero_scores, homo_scores)
        print(f"\n效应量分析:")
        print(f"  Cohen's d = {d:.4f}")
        
        if abs(d) < 0.2:
            interpretation = "可忽略"
        elif abs(d) < 0.5:
            interpretation = "小效应"
        elif abs(d) < 0.8:
            interpretation = "中等效应"
        elif abs(d) < 1.3:
            interpretation = "大效应"
        else:
            interpretation = "极大效应"
        print(f"  解释: {interpretation}")
    
    # t检验
    if hetero_scores and homo_scores and len(hetero_scores) >= 2 and len(homo_scores) >= 2:
        t_stat, p_value = stats.ttest_ind(hetero_scores, homo_scores)
        print(f"\nt检验:")
        print(f"  t统计量 = {t_stat:.4f}")
        print(f"  p值 = {p_value:.6f}")
        print(f"  显著性: {'显著 (p < 0.05)' if p_value < 0.05 else '不显著 (p >= 0.05)'}")
    
    # 相关性分析
    if hetero_scores and diversities and len(hetero_scores) == len(diversities):
        r, p = stats.pearsonr(diversities, hetero_scores)
        print(f"\n相关性分析 (多样性-性能):")
        print(f"  Pearson r = {r:.4f}")
        print(f"  p值 = {p:.6f}")
    
    # 详细数据展示
    print(f"\n前20条记录详情:")
    for r in results[:20]:
        print(f"  {r['file'][:40]}: gen={r['generation']}, hetero={r['hetero_perf']:.3f}, homo={r['homo_perf']:.3f}")
    
    print("\n" + "=" * 60)
    print("分析完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
