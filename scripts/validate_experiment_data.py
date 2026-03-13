"""
实验数据验证与统计分析脚本
遵循 soul.md 第一原则：诚实、严谨、科学理性

Author: CHE Research Team (guided by soul.md)
Date: 2025-03-13
"""

import json
import os
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Tuple
from datetime import datetime

def calculate_cognitive_diversity_index(agents: List[Dict]) -> float:
    """
    计算认知异质性指数 (Shannon熵)
    
    H = -∑ (n_i/N) × log(n_i/N)
    
    Returns:
        H ∈ [0, log(k)]，k为认知类型数量
    """
    type_counts = {'critical': 0, 'awakened': 0, 'standard': 0}
    
    for agent in agents:
        agent_id = agent.get('agent_id', agent.get('current_id', ''))
        if 'critical' in agent_id.lower():
            type_counts['critical'] += 1
        elif 'awakened' in agent_id.lower():
            type_counts['awakened'] += 1
        elif 'standard' in agent_id.lower():
            type_counts['standard'] += 1
    
    total = sum(type_counts.values())
    if total == 0:
        return 0.0
    
    H = 0.0
    for count in type_counts.values():
        if count > 0:
            p = count / total
            H -= p * np.log(p)
    
    return round(H, 4)


def calculate_effect_size(group1: List[float], group2: List[float]) -> Tuple[float, str]:
    """
    计算Cohen's d效应量
    
    Returns:
        (d值, 解释)
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    
    if pooled_std == 0:
        return 0.0, "无法计算（标准差为0）"
    
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    if abs(d) < 0.2:
        interpretation = "小效应"
    elif abs(d) < 0.5:
        interpretation = "中小效应"
    elif abs(d) < 0.8:
        interpretation = "中等效应"
    else:
        interpretation = "大效应"
    
    return round(d, 4), interpretation


def calculate_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """计算置信区间"""
    n = len(data)
    if n < 2:
        return (0.0, 0.0)
    
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    
    return (round(mean - h, 4), round(mean + h, 4))


def validate_experiment_data(experiment_file: str) -> Dict[str, Any]:
    """
    验证实验数据的完整性和有效性
    
    遵循soul.md研究宪法：
    - 零容忍模糊：所有结果必须量化
    - 可证伪性：定义明确的证伪条件
    - 透明度：报告所有数据问题
    """
    with open(experiment_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    issues = []
    results = {}
    
    # 1. 验证多样性计算
    agents = list(data.get('ecosystem_state', {}).get('agents', {}).values())
    if agents:
        correct_diversity = calculate_cognitive_diversity_index(agents)
        reported_diversity = data.get('results_history', [{}])[0].get('diversity', 0)
        
        if abs(correct_diversity - reported_diversity) > 0.01:
            issues.append(f"多样性计算错误：报告值={reported_diversity}，正确值={correct_diversity}")
        
        results['diversity'] = {
            'reported': reported_diversity,
            'correct': correct_diversity,
            'type_distribution': {}
        }
        
        # 统计类型分布
        for agent in agents:
            agent_id = agent.get('agent_id', '')
            for t in ['critical', 'awakened', 'standard']:
                if t in agent_id.lower():
                    results['diversity']['type_distribution'][t] = results['diversity']['type_distribution'].get(t, 0) + 1
    
    # 2. 验证性能数据
    history = data.get('results_history', [])
    if history:
        het_perf = [h.get('heterogeneous_performance', 0) for h in history]
        hom_perf = [h.get('homogeneous_performance', {}) for h in history]
        
        results['performance'] = {
            'heterogeneous_mean': np.mean(het_perf) if het_perf else 0,
            'heterogeneous_std': np.std(het_perf) if het_perf else 0,
            'heterogeneous_ci': calculate_confidence_interval(het_perf) if het_perf else (0, 0),
            'generations': len(history)
        }
        
        # 同构系统性能
        hom_types = {}
        for h in hom_perf:
            for k, v in h.items():
                if k not in hom_types:
                    hom_types[k] = []
                hom_types[k].append(v)
        
        results['performance']['homogeneous'] = {}
        for k, v in hom_types.items():
            results['performance']['homogeneous'][k] = {
                'mean': np.mean(v),
                'std': np.std(v),
                'ci': calculate_confidence_interval(v)
            }
    
    # 3. 统计检验
    if history and len(history) >= 2:
        het_perf = [h.get('heterogeneous_performance', 0) for h in history]
        hom_avg = []
        for h in history:
            hom = h.get('homogeneous_performance', {})
            if hom:
                hom_avg.append(np.mean(list(hom.values())))
        
        if len(het_perf) >= 2 and len(hom_avg) >= 2:
            # t检验
            t_stat, p_value = stats.ttest_ind(het_perf, hom_avg)
            
            # 效应量
            d, d_interp = calculate_effect_size(het_perf, hom_avg)
            
            results['statistical_tests'] = {
                't_statistic': round(t_stat, 4),
                'p_value': round(p_value, 6),
                'cohens_d': d,
                'effect_size_interpretation': d_interp,
                'significant_at_0.05': p_value < 0.05,
                'significant_at_0.01': p_value < 0.01
            }
    
    results['issues'] = issues
    results['validation_passed'] = len(issues) == 0
    
    return results


def generate_validation_report(results: Dict, output_file: str = None) -> str:
    """生成符合soul.md标准的验证报告"""
    
    report = []
    report.append("=" * 60)
    report.append("实验数据验证报告")
    report.append("遵循 soul.md 第一原则：诚实、严谨、科学理性")
    report.append("=" * 60)
    report.append(f"\n验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 数据问题
    if results.get('issues'):
        report.append("\n【数据问题】（必须解决）")
        for issue in results['issues']:
            report.append(f"  ⚠️ {issue}")
    else:
        report.append("\n【数据问题】无")
    
    # 2. 多样性验证
    div = results.get('diversity', {})
    if div:
        report.append("\n【认知异质性】")
        report.append(f"  - 正确值: H = {div.get('correct', 'N/A')}")
        report.append(f"  - 类型分布: {div.get('type_distribution', {})}")
        
        # 评估异质性水平
        h = div.get('correct', 0)
        if h >= 0.6:
            report.append("  - 评估: ✅ 高异质性（H ≥ 0.6）")
        elif h >= 0.3:
            report.append("  - 评估: ⚠️ 中等异质性（0.3 ≤ H < 0.6）")
        else:
            report.append("  - 评估: ❌ 低异质性（H < 0.3）")
    
    # 3. 性能统计
    perf = results.get('performance', {})
    if perf:
        report.append("\n【性能统计】")
        report.append(f"  异构系统:")
        report.append(f"    - 平均性能: {perf.get('heterogeneous_mean', 0):.4f}")
        report.append(f"    - 标准差: {perf.get('heterogeneous_std', 0):.4f}")
        report.append(f"    - 95% CI: {perf.get('heterogeneous_ci', (0,0))}")
        
        hom = perf.get('homogeneous', {})
        if hom:
            report.append(f"  同构系统:")
            for k, v in hom.items():
                report.append(f"    - {k}: {v.get('mean', 0):.4f} ± {v.get('std', 0):.4f}")
    
    # 4. 统计检验
    stats_res = results.get('statistical_tests', {})
    if stats_res:
        report.append("\n【统计验证】")
        report.append(f"  - t统计量: t = {stats_res.get('t_statistic', 'N/A')}")
        report.append(f"  - p值: p = {stats_res.get('p_value', 'N/A')}")
        report.append(f"  - 效应量: Cohen's d = {stats_res.get('cohens_d', 'N/A')} ({stats_res.get('effect_size_interpretation', '')})")
        report.append(f"  - 显著性(α=0.05): {'✅ 显著' if stats_res.get('significant_at_0.05') else '❌ 不显著'}")
        report.append(f"  - 显著性(α=0.01): {'✅ 显著' if stats_res.get('significant_at_0.01') else '❌ 不显著'}")
    
    # 5. 结论
    report.append("\n【验证结论】")
    if results.get('validation_passed'):
        report.append("  ✅ 数据验证通过，符合soul.md标准")
    else:
        report.append("  ❌ 数据验证未通过，需修复以下问题后重新验证")
    
    report.append("\n" + "=" * 60)
    
    report_text = "\n".join(report)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
    
    return report_text


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python validate_experiment_data.py <实验数据文件>")
        sys.exit(1)
    
    experiment_file = sys.argv[1]
    
    if not os.path.exists(experiment_file):
        print(f"错误: 文件不存在 {experiment_file}")
        sys.exit(1)
    
    results = validate_experiment_data(experiment_file)
    report = generate_validation_report(results)
    print(report)
