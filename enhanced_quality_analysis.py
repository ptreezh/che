"""
Enhanced Statistical Analysis for CHE Paper
Quality Enhancement and Robustness Validation

This script performs comprehensive statistical analysis including:
- Effect size with confidence intervals
- Normality tests
- Non-parametric alternatives
- Bootstrap validation
- Cross-model meta-analysis
"""

import json
import os
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


def load_experiment_data(filepath: str) -> Dict:
    """Load experiment data from JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return None


def extract_metrics(data: Dict) -> Tuple[List[float], List[float], float]:
    """Extract performance metrics from experiment data."""
    hetero_scores = []
    homo_scores = []
    diversity = None
    
    if not data:
        return hetero_scores, homo_scores, diversity
    
    # Extract from results_history (main source of performance data)
    if 'results_history' in data:
        for r in data['results_history']:
            hp = r.get('heterogeneous_performance', r.get('hetero_perf'))
            op = r.get('homogeneous_performance', r.get('homo_perf'))
            
            if isinstance(hp, (int, float)) and hp > 0:
                hetero_scores.append(float(hp))
            if isinstance(op, (int, float)) and op > 0:
                homo_scores.append(float(op))
    
    # Extract diversity from ecosystem_state
    if 'ecosystem_state' in data:
        eco = data['ecosystem_state']
        if 'diversity_metrics' in eco:
            div_metrics = eco['diversity_metrics']
            if isinstance(div_metrics, dict):
                diversity = div_metrics.get('shannon_entropy_normalized', 
                                           div_metrics.get('shannon_entropy'))
    
    return hetero_scores, homo_scores, diversity


def cohens_d_with_ci(group1: List[float], group2: List[float], 
                     confidence: float = 0.95) -> Dict:
    """Calculate Cohen's d with confidence interval using non-central t-distribution."""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
    
    # Standard error of d (approximate)
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    
    # Confidence interval
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    ci_lower = d - z * se_d
    ci_upper = d + z * se_d
    
    return {
        'd': d,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'se': se_d,
        'interpretation': interpret_effect_size(d)
    }


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def bootstrap_ci(group1: List[float], group2: List[float], 
                 n_bootstrap: int = 10000, confidence: float = 0.95) -> Dict:
    """Bootstrap confidence interval for Cohen's d."""
    d_values = []
    n1, n2 = len(group1), len(group2)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample1 = np.random.choice(group1, size=n1, replace=True)
        sample2 = np.random.choice(group2, size=n2, replace=True)
        
        # Calculate d for this sample
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
        
        if pooled_std > 0:
            d_values.append((mean1 - mean2) / pooled_std)
    
    d_values = np.array(d_values)
    
    # Percentile CI
    alpha = 1 - confidence
    ci_lower = np.percentile(d_values, 100 * alpha / 2)
    ci_upper = np.percentile(d_values, 100 * (1 - alpha / 2))
    
    return {
        'mean_d': np.mean(d_values),
        'median_d': np.median(d_values),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'std_d': np.std(d_values)
    }


def normality_tests(data: List[float], name: str) -> Dict:
    """Perform normality tests."""
    if len(data) < 3:
        return {'error': 'Insufficient data'}
    
    # Shapiro-Wilk test (for n < 5000)
    if len(data) <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(data)
    else:
        shapiro_stat, shapiro_p = None, None
    
    # D'Agostino-Pearson test (for n >= 20)
    if len(data) >= 20:
        dagostino_stat, dagostino_p = stats.normaltest(data)
    else:
        dagostino_stat, dagostino_p = None, None
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
    
    return {
        'name': name,
        'n': len(data),
        'shapiro_w': shapiro_stat,
        'shapiro_p': shapiro_p,
        'dagostino_chi2': dagostino_stat,
        'dagostino_p': dagostino_p,
        'ks_stat': ks_stat,
        'ks_p': ks_p,
        'is_normal_05': (shapiro_p is not None and shapiro_p > 0.05) or 
                        (dagostino_p is not None and dagostino_p > 0.05)
    }


def non_parametric_tests(group1: List[float], group2: List[float]) -> Dict:
    """Perform non-parametric tests for robustness."""
    
    # Mann-Whitney U test
    u_stat, u_p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    
    # Wilcoxon rank-sum (equivalent to Mann-Whitney)
    w_stat, w_p = stats.ranksums(group1, group2)
    
    # Effect size: rank-biserial correlation
    n1, n2 = len(group1), len(group2)
    r_biserial = 1 - (2 * u_stat) / (n1 * n2)  # Rank-biserial correlation
    
    # Cliff's delta
    cliff_delta = r_biserial  # Equivalent interpretation
    
    return {
        'mann_whitney_u': u_stat,
        'mann_whitney_p': u_p,
        'wilcoxon_z': w_stat,
        'wilcoxon_p': w_p,
        'rank_biserial_r': r_biserial,
        'cliff_delta': cliff_delta,
        'significant_05': u_p < 0.05,
        'significant_001': u_p < 0.001
    }


def power_analysis(d: float, n1: int, n2: int, alpha: float = 0.05) -> Dict:
    """Calculate achieved statistical power."""
    from scipy import stats
    
    # Effect size for power analysis
    effect_size = abs(d)
    
    # Total sample size
    n_total = n1 + n2
    
    # Approximate power calculation using normal approximation
    # For two-sample t-test
    n_harmonic = (2 * n1 * n2) / (n1 + n2)
    delta = effect_size * np.sqrt(n_harmonic / 2)
    
    # Critical value for alpha
    z_crit = stats.norm.ppf(1 - alpha/2)
    
    # Power
    power = stats.norm.cdf(delta - z_crit) + stats.norm.sf(delta + z_crit)
    
    return {
        'effect_size': effect_size,
        'n1': n1,
        'n2': n2,
        'alpha': alpha,
        'power': power,
        'power_adequate': power >= 0.80
    }


def cross_model_meta_analysis(results: List[Dict]) -> Dict:
    """Perform meta-analysis across different models."""
    
    d_values = [r['cohens_d'] for r in results if 'cohens_d' in r]
    n_values = [r.get('n', 100) for r in results if 'cohens_d' in r]
    
    if not d_values:
        return {'error': 'No valid effect sizes'}
    
    # Fixed effects meta-analysis
    weights = [n / sum(n_values) for n in n_values]
    weighted_d = sum(d * w for d, w in zip(d_values, weights))
    
    # Heterogeneity (Q statistic)
    q = sum(w * (d - weighted_d)**2 for d, w in zip(d_values, weights))
    
    # I-squared
    k = len(d_values)
    if k > 1:
        i_squared = max(0, (q - (k - 1)) / q * 100) if q > 0 else 0
    else:
        i_squared = 0
    
    return {
        'k_models': k,
        'weighted_d': weighted_d,
        'q_statistic': q,
        'i_squared': i_squared,
        'heterogeneity': 'low' if i_squared < 25 else ('moderate' if i_squared < 75 else 'high'),
        'individual_d': d_values
    }


def collect_all_data(base_path: str) -> Tuple[List[float], List[float], List[float]]:
    """Collect data from all experiment files."""
    hetero_scores = []
    homo_scores = []
    diversities = []
    
    experiments_path = Path(base_path) / 'experiments'
    gemma3_path = Path(base_path) / 'experiments_gemma3'
    
    # Load main experiments
    if experiments_path.exists():
        for f in experiments_path.glob('*.json'):
            data = load_experiment_data(str(f))
            if data:
                h_scores, hm_scores, div = extract_metrics(data)
                hetero_scores.extend(h_scores)
                homo_scores.extend(hm_scores)
                if div is not None:
                    diversities.append(div)
    
    # Load gemma3 experiments
    if gemma3_path.exists():
        for f in gemma3_path.glob('*.json'):
            data = load_experiment_data(str(f))
            if data:
                h_scores, hm_scores, div = extract_metrics(data)
                hetero_scores.extend(h_scores)
                homo_scores.extend(hm_scores)
                if div is not None:
                    diversities.append(div)
    
    return hetero_scores, homo_scores, diversities


def main():
    print("="*80)
    print("CHE 论文数据质量增强分析")
    print("="*80)
    print()
    
    # Collect all data
    base_path = Path(__file__).parent
    hetero_scores, homo_scores, diversities = collect_all_data(str(base_path))
    
    print(f"数据收集结果:")
    print(f"  异质系统样本: {len(hetero_scores)} 条")
    print(f"  同质系统样本: {len(homo_scores)} 条")
    print(f"  多样性记录: {len(diversities)} 条")
    print()
    
    if len(hetero_scores) < 3 or len(homo_scores) < 3:
        print("错误: 样本量不足，无法进行分析")
        return
    
    # 1. Descriptive Statistics
    print("-"*80)
    print("1. 描述性统计")
    print("-"*80)
    print(f"异质系统: M = {np.mean(hetero_scores):.4f}, SD = {np.std(hetero_scores, ddof=1):.4f}")
    print(f"同质系统: M = {np.mean(homo_scores):.4f}, SD = {np.std(homo_scores, ddof=1):.4f}")
    print(f"改进幅度: {(np.mean(hetero_scores) - np.mean(homo_scores)) / np.mean(homo_scores) * 100:.1f}%")
    print()
    
    # 2. Normality Tests
    print("-"*80)
    print("2. 正态性检验")
    print("-"*80)
    norm_hetero = normality_tests(hetero_scores, "异质系统")
    norm_homo = normality_tests(homo_scores, "同质系统")
    
    print(f"异质系统正态性 (Shapiro-Wilk): W = {norm_hetero['shapiro_w']:.4f}, p = {norm_hetero['shapiro_p']:.2e}")
    print(f"同质系统正态性 (Shapiro-Wilk): W = {norm_homo['shapiro_w']:.4f}, p = {norm_homo['shapiro_p']:.2e}")
    print(f"结论: {'正态分布假设成立' if norm_hetero['is_normal_05'] and norm_homo['is_normal_05'] else '建议使用非参数检验'}")
    print()
    
    # 3. Effect Size with CI
    print("-"*80)
    print("3. 效应量分析 (Cohen's d)")
    print("-"*80)
    d_result = cohens_d_with_ci(hetero_scores, homo_scores)
    print(f"Cohen's d = {d_result['d']:.4f}")
    print(f"95% CI = [{d_result['ci_lower']:.4f}, {d_result['ci_upper']:.4f}]")
    print(f"标准误 = {d_result['se']:.4f}")
    print(f"解释: {d_result['interpretation']} 效应")
    print()
    
    # 4. Bootstrap Validation
    print("-"*80)
    print("4. Bootstrap 验证 (10,000次重采样)")
    print("-"*80)
    bootstrap_result = bootstrap_ci(hetero_scores, homo_scores, n_bootstrap=10000)
    print(f"Bootstrap d = {bootstrap_result['mean_d']:.4f} ± {bootstrap_result['std_d']:.4f}")
    print(f"Bootstrap 95% CI = [{bootstrap_result['ci_lower']:.4f}, {bootstrap_result['ci_upper']:.4f}]")
    print()
    
    # 5. Parametric t-test
    print("-"*80)
    print("5. 参数检验 (独立样本t检验)")
    print("-"*80)
    t_stat, t_p = stats.ttest_ind(hetero_scores, homo_scores)
    print(f"t({len(hetero_scores) + len(homo_scores) - 2}) = {t_stat:.4f}")
    print(f"p = {t_p:.2e}")
    print(f"结论: {'高度显著' if t_p < 0.001 else '显著' if t_p < 0.05 else '不显著'}")
    print()
    
    # 6. Non-parametric tests
    print("-"*80)
    print("6. 非参数检验 (稳健性验证)")
    print("-"*80)
    nonpara = non_parametric_tests(hetero_scores, homo_scores)
    print(f"Mann-Whitney U = {nonpara['mann_whitney_u']:.1f}, p = {nonpara['mann_whitney_p']:.2e}")
    print(f"Wilcoxon Z = {nonpara['wilcoxon_z']:.4f}, p = {nonpara['wilcoxon_p']:.2e}")
    print(f"Rank-biserial r = {nonpara['rank_biserial_r']:.4f}")
    print(f"Cliff's δ = {nonpara['cliff_delta']:.4f}")
    print(f"结论: {'结果稳健，非参数检验同样显著' if nonpara['significant_001'] else '需谨慎解释'}")
    print()
    
    # 7. Power Analysis
    print("-"*80)
    print("7. 统计功效分析")
    print("-"*80)
    power = power_analysis(d_result['d'], len(hetero_scores), len(homo_scores))
    print(f"效应量 (d) = {power['effect_size']:.4f}")
    print(f"样本量: n1 = {power['n1']}, n2 = {power['n2']}")
    print(f"统计功效 = {power['power']:.4f}")
    print(f"功效充足: {'是 (≥ 0.80)' if power['power_adequate'] else '否 (< 0.80)'}")
    print()
    
    # 8. Summary
    print("="*80)
    print("数据质量评估总结")
    print("="*80)
    print()
    print("┌" + "─"*76 + "┐")
    print(f"│ {'指标':<20} {'值':<20} {'评估':<32} │")
    print("├" + "─"*76 + "┤")
    print(f"│ {'效应量 (Cohen\'s d)':<18} {d_result['d']:.4f} ({d_result['interpretation']}){' '*20} │")
    print(f"│ {'95% CI':<20} [{d_result['ci_lower']:.4f}, {d_result['ci_upper']:.4f}]{' '*28} │")
    print(f"│ {'p值':<20} {t_p:.2e}{' '*34} │")
    print(f"│ {'样本量':<20} n = {len(hetero_scores) + len(homo_scores)}{' '*42} │")
    print(f"│ {'统计功效':<20} {power['power']:.4f}{' '*34} │")
    print(f"│ {'结果稳健性':<20} {'已验证 (参数+非参数)' if nonpara['significant_001'] else '需验证'}{' '*20} │")
    print("└" + "─"*76 + "┘")
    print()
    
    # Quality metrics
    print("数据质量指标:")
    quality_score = 0
    max_score = 5
    
    if d_result['d'] >= 0.8:
        print("  ✓ 效应量充足 (d ≥ 0.8)")
        quality_score += 1
    else:
        print("  ✗ 效应量偏小 (d < 0.8)")
    
    if t_p < 0.001:
        print("  ✓ 统计显著性高 (p < 0.001)")
        quality_score += 1
    else:
        print("  ✗ 统计显著性不足")
    
    if power['power'] >= 0.80:
        print("  ✓ 统计功效充足 (≥ 0.80)")
        quality_score += 1
    else:
        print("  ✗ 统计功效不足")
    
    if nonpara['significant_001']:
        print("  ✓ 非参数验证通过")
        quality_score += 1
    else:
        print("  ✗ 非参数验证未通过")
    
    if d_result['ci_lower'] > 0.5:
        print("  ✓ 置信区间下限 > 0.5")
        quality_score += 1
    else:
        print("  ✗ 置信区间下限偏低")
    
    print()
    print(f"数据质量评分: {quality_score}/{max_score} ({quality_score/max_score*100:.0f}%)")
    print()
    
    # Save results
    results = {
        'cohens_d': d_result['d'],
        'ci_lower': d_result['ci_lower'],
        'ci_upper': d_result['ci_upper'],
        'p_value': t_p,
        'power': power['power'],
        'n_hetero': len(hetero_scores),
        'n_homo': len(homo_scores),
        'bootstrap_d': bootstrap_result['mean_d'],
        'bootstrap_ci_lower': bootstrap_result['ci_lower'],
        'bootstrap_ci_upper': bootstrap_result['ci_upper'],
        'mann_whitney_p': nonpara['mann_whitney_p'],
        'quality_score': quality_score
    }
    
    output_path = Path(__file__).parent / 'enhanced_quality_analysis.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"分析结果已保存: {output_path}")


if __name__ == '__main__':
    main()
