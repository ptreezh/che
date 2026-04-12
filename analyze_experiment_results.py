#!/usr/bin/env python3
"""
CHE Two-Group Experiment Analysis
Calculates Cohen's d and validates data consistency
"""

import json
import numpy as np
from pathlib import Path

def load_jsonl(filepath):
    """Load JSONL file"""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d with confidence interval"""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    
    # Cohen's d
    d = (mean1 - mean2) / pooled_std
    
    # Confidence interval (approximate)
    se = np.sqrt((n1+n2)/(n1*n2) + d**2/(2*(n1+n2)))
    ci_lower = d - 1.96 * se
    ci_upper = d + 1.96 * se
    
    return d, ci_lower, ci_upper, mean1, mean2, pooled_std

def main():
    data_dir = Path("experiment_data/CHE_TWO_GROUP_EXPERIMENT")
    
    # Load data
    print("=" * 60)
    print("CHE Two-Group Experiment Analysis")
    print("=" * 60)
    
    het_file = data_dir / "heterogeneous_responses.jsonl"
    hom_file = data_dir / "homogeneous_responses.jsonl"
    
    het_data = load_jsonl(het_file)
    hom_data = load_jsonl(hom_file)
    
    print(f"\n📊 Data Loading:")
    print(f"   Heterogeneous: {len(het_data)} records")
    print(f"   Homogeneous: {len(hom_data)} records")
    print(f"   Total: {len(het_data) + len(hom_data)} records")
    
    # Extract scores
    het_scores = [r['score'] for r in het_data]
    hom_scores = [r['score'] for r in hom_data]
    
    # Calculate statistics
    d, ci_lower, ci_upper, mean_het, mean_hom, pooled_std = calculate_cohens_d(het_scores, hom_scores)
    
    # Calculate improvement
    improvement = ((mean_het - mean_hom) / mean_hom) * 100
    
    print(f"\n📈 Descriptive Statistics:")
    print(f"   Heterogeneous: Mean={mean_het:.3f}, SD={np.std(het_scores, ddof=1):.3f}, N={len(het_scores)}")
    print(f"   Homogeneous: Mean={mean_hom:.3f}, SD={np.std(hom_scores, ddof=1):.3f}, N={len(hom_scores)}")
    
    print(f"\n📊 Effect Size (Cohen's d):")
    print(f"   d = {d:.2f} (95% CI: [{ci_lower:.2f}, {ci_upper:.2f}])")
    print(f"   Interpretation: {'LARGE' if abs(d) >= 0.8 else 'MEDIUM' if abs(d) >= 0.5 else 'SMALL'} effect")
    
    print(f"\n📈 Improvement:")
    print(f"   {improvement:.1f}% higher score in heterogeneous group")
    
    # Model distribution
    print(f"\n🤖 Model Distribution:")
    het_models = {}
    hom_models = {}
    for r in het_data:
        m = r.get('model_name', 'unknown')
        het_models[m] = het_models.get(m, 0) + 1
    for r in hom_data:
        m = r.get('model_name', 'unknown')
        hom_models[m] = hom_models.get(m, 0) + 1
    
    print(f"   Heterogeneous models: {het_models}")
    print(f"   Homogeneous models: {hom_models}")
    
    # Agent type distribution
    print(f"\n👥 Agent Type Distribution:")
    het_types = {}
    for r in het_data:
        t = r.get('agent_type', 'unknown')
        het_types[t] = het_types.get(t, 0) + 1
    print(f"   Heterogeneous: {het_types}")
    
    # Score distribution
    print(f"\n📊 Score Distribution:")
    print(f"   Heterogeneous: min={min(het_scores)}, max={max(het_scores)}, median={np.median(het_scores):.2f}")
    print(f"   Homogeneous: min={min(hom_scores)}, max={max(hom_scores)}, median={np.median(hom_scores):.2f}")
    
    # Validation
    print(f"\n✅ Data Validation:")
    issues = []
    
    # Check for missing fields
    for i, r in enumerate(het_data):
        if 'score' not in r or 'experimental_group' not in r:
            issues.append(f"Heterogeneous record {i}: missing field")
    
    for i, r in enumerate(hom_data):
        if 'score' not in r or 'experimental_group' not in r:
            issues.append(f"Homogeneous record {i}: missing field")
    
    # Check group labels
    for r in het_data:
        if r.get('experimental_group') != 'heterogeneous':
            issues.append(f"Wrong group label in heterogeneous: {r.get('experimental_group')}")
    
    for r in hom_data:
        if r.get('experimental_group') != 'homogeneous':
            issues.append(f"Wrong group label in homogeneous: {r.get('experimental_group')}")
    
    if issues:
        print(f"   ⚠️ Found {len(issues)} issues:")
        for issue in issues[:10]:
            print(f"      - {issue}")
    else:
        print("   ✅ All records valid")
    
    print(f"\n" + "=" * 60)
    print("SUMMARY FOR PAPER")
    print("=" * 60)
    print(f"Cohen's d = {d:.2f} (95% CI [{ci_lower:.2f}, {ci_upper:.2f}])")
    print(f"N = {len(het_scores) + len(hom_scores)} ({len(het_scores)} heterogeneous + {len(hom_scores)} homogeneous)")
    print(f"Improvement = {improvement:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()