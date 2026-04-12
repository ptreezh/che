#!/usr/bin/env python3
"""CHE实验数据分析 - 对齐论文结果"""
import json
import numpy as np
from scipy import stats

def load_scores(filepath):
    scores = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            scores.append(record['score'])
    return np.array(scores)

# 加载数据
old_control = load_scores('experiment_data/old_standard_control.jsonl')
heterogeneous = load_scores('experiment_data/CHE_TWO_GROUP_EXPERIMENT/heterogeneous_responses.jsonl')
homogeneous = load_scores('experiment_data/CHE_TWO_GROUP_EXPERIMENT/homogeneous_responses.jsonl')

print('='*70)
print('CHE实验数据完整分析报告')
print('='*70)

print('\n1. 描述性统计')
print('-'*70)
print(f'Old Control (llama3.2:1b):  N={len(old_control)}, Mean={old_control.mean():.3f}, SD={old_control.std(ddof=1):.3f}')
print(f'Heterogeneous (5模型):      N={len(heterogeneous)}, Mean={heterogeneous.mean():.3f}, SD={heterogeneous.std(ddof=1):.3f}')
print(f'Homogeneous (glm-4.7-flash): N={len(homogeneous)}, Mean={homogeneous.mean():.3f}, SD={homogeneous.std(ddof=1):.3f}')

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    s1, s2 = group1.std(ddof=1), group2.std(ddof=1)
    s_pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    d = (group1.mean() - group2.mean()) / s_pooled
    return d, s_pooled

print('\n2. 论文核心结果: Heterogeneous vs Old Control')
print('-'*70)
d1, _ = cohens_d(heterogeneous, old_control)
t1, p1 = stats.ttest_ind(heterogeneous, old_control)
imp1 = (heterogeneous.mean() - old_control.mean()) / old_control.mean() * 100

# 95% CI for Cohen's d
n1, n2 = len(heterogeneous), len(old_control)
se_d = np.sqrt((n1+n2)/(n1*n2) + d1**2/(2*(n1+n2)))
ci_lower = d1 - 1.96*se_d
ci_upper = d1 + 1.96*se_d

print(f"Cohen's d = {d1:.3f}")
print(f"95% CI = [{ci_lower:.2f}, {ci_upper:.2f}]")
print(f"p-value = {p1:.2e}")
print(f"改进率 = {imp1:.1f}%")

print('\n3. 新对照结果: Heterogeneous vs New Homogeneous')
print('-'*70)
d2, _ = cohens_d(heterogeneous, homogeneous)
t2, p2 = stats.ttest_ind(heterogeneous, homogeneous)
imp2 = (heterogeneous.mean() - homogeneous.mean()) / homogeneous.mean() * 100

n1, n2 = len(heterogeneous), len(homogeneous)
se_d = np.sqrt((n1+n2)/(n1*n2) + d2**2/(2*(n1+n2)))
ci_lower = d2 - 1.96*se_d
ci_upper = d2 + 1.96*se_d

print(f"Cohen's d = {d2:.3f}")
print(f"95% CI = [{ci_lower:.2f}, {ci_upper:.2f}]")
print(f"p-value = {p2:.2e}")
print(f"改进率 = {imp2:.1f}%")

print('\n4. 关键发现')
print('-'*70)
print(f'glm-4.7-flash (Mean={homogeneous.mean():.3f}) 显著优于 llama3.2:1b (Mean={old_control.mean():.3f})')
d3, _ = cohens_d(homogeneous, old_control)
t3, p3 = stats.ttest_ind(homogeneous, old_control)
print(f"  两对照组差异: Cohen's d = {d3:.3f}, p = {p3:.2e}")

print('\n5. 结论')
print('-'*70)
if d1 > 0.8:
    print(f'论文核心结论成立: Heterogeneous vs Old Control, Cohen d = {d1:.3f} (大效应)')
else:
    print(f'论文核心结论需要修正: Cohen d = {d1:.3f}')

if d2 > 0:
    print(f'新对照支持假设: Heterogeneous > Homogeneous')
else:
    print(f'新对照不支持假设: Homogeneous (单一最佳模型) 表现更好')

print('='*70)
