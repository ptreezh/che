#!/usr/bin/env python3
"""
CHE补充实验 - 试点数据分析
生成统计报告和可视化
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_pilot_results(results_dir: Path):
    """加载所有组的结果"""
    groups = {}
    for group_id in ['A', 'B', 'C', 'D', 'E', 'F']:
        file_path = results_dir / f"group_{group_id}_result.json"
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                groups[group_id] = json.load(f)
        else:
            print(f"警告: 未找到组 {group_id} 的结果文件")
    return groups

def extract_final_scores(groups: dict):
    """提取各组最终分数"""
    final_scores = {}
    for gid, data in groups.items():
        # 确保键是字符串
        key = str(gid)
        final_scores[key] = data['generations'][-1]['mean_score']
    return final_scores

def perform_statistical_analysis(final_scores: dict):
    """执行统计分析"""
    # 组定义
    group_definitions = {
        'A': ('deepseek', 'single'),
        'B': ('deepseek', 'multi'),
        'C': ('glm4', 'single'),
        'D': ('glm4', 'multi'),
        'E': ('mixed', 'single'),
        'F': ('mixed', 'multi')
    }
    
    print("="*70)
    print("试点实验统计分析报告")
    print("="*70)
    
    # 1. 描述性统计
    print("\n【1. 描述性统计】")
    print(f"{'组':<6} {'模型':<15} {'角色':<10} {'分数':<8}")
    print("-"*45)
    for gid in ['A', 'B', 'C', 'D', 'E', 'F']:
        model, role = group_definitions[gid]
        score = final_scores[gid]
        print(f"{gid:<6} {model:<15} {role:<10} {score:<8.3f}")
    
    # 2. 关键对比
    print("\n【2. 关键对比分析】")
    comparisons = [
        ('F', 'A', '总异质性效应'),
        ('F', 'E', '角色多样性贡献'),
        ('F', 'B', '模型多样性贡献'),
        ('F', 'D', '异质性vs先天最强'),
        ('B', 'A', 'DeepSeek纯角色效应'),
        ('E', 'A', '混合模型纯模型效应'),
    ]
    
    for g1, g2, desc in comparisons:
        score1 = final_scores[g1]
        score2 = final_scores[g2]
        diff = score1 - score2
        pct = (diff / score2) * 100
        print(f"\n{desc}:")
        print(f"  {g1} ({score1:.3f}) vs {g2} ({score2:.3f})")
        print(f"  差异: {diff:+.3f} ({pct:+.1f}%)")
        
        # 简单效应量（基于假设的标准差0.2）
        cohens_d = diff / 0.2
        print(f"  Cohen's d (估计): {cohens_d:.2f}")
    
    # 3. 因子效应分析
    print("\n【3. 因子效应分析】")
    
    # 模型类型效应
    model_means = {
        'deepseek': np.mean([final_scores['A'], final_scores['B']]),
        'glm4': np.mean([final_scores['C'], final_scores['D']]),
        'mixed': np.mean([final_scores['E'], final_scores['F']])
    }
    
    print("\n模型类型边际均值:")
    for model, mean in model_means.items():
        print(f"  {model}: {mean:.3f}")
    
    # 角色多样性效应
    role_single_mean = np.mean([final_scores['A'], final_scores['C'], final_scores['E']])
    role_multi_mean = np.mean([final_scores['B'], final_scores['D'], final_scores['F']])
    
    print(f"\n角色多样性边际均值:")
    print(f"  单角色: {role_single_mean:.3f}")
    print(f"  多角色: {role_multi_mean:.3f}")
    print(f"  差异: {role_multi_mean - role_single_mean:+.3f}")
    
    # 4. 成功标准评估
    print("\n【4. 试点成功标准评估】")
    
    scores = list(final_scores.values())
    cv = np.std(scores) / np.mean(scores) * 100
    
    print(f"  变异系数 (CV): {cv:.1f}%")
    print(f"  标准: CV < 20% {'✓ 通过' if cv < 20 else '✗ 未通过'}")
    
    # 检查方向一致性
    expected_patterns = [
        final_scores['B'] > final_scores['A'],  # DeepSeek多角色 > 单角色
        final_scores['D'] > final_scores['C'],  # GLM4多角色 > 单角色
        final_scores['F'] > final_scores['E'],  # 混合多角色 > 单角色
    ]
    
    print(f"\n  方向一致性:")
    print(f"    B>A: {final_scores['B']:.3f} > {final_scores['A']:.3f} {'✓' if expected_patterns[0] else '✗'}")
    print(f"    D>C: {final_scores['D']:.3f} > {final_scores['C']:.3f} {'✓' if expected_patterns[1] else '✗'}")
    print(f"    F>E: {final_scores['F']:.3f} > {final_scores['E']:.3f} {'✓' if expected_patterns[2] else '✗'}")
    
    return final_scores

def create_visualizations(groups: dict, final_scores: dict, output_dir: Path):
    """创建可视化图表"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 最终分数对比柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    group_ids = ['A', 'B', 'C', 'D', 'E', 'F']
    scores = [final_scores[gid] for gid in group_ids]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    
    bars = ax.bar(group_ids, scores, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('实验组', fontsize=12)
    ax.set_ylabel('最终平均分', fontsize=12)
    ax.set_title('试点实验：各组最终表现对比', fontsize=14, fontweight='bold')
    ax.set_ylim(6.0, 9.0)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'final_scores_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ 图表已保存: {output_dir / 'final_scores_comparison.png'}")
    plt.close()
    
    # 2. 进化轨迹图
    fig, ax = plt.subplots(figsize=(12, 7))
    
    group_definitions = {
        'A': ('DeepSeek单角色', '#FF6B6B'),
        'B': ('DeepSeek多角色', '#FF8E8E'),
        'C': ('GLM4单角色', '#4ECDC4'),
        'D': ('GLM4多角色', '#6EDDD6'),
        'E': ('混合单角色', '#45B7D1'),
        'F': ('混合多角色', '#67C2D9'),
    }
    
    for gid in ['A', 'B', 'C', 'D', 'E', 'F']:
        data = groups[gid]
        generations = [g['generation'] for g in data['generations']]
        scores = [g['mean_score'] for g in data['generations']]
        label, color = group_definitions[gid]
        
        ax.plot(generations, scores, marker='o', linewidth=2, 
                label=f'{gid}: {label}', color=color, markersize=6)
    
    ax.set_xlabel('代数', fontsize=12)
    ax.set_ylabel('平均分', fontsize=12)
    ax.set_title('试点实验：各组进化轨迹', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'evolution_trajectories.png', dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存: {output_dir / 'evolution_trajectories.png'}")
    plt.close()
    
    # 3. 因子效应图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 模型类型效应
    model_means = {
        'DeepSeek': np.mean([final_scores['A'], final_scores['B']]),
        'GLM4': np.mean([final_scores['C'], final_scores['D']]),
        '混合': np.mean([final_scores['E'], final_scores['F']])
    }
    
    ax1.bar(model_means.keys(), model_means.values(), 
            color=['#FF6B6B', '#4ECDC4', '#45B7D1'], edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('边际均值', fontsize=12)
    ax1.set_title('模型类型主效应', fontsize=13, fontweight='bold')
    ax1.set_ylim(6.5, 8.5)
    ax1.grid(axis='y', alpha=0.3)
    
    for i, (k, v) in enumerate(model_means.items()):
        ax1.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 角色多样性效应
    role_single = np.mean([final_scores['A'], final_scores['C'], final_scores['E']])
    role_multi = np.mean([final_scores['B'], final_scores['D'], final_scores['F']])
    
    ax2.bar(['单角色', '多角色'], [role_single, role_multi],
            color=['#95A5A6', '#2ECC71'], edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('边际均值', fontsize=12)
    ax2.set_title('角色多样性主效应', fontsize=13, fontweight='bold')
    ax2.set_ylim(6.5, 8.5)
    ax2.grid(axis='y', alpha=0.3)
    
    ax2.text(0, role_single, f'{role_single:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.text(1, role_multi, f'{role_multi:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'factor_effects.png', dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存: {output_dir / 'factor_effects.png'}")
    plt.close()

def generate_report(groups: dict, final_scores: dict, output_dir: Path):
    """生成实验报告"""
    report_path = output_dir / 'ANALYSIS_REPORT.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# CHE补充实验 - 试点数据分析报告\n\n")
        f.write("**生成时间**: " + str(np.datetime64('now')) + "\n\n")
        
        f.write("## 1. 实验概况\n\n")
        f.write("- **实验类型**: 试点研究（Pilot Study）\n")
        f.write("- **组数**: 6组\n")
        f.write(f"- **每组样本**: {groups['A']['config']['population_size']}代理\n")
        f.write(f"- **代数**: {groups['A']['config']['generations']}代\n\n")
        
        f.write("## 2. 主要结果\n\n")
        f.write("| 组 | 模型类型 | 角色配置 | 最终分数 |\n")
        f.write("|---|---------|---------|---------|\n")
        group_defs = {
            'A': ('DeepSeek', '单角色'),
            'B': ('DeepSeek', '多角色'),
            'C': ('GLM4', '单角色'),
            'D': ('GLM4', '多角色'),
            'E': ('混合', '单角色'),
            'F': ('混合', '多角色'),
        }
        for gid in ['A', 'B', 'C', 'D', 'E', 'F']:
            model, role = group_defs[gid]
            score = final_scores[gid]
            f.write(f"| {gid} | {model} | {role} | {score:.3f} |\n")
        
        f.write("\n## 3. 关键发现\n\n")
        
        # 计算关键对比
        effects = {
            '总异质性效应': (final_scores['F'] - final_scores['A']) / final_scores['A'] * 100,
            '角色多样性贡献': (final_scores['F'] - final_scores['E']) / final_scores['E'] * 100,
            '模型多样性贡献': (final_scores['F'] - final_scores['B']) / final_scores['B'] * 100,
        }
        
        for desc, pct in effects.items():
            f.write(f"- **{desc}**: {pct:+.1f}%\n")
        
        f.write("\n## 4. 成功标准评估\n\n")
        scores = list(final_scores.values())
        cv = np.std(scores) / np.mean(scores) * 100
        f.write(f"- 变异系数 (CV): {cv:.1f}% {'✓ 通过' if cv < 20 else '✗ 未通过'}\n")
        
        f.write("\n## 5. 结论与建议\n\n")
        f.write("[在此添加分析结论和下一步建议]\n\n")
        
        f.write("## 6. 附件\n\n")
        f.write("- 最终分数对比图: `final_scores_comparison.png`\n")
        f.write("- 进化轨迹图: `evolution_trajectories.png`\n")
        f.write("- 因子效应图: `factor_effects.png`\n")
    
    print(f"\n✓ 分析报告已保存: {report_path}")

def main():
    """主函数"""
    results_dir = Path('pilot_results')
    
    if not results_dir.exists():
        print("错误: 未找到pilot_results目录。请先运行试点实验。")
        return
    
    print("加载实验结果...")
    groups = load_pilot_results(results_dir)
    
    if len(groups) < 6:
        print(f"警告: 仅找到{len(groups)}组结果，期望6组")
    
    final_scores = extract_final_scores(groups)
    
    print("\n执行统计分析...")
    perform_statistical_analysis(final_scores)
    
    print("\n生成可视化图表...")
    create_visualizations(groups, final_scores, results_dir)
    
    print("\n生成实验报告...")
    generate_report(groups, final_scores, results_dir)
    
    print("\n" + "="*70)
    print("分析完成！")
    print("="*70)

if __name__ == '__main__':
    main()
