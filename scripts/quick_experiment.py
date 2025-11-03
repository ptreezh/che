#!/usr/bin/env python3
"""
快速实验脚本 - 直接运行科学实验

Authors: Zhang Shuren, AI Personality LAB
Date: 2025-09-20
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.demo_experiment import run_simple_demo, run_comparison_demo

def run_full_factorial_experiment():
    """运行完整的2×2×3因子实验"""

    from scripts.run_experiment import ExperimentRunner

    print("🧬 开始完整的2×2×3因子科学实验")
    print("这将运行多个实验条件，可能需要一些时间...")

    runner = ExperimentRunner()

    # 运行中等规模的因子实验
    results = runner.run_factorial_experiment(
        replications=2,  # 每个条件2次重复
        generations=10   # 10代演化
    )

    print(f"\n✅ 实验完成! 共运行了 {len(results)} 个实验")
    print("结果保存在 results/ 目录中")

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='认知异质性实验系统')
    parser.add_argument('--type', choices=['demo', 'comparison', 'factorial'],
                       default='demo', help='实验类型')

    args = parser.parse_args()

    if args.type == 'demo':
        print("🎯 运行演示实验...")
        run_simple_demo()

    elif args.type == 'comparison':
        print("🔬 运行对比实验...")
        run_comparison_demo()

    elif args.type == 'factorial':
        print("🧬 运行完整因子实验...")
        run_full_factorial_experiment()