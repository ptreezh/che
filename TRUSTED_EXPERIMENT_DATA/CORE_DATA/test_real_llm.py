#!/usr/bin/env python3
"""
真实LLM快速测试 - 验证功能正常
只运行组A的第1代
"""

import sys
sys.path.insert(0, '.')

from pilot_experiment import (
    initialize_population, simulate_task_evaluation,
    POPULATION_SIZE, ollama_client, TASKS
)

def main():
    print("="*70)
    print("真实LLM快速测试")
    print("="*70)
    print(f"测试配置：")
    print(f"  - 组：A (DeepSeek单角色)")
    print(f"  - 代理数：{POPULATION_SIZE}")
    print(f"  - 任务数：{len(TASKS)}")
    print(f"  - 代数：仅第1代（测试）")
    print(f"预计时间：2-5分钟")
    print("="*70)
    
    # 初始化种群
    print("\n1. 初始化种群...")
    population = initialize_population('A', 'deepseek', 'single')
    print(f"   ✓ 创建了 {len(population)} 个代理")
    print(f"   ✓ 模型：{population[0].model}")
    print(f"   ✓ 角色：{population[0].role}")
    
    # 执行评估
    print("\n2. 执行真实LLM评估...")
    print("   正在调用Ollama API（可能需要几分钟）...")
    print("   进度：", end="", flush=True)
    
    try:
        score = simulate_task_evaluation(population)
        print(f"\n   ✓ 评估完成！")
        print(f"   ✓ 平均分：{score:.3f}")
        
        print("\n" + "="*70)
        print("测试成功！真实LLM功能正常")
        print("="*70)
        print("\n可以开始完整实验，预计时间：")
        print("  - 每组：10-20分钟")
        print("  - 6组总计：60-120分钟")
        print("\n运行命令：python pilot_experiment.py")
        
    except Exception as e:
        print(f"\n   ✗ 评估失败：{e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())