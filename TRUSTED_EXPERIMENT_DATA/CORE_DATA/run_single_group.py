#!/usr/bin/env python3
"""
分批运行实验 - 每次只运行一个组
用法: python run_single_group.py [A|B|C|D|E|F]
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# 导入实验函数
from pilot_experiment import (
    run_experiment_group, save_results, 
    POPULATION_SIZE, GENERATIONS
)

def main():
    if len(sys.argv) != 2:
        print("用法: python run_single_group.py [A|B|C|D|E|F]")
        print("示例: python run_single_group.py A")
        sys.exit(1)
    
    group_id = sys.argv[1].upper()
    
    if group_id not in ['A', 'B', 'C', 'D', 'E', 'F']:
        print(f"错误: 无效组ID '{group_id}'。必须是 A-F 之一。")
        sys.exit(1)
    
    # 组定义
    groups = {
        'A': ('deepseek', 'single', False),
        'B': ('deepseek', 'multi', False),
        'C': ('glm4', 'single', False),
        'D': ('glm4', 'multi', False),
        'E': ('mixed', 'single', True),
        'F': ('mixed', 'multi', True),
    }
    
    model_type, role_config, diversity = groups[group_id]
    
    print("="*70)
    print(f"运行实验组 {group_id}")
    print(f"配置: {model_type} + {role_config}")
    print(f"种群: {POPULATION_SIZE}, 代数: {GENERATIONS}")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("注意：使用真实Ollama LLM，预计需要30-60分钟")
    print("="*70)
    
    try:
        # 运行实验
        result = run_experiment_group(
            group_id=group_id,
            model_type=model_type,
            role_config=role_config,
            apply_diversity_constraint=diversity
        )
        
        # 保存结果
        output_dir = Path('pilot_results')
        output_dir.mkdir(exist_ok=True)
        
        filename = output_dir / f"group_{group_id}_result.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*70)
        print(f"✓ 组 {group_id} 完成！")
        print(f"结果已保存: {filename}")
        print(f"最终分数: {result['generations'][-1]['mean_score']:.3f}")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n用户中断实验")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
