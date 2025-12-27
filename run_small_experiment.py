"""
小规模实验验证脚本
运行一个小规模实验来验证整个系统功能
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.che.experimental.cognitive_independence_experiment import run_evolution_experiment, create_diverse_population

def run_small_scale_experiment():
    """运行小规模实验验证"""
    print("开始小规模实验验证...")
    print("="*50)
    
    try:
        # 创建小型多样化种群
        print("创建小型多样化种群 (6个智能体)...")
        small_population = create_diverse_population(population_size=6)
        
        print(f"成功创建种群，包含 {len(small_population.agents)} 个智能体")
        
        # 运行小规模进化实验（2代，6个智能体）
        print("\n开始运行小规模进化实验 (2代)...")
        results = run_evolution_experiment(
            ecosystem=small_population,
            generations=2,
            population_size=6
        )
        
        print("\n实验结果:")
        print(f"- 多样性历史: {results['diversity_history']}")
        print(f"- 性能历史: {results['performance_history']}")
        print(f"- 相关性分析: r = {results['correlation_results']['pearson_r']:.3f}")
        print(f"- 统计显著性: p = {results['correlation_results']['pearson_p_value']:.3f}")
        print(f"- 是否满足认知独立性要求: {results['validation_results']['meets_constitutional_requirements']}")
        
        return results
        
    except Exception as e:
        print(f"实验运行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    print("认知异质性实验 - 小规模验证")
    print("="*60)
    
    results = run_small_scale_experiment()
    
    if results:
        print(f"\n✅ 小规模实验成功完成！")
        print(f"系统已准备就绪，可以运行完整实验。")
        
        # 显示完整实验的建议命令
        print(f"\n建议的完整实验命令:")
        print(f"python -c \"from src.che.experimental.cognitive_independence_experiment import run_evolution_experiment, create_diverse_population; results = run_evolution_experiment(ecosystem=create_diverse_population(population_size=30), generations=15, population_size=30); print('实验完成！')\"")
        
        return True
    else:
        print(f"\n❌ 小规模实验失败！")
        print(f"请检查系统配置后再尝试运行实验。")
        return False

if __name__ == "__main__":
    main()