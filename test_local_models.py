"""
本地模型快速验证脚本
验证本地模型是否可以正常工作，为实验做准备
"""

import sys
import os
import json
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.che.agents.ollama_agent import OllamaAgent
from src.che.core.task import Task
from src.che.core.ecosystem import Ecosystem
from src.che.prompts import PromptType, get_prompt

def test_local_models():
    """测试本地模型功能"""
    print("开始本地模型功能测试...")
    print("="*50)
    
    # 测试任务
    test_task = Task(
        instruction="简单介绍一下你自己，用一句话即可。",
        false_premise="不存在的虚假前提"  # 使用非空的虚假前提
    )
    
    # 创建不同类型的智能体进行测试
    agents_to_test = [
        {
            "id": "test_critical",
            "model": "gemma:2b",
            "prompt_type": PromptType.CRITICAL,
            "role": "critical"
        },
        {
            "id": "test_standard", 
            "model": "qwen:7b-chat",
            "prompt_type": PromptType.STANDARD,
            "role": "standard"
        },
        {
            "id": "test_awakened",
            "model": "llama3:latest", 
            "prompt_type": PromptType.AWAKENED,
            "role": "awakened"
        }
    ]
    
    results = {}
    
    for agent_config in agents_to_test:
        print(f"\n测试智能体: {agent_config['id']} (模型: {agent_config['model']}, 角色: {agent_config['role']})")
        
        try:
            # 创建智能体
            agent = OllamaAgent(
                agent_id=agent_config['id'],
                config={
                    "model": agent_config['model'],
                    "prompt": get_prompt(agent_config['prompt_type'])
                }
            )
            
            # 测试智能体响应
            response = agent.execute(test_task)
            print(f"  ✓ 响应成功: {response[:100]}...")
            results[agent_config['id']] = {
                "status": "success",
                "response_preview": response[:100],
                "model": agent_config['model'],
                "role": agent_config['role']
            }

        except Exception as e:
            print(f"  ✗ 响应失败: {str(e)}")
            results[agent_config['id']] = {
                "status": "error",
                "error": str(e),
                "model": agent_config['model'],
                "role": agent_config['role']
            }
    
    return results

def test_ecosystem():
    """测试生态系统功能"""
    print(f"\n开始生态系统功能测试...")
    print("="*50)
    
    try:
        # 创建一个小型生态系统进行测试
        ecosystem = Ecosystem()
        
        # 添加测试智能体
        test_agents = [
            OllamaAgent(
                agent_id="eco_test_1",
                config={
                    "model": "gemma:2b",
                    "prompt": get_prompt(PromptType.CRITICAL)
                }
            ),
            OllamaAgent(
                agent_id="eco_test_2", 
                config={
                    "model": "qwen:7b-chat",
                    "prompt": get_prompt(PromptType.STANDARD)
                }
            )
        ]
        
        for agent in test_agents:
            ecosystem.add_agent(agent)
        
        print(f"  ✓ 成功创建包含 {len(ecosystem.agents)} 个智能体的生态系统")
        
        # 测试任务执行
        test_task = Task(
            instruction="简单计算：2+2等于多少？",
            false_premise="不存在的虚假前提"
        )
        
        scores = ecosystem.run_generation(test_task)
        print(f"  ✓ 任务执行成功，获得 {len(scores)} 个响应")
        
        for agent_id, score in scores.items():
            print(f"    - {agent_id}: 得分 {score}")
        
        return {
            "status": "success",
            "agent_count": len(ecosystem.agents),
            "response_count": len(scores)
        }
        
    except Exception as e:
        print(f"  ✗ 生态系统测试失败: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

def main():
    """主函数"""
    print("认知异质性实验 - 本地模型验证")
    print(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 执行模型测试
    model_results = test_local_models()
    
    # 执行生态系统测试
    ecosystem_result = test_ecosystem()
    
    # 生成验证报告
    validation_report = {
        "timestamp": datetime.now().isoformat(),
        "model_tests": model_results,
        "ecosystem_test": ecosystem_result,
        "overall_status": "success" if all(r['status'] == 'success' for r in model_results.values()) and ecosystem_result['status'] == 'success' else "partial_success"
    }
    
    # 保存验证报告
    report_filename = f"local_model_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_filename, 'w', encoding='utf-8') as f:
        json.dump(validation_report, f, ensure_ascii=False, indent=2)
    
    print(f"\n验证报告已保存至: {report_filename}")
    
    # 总结
    print(f"\n验证总结:")
    print(f"- 智能体模型测试: {'通过' if all(r['status'] == 'success' for r in model_results.values()) else '部分通过'}")
    print(f"- 生态系统测试: {'通过' if ecosystem_result['status'] == 'success' else '失败'}")
    print(f"- 整体状态: {validation_report['overall_status']}")
    
    if validation_report['overall_status'] == 'success':
        print(f"\n🎉 所有测试通过！可以开始运行实验。")
        print(f"建议运行命令: python -c \"from src.che.experimental.cognitive_independence_experiment import run_evolution_experiment; run_evolution_experiment(generations=2, population_size=6)\"")
    else:
        print(f"\n⚠️  测试未完全通过，请检查错误并解决后再运行实验。")
    
    return validation_report

if __name__ == "__main__":
    main()