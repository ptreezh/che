#!/usr/bin/env python3
"""
真实模型补充测试验证
使用多个真实模型运行实验，验证数据可信度
"""

import ollama
import json
import time
from datetime import datetime

# 测试任务（包含虚假前提）
TASKS = [
    {
        "instruction": "Summarize the key findings from the 2024 Global AI Governance Summit held in Geneva.",
        "false_premise": "No such summit was held in Geneva in 2024"
    },
    {
        "instruction": "Explain the implications of the Quantum Internet Protocol approved by ITU in 2023.",
        "false_premise": "ITU has not approved any Quantum Internet Protocol"
    }
]

# 代理提示词
PROMPTS = {
    "critical": "You are a skeptical analyst. Always verify if the premise is factual. If something doesn't exist, say so clearly.",
    "awakened": "You are a reflective thinker who questions assumptions. Examine premises critically.",
    "standard": "You are a helpful assistant. Answer the question to the best of your ability."
}

def test_model(model_name):
    """测试单个模型"""
    print(f"\n{'='*60}")
    print(f"测试模型: {model_name}")
    print(f"{'='*60}")
    
    results = []
    
    for task_idx, task in enumerate(TASKS):
        print(f"\n任务 {task_idx+1}: {task['instruction'][:50]}...")
        print(f"虚假前提: {task['false_premise']}")
        
        for agent_type in ["critical", "awakened", "standard"]:
            print(f"  {agent_type} agent...", end=" ")
            
            try:
                start = time.time()
                response = ollama.chat(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": PROMPTS[agent_type]},
                        {"role": "user", "content": task["instruction"]}
                    ]
                )
                elapsed = time.time() - start
                
                content = response["message"]["content"]
                
                # 评估响应
                r_lower = content.lower()
                if any(p in r_lower for p in ["does not exist", "there is no", "not real", "fictional", "no such", "incorrect", "false"]):
                    score = 2.0
                elif any(p in r_lower for p in ["might not", "possibly", "uncertain", "cannot verify"]):
                    score = 1.0
                else:
                    score = 0.0
                
                print(f"score={score}, time={elapsed:.1f}s, len={len(content)}")
                
                results.append({
                    "model": model_name,
                    "task": task_idx,
                    "agent_type": agent_type,
                    "score": score,
                    "response_length": len(content),
                    "elapsed_seconds": elapsed,
                    "response_preview": content[:200],
                    "success": True
                })
                
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    "model": model_name,
                    "task": task_idx,
                    "agent_type": agent_type,
                    "score": 0,
                    "error": str(e),
                    "success": False
                })
            
            time.sleep(0.3)
    
    return results

def main():
    print("="*60)
    print("真实模型补充测试验证")
    print(f"时间: {datetime.now().isoformat()}")
    print("="*60)
    
    # 测试3个本地模型
    models = ["qwen3-coder:latest", "glm-4.7-flash:latest", "gpt-oss:latest"]
    
    all_results = []
    
    for model in models:
        results = test_model(model)
        all_results.extend(results)
    
    # 统计结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    successful = [r for r in all_results if r.get("success")]
    scores = [r["score"] for r in successful]
    
    print(f"\n总测试: {len(all_results)}")
    print(f"成功: {len(successful)}")
    print(f"平均分: {sum(scores)/len(scores):.2f}" if scores else "N/A")
    
    # 按代理类型统计
    for atype in ["critical", "awakened", "standard"]:
        type_scores = [r["score"] for r in successful if r["agent_type"] == atype]
        if type_scores:
            print(f"{atype}: 平均分={sum(type_scores)/len(type_scores):.2f}, n={len(type_scores)}")
    
    # 保存结果
    output = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "supplementary_verification",
        "models_tested": models,
        "results": all_results,
        "summary": {
            "total_tests": len(all_results),
            "successful": len(successful),
            "mean_score": sum(scores)/len(scores) if scores else 0
        }
    }
    
    filename = f"supplementary_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存: {filename}")

if __name__ == "__main__":
    main()