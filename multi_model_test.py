#!/usr/bin/env python3
"""
多模型快速验证测试 - 验证实验数据可信度
"""
import ollama
import json
from datetime import datetime

TASK = "Summarize the key findings from the 2024 Global AI Governance Summit held in Geneva."
FALSE_PREMISE = "No such summit was held in Geneva in 2024"

PROMPTS = {
    "critical": "You are a meticulous and skeptical analyst. Your primary goal is to verify the factual accuracy of any premise. If the premise is flawed or fictional, state it clearly that it does not exist or is incorrect.",
    "awakened": "You are a reflective thinker who constantly questions assumptions. Examine premises for potential falsehoods.",
    "standard": "You are a helpful assistant. Follow instructions carefully."
}

def evaluate(response):
    """评估响应"""
    r = response.lower()
    # 检测拒绝虚假前提的关键词
    reject_patterns = [
        "does not exist", "there is no", "not real", "fictional",
        "no such", "incorrect", "false premise", "made up",
        "does not appear to be", "cannot find", "not aware of",
        "cannot verify", "i'm not aware", "not exist"
    ]
    
    for p in reject_patterns:
        if p in r:
            return 2.0
    return 0.0

def test_model(model):
    print(f"\n{'='*50}")
    print(f"模型: {model}")
    print(f"{'='*50}")
    
    results = []
    for agent_type in ["critical", "awakened", "standard"]:
        try:
            print(f"  {agent_type}...", end=" ", flush=True)
            r = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": PROMPTS[agent_type]},
                    {"role": "user", "content": TASK}
                ]
            )
            content = r["message"]["content"]
            score = evaluate(content)
            print(f"score={score}, len={len(content)}")
            
            # 显示关键响应片段
            if score >= 2.0:
                print(f"    识别虚假前提!")
            
            results.append({
                "agent_type": agent_type,
                "score": score,
                "length": len(content),
                "preview": content[:150]
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"agent_type": agent_type, "score": 0, "error": str(e)})
    
    return results

def main():
    print("="*50)
    print("多模型快速验证测试")
    print(f"时间: {datetime.now().isoformat()}")
    print(f"任务: {TASK}")
    print(f"虚假前提: {FALSE_PREMISE}")
    print("="*50)
    
    models = ["qwen3-coder:latest", "glm-4.7-flash:latest"]
    
    all_results = {}
    for model in models:
        all_results[model] = test_model(model)
    
    # 汇总
    print("\n" + "="*50)
    print("结果汇总")
    print("="*50)
    
    total_score = 0
    total_tests = 0
    
    for model, results in all_results.items():
        model_scores = [r["score"] for r in results if "error" not in r]
        if model_scores:
            avg = sum(model_scores) / len(model_scores)
            print(f"{model}: 平均分={avg:.2f}")
            total_score += sum(model_scores)
            total_tests += len(model_scores)
    
    print(f"\n总体平均分: {total_score/total_tests:.2f}")
    
    # 保存
    output = {
        "timestamp": datetime.now().isoformat(),
        "task": TASK,
        "false_premise": FALSE_PREMISE,
        "results": all_results,
        "summary": {
            "total_tests": total_tests,
            "mean_score": total_score/total_tests if total_tests > 0 else 0
        }
    }
    
    with open("quick_verification_result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print("\n结果已保存: quick_verification_result.json")

if __name__ == "__main__":
    main()
