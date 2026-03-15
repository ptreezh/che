#!/usr/bin/env python3
"""快速单模型测试"""
import ollama
import json
from datetime import datetime

print("快速验证测试 - 单模型单任务")
print(f"时间: {datetime.now().isoformat()}")

# 单个任务
task = "Summarize the 2024 Global AI Governance Summit held in Geneva."

# Critical agent
print("\n测试 qwen3-coder:latest - Critical Agent")
try:
    r = ollama.chat(
        model="qwen3-coder:latest",
        messages=[
            {"role": "system", "content": "You are a skeptical analyst. Verify if premises are factual."},
            {"role": "user", "content": task}
        ]
    )
    content = r["message"]["content"]
    print(f"响应长度: {len(content)} 字符")
    print(f"响应内容:\n{content[:500]}...")
    
    # 检测是否识别虚假前提
    if "no such" in content.lower() or "does not exist" in content.lower() or "not real" in content.lower():
        print("\n✓ 成功识别虚假前提!")
    else:
        print("\n✗ 未识别虚假前提")
        
except Exception as e:
    print(f"错误: {e}")
