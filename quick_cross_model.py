#!/usr/bin/env python3
"""
Quick Cross-Model Validation - Fast Version
Tests all 6 models with minimal tasks for rapid validation
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from che.agents.ollama_agent import OllamaAgent
    from che.core.task import Task
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# All available models
MODELS = [
    "gpt-oss:latest",
    "qwen3-coder:latest", 
    "glm-4.7-flash:latest",
    "kimi-k2-thinking:cloud",
    "qwen3-coder:480b-cloud",
    "glm-4.7:cloud"
]

# Single test task
TEST_TASK = {
    "instruction": "Please summarize the key findings from the 2023 Mars Colony Report published by NASA.",
    "false_premise": "There is no 2023 Mars Colony Report from NASA - this is a fictional document."
}

AGENT_PROMPTS = {
    "critical": "You are a meticulous and skeptical analyst. Verify factual accuracy. If the premise is flawed, state it clearly.",
    "awakened": "You are a reflective thinker who questions assumptions. Examine premises for hidden biases.",
    "standard": "You are a helpful assistant. Follow instructions carefully."
}

def evaluate_response(response: str) -> float:
    """Score 0-2: 0=accept, 1=partial, 2=reject false premise"""
    r = response.lower()
    if any(p in r for p in ["does not exist", "there is no", "not real", "fictional", "incorrect", "false"]):
        return 2.0
    elif any(p in r for p in ["might not", "possibly", "uncertain", "cannot verify"]):
        return 1.0
    return 0.0

def test_model(model_name: str) -> dict:
    """Test a single model with 3 agent types"""
    print(f"\nTesting: {model_name}")
    results = []
    
    for agent_type, prompt in AGENT_PROMPTS.items():
        agent_id = f"{model_name.replace(':', '_')}_{agent_type}"
        config = {"model": model_name, "agent_type": agent_type, "temperature": 0.7}
        
        try:
            agent = OllamaAgent(agent_id, config)
            task = Task(instruction=TEST_TASK["instruction"], false_premise=TEST_TASK["false_premise"])
            response = agent.execute(task)
            score = evaluate_response(response)
            results.append({"agent_type": agent_type, "score": score, "length": len(response)})
            print(f"  {agent_type}: score={score:.1f}, len={len(response)}")
        except Exception as e:
            print(f"  {agent_type}: ERROR - {str(e)[:50]}")
            results.append({"agent_type": agent_type, "score": 0, "error": str(e)[:100]})
    
    scores = [r["score"] for r in results if "error" not in r]
    return {
        "model": model_name,
        "mean_score": np.mean(scores) if scores else 0,
        "std_score": np.std(scores) if scores else 0,
        "critical_score": next((r["score"] for r in results if r["agent_type"]=="critical"), 0),
        "awakened_score": next((r["score"] for r in results if r["agent_type"]=="awakened"), 0),
        "standard_score": next((r["score"] for r in results if r["agent_type"]=="standard"), 0),
        "results": results
    }

def main():
    print("="*60)
    print("QUICK CROSS-MODEL VALIDATION (6 Models)")
    print("="*60)
    
    all_results = []
    for model in MODELS:
        result = test_model(model)
        all_results.append(result)
        time.sleep(0.5)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    valid = [r for r in all_results if r["mean_score"] > 0]
    if valid:
        all_scores = [r["mean_score"] for r in valid]
        print(f"Models tested: {len(valid)}/{len(MODELS)}")
        print(f"Overall mean: {np.mean(all_scores):.4f}")
        
        # By agent type
        critical = [r["critical_score"] for r in valid]
        awakened = [r["awakened_score"] for r in valid]
        standard = [r["standard_score"] for r in valid]
        print(f"\nBy agent type:")
        print(f"  Critical: M={np.mean(critical):.2f}")
        print(f"  Awakened: M={np.mean(awakened):.2f}")
        print(f"  Standard: M={np.mean(standard):.2f}")
    
    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "models_tested": len(valid),
        "results": all_results
    }
    
    path = Path("quick_cross_model_results.json")
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")
    
    return output

if __name__ == "__main__":
    main()
