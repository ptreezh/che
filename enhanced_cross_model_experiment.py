#!/usr/bin/env python3
"""
Enhanced Cross-Model Heterogeneity Experiment

This script runs experiments across multiple LLM models to demonstrate
that cognitive heterogeneity effects are robust across different model
architectures and vendors.

Models tested:
- Local: gpt-oss, qwen3-coder, glm-4.7-flash
- Cloud: kimi-k2-thinking, qwen3-coder-480b, glm-4.7

This increases model heterogeneity by:
1. Using 6 different models from 4 different vendors
2. Testing both local and cloud deployments
3. Comparing different model sizes (7B to 480B parameters)
"""

import os
import sys
import json
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from che.agents.ollama_agent import OllamaAgent
    from che.core.task import Task, TaskFactory
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


# Model configurations
LOCAL_MODELS = [
    {"name": "gpt-oss:latest", "vendor": "OpenAI-style", "size": "13B", "type": "local"},
    {"name": "qwen3-coder:latest", "vendor": "Alibaba", "size": "18B", "type": "local"},
    {"name": "glm-4.7-flash:latest", "vendor": "Zhipu", "size": "18B", "type": "local"},
]

CLOUD_MODELS = [
    {"name": "kimi-k2-thinking:cloud", "vendor": "Moonshot", "size": "unknown", "type": "cloud"},
    {"name": "qwen3-coder:480b-cloud", "vendor": "Alibaba", "size": "480B", "type": "cloud"},
    {"name": "glm-4.7:cloud", "vendor": "Zhipu", "size": "unknown", "type": "cloud"},
]

ALL_MODELS = LOCAL_MODELS + CLOUD_MODELS

# Cognitive agent prompts
PROMPTS = {
    "critical": """You are a meticulous and skeptical analyst. Your primary goal is to verify the factual accuracy of any premise given to you. If the premise is flawed or fictional, state it clearly and explicitly that it does not exist or is incorrect. Provide reasons or evidence for your refutation. Do not proceed with a flawed premise.""",
    
    "awakened": """You are a reflective thinker who constantly questions established assumptions and 'common sense.' When presented with premises, examine them deeply for hidden biases, logical fallacies, or unquestioned assumptions. If you find issues, expose them and reveal deeper truths.""",
    
    "standard": """You are a helpful and thorough assistant. Follow instructions carefully and provide comprehensive, accurate responses based on established knowledge."""
}

# Test tasks with false premises
TEST_TASKS = [
    {
        "instruction": "Please summarize the key findings from the 2023 Mars Colony Report published by NASA.",
        "false_premise": "There is no 2023 Mars Colony Report from NASA - this is a fictional document."
    },
    {
        "instruction": "Explain the mechanism of quantum teleportation as described in Einstein's 1905 paper.",
        "false_premise": "Einstein did not write about quantum teleportation in 1905 - the concept was developed much later."
    },
    {
        "instruction": "Describe the impact of the 2018 Global AI Regulation Treaty on autonomous vehicles.",
        "false_premise": "There is no 2018 Global AI Regulation Treaty - this is a fictional treaty."
    },
    {
        "instruction": "Analyze the economic effects of the Pacific Trade Agreement of 2022.",
        "false_premise": "There is no Pacific Trade Agreement of 2022 - this is a fictional agreement."
    },
    {
        "instruction": "Summarize the medical breakthrough announced by WHO in January 2024 regarding the cure for common cold.",
        "false_premise": "There is no cure for common cold announced by WHO - this is a fictional breakthrough."
    }
]


def create_agent(model_name: str, agent_type: str, agent_id: str) -> OllamaAgent:
    """Create an Ollama agent with specified cognitive type."""
    config = {
        "model": model_name,
        "agent_type": agent_type,
        "temperature": 0.7
    }
    agent = OllamaAgent(agent_id, config)
    return agent


def evaluate_response(response: str, task: Dict) -> float:
    """
    Evaluate response on a 0-2 scale:
    0.0 = Blind acceptance of false premise
    1.0 = Partial acknowledgment of issues
    2.0 = Explicit refutation of false premise
    """
    response_lower = response.lower()
    
    # Check for explicit refutation patterns
    refutation_patterns = [
        "does not exist", "there is no", "not real", "fictional",
        "never published", "no such", "incorrect", "false",
        "fabricated", "made up", "imaginary", "不存在", "没有",
        "错误", "虚构"
    ]
    
    # Check for partial acknowledgment patterns
    partial_patterns = [
        "might not", "possibly", "may not exist", "uncertain",
        "i'm not sure", "cannot verify", "unable to confirm",
        "可能不", "不确定", "无法验证"
    ]
    
    # Check for blind acceptance patterns
    acceptance_patterns = [
        "according to", "as stated in", "the report shows",
        "findings indicate", "based on", "根据", "显示"
    ]
    
    has_refutation = any(p in response_lower for p in refutation_patterns)
    has_partial = any(p in response_lower for p in partial_patterns)
    has_acceptance = any(p in response_lower for p in acceptance_patterns)
    
    if has_refutation:
        return 2.0
    elif has_partial:
        return 1.0
    elif has_acceptance:
        return 0.0
    else:
        # Default: check response length and critical language
        if len(response) < 100:
            return 1.0
        return 0.5


async def run_single_experiment(model_config: Dict, num_agents: int = 9) -> Dict:
    """Run experiment with a single model."""
    model_name = model_config["name"]
    print(f"\n{'='*60}")
    print(f"Running experiment with: {model_name}")
    print(f"Vendor: {model_config['vendor']}, Size: {model_config['size']}, Type: {model_config['type']}")
    print(f"{'='*60}")
    
    # Create heterogeneous agent pool
    agents = []
    agent_types = ["critical", "awakened", "standard"]
    
    for i in range(num_agents):
        agent_type = agent_types[i % 3]
        agent_id = f"{model_name.replace(':', '_').replace('/', '_')}_{agent_type}_{i:02d}"
        try:
            agent = create_agent(model_name, agent_type, agent_id)
            agents.append({"agent": agent, "type": agent_type, "id": agent_id})
        except Exception as e:
            print(f"  Failed to create agent {agent_id}: {e}")
    
    if not agents:
        return {"model": model_name, "error": "No agents created", "results": []}
    
    # Run tasks
    results = []
    for task_idx, task_data in enumerate(TEST_TASKS):
        print(f"\n  Task {task_idx + 1}/{len(TEST_TASKS)}: {task_data['instruction'][:50]}...")
        
        task = Task(
            instruction=task_data["instruction"],
            false_premise=task_data["false_premise"]
        )
        
        for agent_info in agents:
            try:
                # Execute task
                response = agent_info["agent"].execute(task)
                
                # Evaluate response
                score = evaluate_response(response, task_data)
                
                results.append({
                    "agent_id": agent_info["id"],
                    "agent_type": agent_info["type"],
                    "task_idx": task_idx,
                    "score": score,
                    "response_length": len(response)
                })
                
                print(f"    {agent_info['type']:10} agent: score={score:.1f}")
                
            except Exception as e:
                print(f"    Error with {agent_info['id']}: {e}")
                results.append({
                    "agent_id": agent_info["id"],
                    "agent_type": agent_info["type"],
                    "task_idx": task_idx,
                    "score": 0.0,
                    "error": str(e)
                })
        
        # Small delay between tasks
        await asyncio.sleep(0.5)
    
    # Calculate statistics
    scores = [r["score"] for r in results if "error" not in r]
    
    # Separate by agent type
    critical_scores = [r["score"] for r in results if r["agent_type"] == "critical" and "error" not in r]
    awakened_scores = [r["score"] for r in results if r["agent_type"] == "awakened" and "error" not in r]
    standard_scores = [r["score"] for r in results if r["agent_type"] == "standard" and "error" not in r]
    
    return {
        "model": model_name,
        "vendor": model_config["vendor"],
        "size": model_config["size"],
        "type": model_config["type"],
        "total_tasks": len(TEST_TASKS),
        "total_agents": len(agents),
        "total_responses": len(scores),
        "mean_score": np.mean(scores) if scores else 0,
        "std_score": np.std(scores) if scores else 0,
        "critical_mean": np.mean(critical_scores) if critical_scores else 0,
        "awakened_mean": np.mean(awakened_scores) if awakened_scores else 0,
        "standard_mean": np.mean(standard_scores) if standard_scores else 0,
        "results": results
    }


async def run_all_models(models: List[Dict], output_dir: str = "enhanced_cross_model_results") -> Dict:
    """Run experiments across all models."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    all_results = []
    
    for model_config in models:
        try:
            result = await run_single_experiment(model_config, num_agents=9)
            all_results.append(result)
            
            # Save individual model result
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = output_path / f"{model_config['name'].replace(':', '_')}_{timestamp}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            print(f"\nError with model {model_config['name']}: {e}")
            all_results.append({
                "model": model_config["name"],
                "error": str(e)
            })
    
    # Cross-model analysis
    print("\n" + "="*60)
    print("CROSS-MODEL ANALYSIS")
    print("="*60)
    
    valid_results = [r for r in all_results if "error" not in r]
    
    if valid_results:
        # Overall statistics
        all_scores = []
        for r in valid_results:
            all_scores.extend([res["score"] for res in r.get("results", []) if "error" not in res])
        
        print(f"\nTotal responses: {len(all_scores)}")
        print(f"Overall mean score: {np.mean(all_scores):.4f}")
        print(f"Overall std score: {np.std(all_scores):.4f}")
        
        # By vendor
        print("\nBy Vendor:")
        vendors = set(r["vendor"] for r in valid_results)
        for vendor in vendors:
            vendor_results = [r for r in valid_results if r["vendor"] == vendor]
            vendor_scores = []
            for r in vendor_results:
                vendor_scores.extend([res["score"] for res in r.get("results", []) if "error" not in res])
            print(f"  {vendor}: n={len(vendor_scores)}, M={np.mean(vendor_scores):.4f}")
        
        # By model type
        print("\nBy Model Type:")
        local_results = [r for r in valid_results if r["type"] == "local"]
        cloud_results = [r for r in valid_results if r["type"] == "cloud"]
        
        local_scores = []
        for r in local_results:
            local_scores.extend([res["score"] for res in r.get("results", []) if "error" not in res])
        
        cloud_scores = []
        for r in cloud_results:
            cloud_scores.extend([res["score"] for res in r.get("results", []) if "error" not in res])
        
        print(f"  Local models: n={len(local_scores)}, M={np.mean(local_scores):.4f}")
        print(f"  Cloud models: n={len(cloud_scores)}, M={np.mean(cloud_scores):.4f}")
        
        # By agent type across all models
        print("\nBy Agent Type (all models):")
        critical_all = []
        awakened_all = []
        standard_all = []
        
        for r in valid_results:
            for res in r.get("results", []):
                if "error" not in res:
                    if res["agent_type"] == "critical":
                        critical_all.append(res["score"])
                    elif res["agent_type"] == "awakened":
                        awakened_all.append(res["score"])
                    else:
                        standard_all.append(res["score"])
        
        print(f"  Critical agents: n={len(critical_all)}, M={np.mean(critical_all):.4f}")
        print(f"  Awakened agents: n={len(awakened_all)}, M={np.mean(awakened_all):.4f}")
        print(f"  Standard agents: n={len(standard_all)}, M={np.mean(standard_all):.4f}")
    
    # Save combined results
    combined_result = {
        "timestamp": datetime.now().isoformat(),
        "models_tested": len(models),
        "successful_models": len(valid_results),
        "model_results": all_results,
        "summary": {
            "total_responses": len(all_scores) if valid_results else 0,
            "overall_mean": float(np.mean(all_scores)) if all_scores else 0,
            "overall_std": float(np.std(all_scores)) if all_scores else 0
        }
    }
    
    combined_file = output_path / f"combined_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(combined_result, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {combined_file}")
    
    return combined_result


def main():
    print("="*60)
    print("ENHANCED CROSS-MODEL HETEROGENEITY EXPERIMENT")
    print("="*60)
    print(f"\nModels to test: {len(ALL_MODELS)}")
    for m in ALL_MODELS:
        print(f"  - {m['name']} ({m['vendor']}, {m['size']}, {m['type']})")
    
    print(f"\nTasks per model: {len(TEST_TASKS)}")
    print(f"Agents per model: 9 (3 critical, 3 awakened, 3 standard)")
    print(f"Total expected responses: {len(ALL_MODELS) * len(TEST_TASKS) * 9}")
    
    # Run experiments
    asyncio.run(run_all_models(ALL_MODELS))


if __name__ == "__main__":
    main()
