#!/usr/bin/env python3
"""
Analyze model heterogeneity from existing data
"""

import os
import json
from pathlib import Path
from collections import Counter

def analyze_model_diversity():
    """Analyze model diversity across all experiments"""
    
    models_found = Counter()
    agent_types = Counter()
    
    exp_dirs = ['experiments', 'experiments_gemma3']
    
    for exp_dir in exp_dirs:
        if not os.path.exists(exp_dir):
            continue
        
        for f in os.listdir(exp_dir):
            if not f.endswith('.json'):
                continue
            
            try:
                with open(os.path.join(exp_dir, f), 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    
                    # Extract model
                    config = data.get('config', {})
                    model = config.get('model', 'unknown')
                    models_found[model] += 1
                    
                    # Extract agent types
                    agents = data.get('ecosystem_state', {}).get('agents', {})
                    for agent_id in agents.keys():
                        if 'critical' in agent_id.lower():
                            agent_types['critical'] += 1
                        elif 'awakened' in agent_id.lower():
                            agent_types['awakened'] += 1
                        else:
                            agent_types['standard'] += 1
                            
            except Exception as e:
                pass
    
    print("="*60)
    print("MODEL HETEROGENEITY ANALYSIS")
    print("="*60)
    
    print(f"\nModels found in experiments:")
    for model, count in models_found.most_common():
        print(f"  {model}: {count} experiments")
    
    print(f"\nAgent types distribution:")
    total = sum(agent_types.values())
    for atype, count in agent_types.most_common():
        pct = count / total * 100 if total > 0 else 0
        print(f"  {atype}: {count} ({pct:.1f}%)")
    
    # Cross-model validation results
    print(f"\nCross-Model Validation Results:")
    cross_model_files = [
        'cross_model_validation_result_20260315_091233.json',
        'cross_model_validation_result_20260315_091350.json',
        'cross_model_validation_result_20260315_103500.json'
    ]
    
    cross_model_data = []
    for f in cross_model_files:
        if os.path.exists(f):
            with open(f, 'r') as file:
                data = json.load(file)
                cross_model_data.append(data)
    
    models_validated = set()
    for data in cross_model_data:
        for result in data.get('results', []):
            model = result.get('model', 'unknown')
            models_validated.add(model)
            avg_score = result.get('average_score', 0)
            print(f"  {model}: avg_score={avg_score:.2f}")
    
    # Model diversity metrics
    print(f"\n" + "="*60)
    print("MODEL DIVERSITY METRICS")
    print("="*60)
    
    total_models = len(models_found) + len(models_validated)
    all_models = set(models_found.keys()) | models_validated
    
    print(f"Unique models tested: {len(all_models)}")
    print(f"Total experiment files: {sum(models_found.values())}")
    
    # Vendor diversity
    vendors = {
        'gpt-oss': 'OpenAI-style',
        'qwen': 'Alibaba',
        'glm': 'Zhipu',
        'gemma': 'Google',
        'kimi': 'Moonshot'
    }
    
    vendors_found = set()
    for model in all_models:
        for prefix, vendor in vendors.items():
            if prefix in model.lower():
                vendors_found.add(vendor)
    
    print(f"Vendors represented: {len(vendors_found)} ({', '.join(vendors_found)})")
    
    # Calculate model heterogeneity score
    # Using normalized Shannon entropy
    model_counts = list(models_found.values())
    if model_counts:
        import math
        total = sum(model_counts)
        probs = [c/total for c in model_counts]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        max_entropy = math.log2(len(model_counts)) if len(model_counts) > 1 else 1
        normalized = entropy / max_entropy if max_entropy > 0 else 0
        print(f"Model entropy: {entropy:.3f} (normalized: {normalized:.3f})")
    
    return {
        "models_tested": list(all_models),
        "vendors": list(vendors_found),
        "total_experiments": sum(models_found.values()),
        "agent_distribution": dict(agent_types)
    }

if __name__ == "__main__":
    analyze_model_diversity()
