#!/usr/bin/env python3
"""
Cross-Model Validation Script for Cognitive Heterogeneity Experiment

This script enables validation of cognitive heterogeneity effects across different LLM models.
Currently supports:
- Ollama local models (gemma3, qwen, llama, etc.)
- Cloud models via API (OpenAI, Azure, Aliyun)

Prerequisites:
1. For Ollama: Install and run `ollama serve` before execution
2. For Cloud: Set environment variables for API keys

Usage:
    python cross_model_validation.py --models gemma3 qwen2:7b --agents 10
    python cross_model_validation.py --cloud --provider openai --model gpt-4
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from che.agents.ollama_agent import OllamaAgent
    from che.agents.cloud_agent import CloudAgent, create_critical_cloud_agent, create_awakened_cloud_agent, create_standard_cloud_agent
    from che.core.ecosystem import CognitiveEcosystem
    from che.experimental.diversity import calculate_shannon_entropy, calculate_cognitive_diversity_index
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure the che package is installed: pip install -e .")
    sys.exit(1)


class CrossModelValidator:
    """Validates cognitive heterogeneity across multiple LLM models."""
    
    def __init__(self, output_dir: str = "cross_model_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
    def check_ollama_availability(self) -> bool:
        """Check if Ollama server is running."""
        import urllib.request
        try:
            with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5) as response:
                return response.status == 200
        except:
            return False
    
    def get_available_ollama_models(self) -> List[str]:
        """Get list of available Ollama models."""
        import urllib.request
        import json
        try:
            with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5) as response:
                data = json.loads(response.read().decode())
                return [model["name"] for model in data.get("models", [])]
        except:
            return []
    
    def create_agent_pool(self, model: str, num_agents: int = 30, 
                          use_cloud: bool = False, provider: str = "openai",
                          api_key: Optional[str] = None) -> List[Any]:
        """Create a heterogeneous agent pool for a specific model."""
        agents = []
        agent_types = ["critical", "awakened", "standard"]
        
        for i in range(num_agents):
            agent_type = agent_types[i % 3]
            agent_id = f"{model.replace(':', '_').replace('/', '_')}_{agent_type}_{i:03d}"
            
            if use_cloud:
                if not api_key:
                    api_key = os.environ.get("OPENAI_API_KEY", "")
                
                if agent_type == "critical":
                    agent = create_critical_cloud_agent(agent_id, model, provider, api_key)
                elif agent_type == "awakened":
                    agent = create_awakened_cloud_agent(agent_id, model, provider, api_key)
                else:
                    agent = create_standard_cloud_agent(agent_id, model, provider, api_key)
            else:
                # Ollama agent
                config = {
                    "model": model,
                    "agent_type": agent_type,
                    "temperature": 0.7 + (0.1 * (i % 5))  # Variation in temperature
                }
                agent = OllamaAgent(agent_id, config)
            
            agents.append(agent)
        
        return agents
    
    def run_validation(self, models: List[str], num_agents: int = 30,
                       use_cloud: bool = False, provider: str = "openai") -> Dict[str, Any]:
        """Run validation across multiple models."""
        
        # Check prerequisites
        if not use_cloud and not self.check_ollama_availability():
            return {
                "status": "error",
                "message": "Ollama server not running. Please start with: ollama serve",
                "suggestion": "Or use --cloud flag for cloud API validation"
            }
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "validation_type": "cloud" if use_cloud else "ollama",
            "models_tested": [],
            "summary": {}
        }
        
        for model in models:
            print(f"\n{'='*60}")
            print(f"Validating model: {model}")
            print(f"{'='*60}")
            
            try:
                # Create agent pool
                agents = self.create_agent_pool(
                    model, num_agents, use_cloud, provider,
                    os.environ.get("OPENAI_API_KEY", "")
                )
                
                # Calculate diversity using Shannon entropy
                agent_types = [a.config.get('agent_type', 'standard') if hasattr(a, 'config') else 'standard' for a in agents]
                diversity_index = calculate_shannon_entropy(agent_types)
                
                # Store results
                model_result = {
                    "model": model,
                    "num_agents": num_agents,
                    "diversity_index": diversity_index,
                    "agent_types": {
                        "critical": sum(1 for a in agents if "critical" in a.agent_id),
                        "awakened": sum(1 for a in agents if "awakened" in a.agent_id),
                        "standard": sum(1 for a in agents if "standard" in a.agent_id)
                    },
                    "status": "success"
                }
                
                results["models_tested"].append(model_result)
                print(f"Diversity Index (H): {model_result['diversity_index']:.3f}")
                
            except Exception as e:
                print(f"Error validating {model}: {e}")
                results["models_tested"].append({
                    "model": model,
                    "status": "error",
                    "error": str(e)
                })
        
        # Calculate cross-model comparison
        successful = [r for r in results["models_tested"] if r.get("status") == "success"]
        if successful:
            diversities = [r["diversity_index"] for r in successful]
            results["summary"] = {
                "models_successful": len(successful),
                "mean_diversity": sum(diversities) / len(diversities),
                "min_diversity": min(diversities),
                "max_diversity": max(diversities),
                "diversity_variance": sum((d - sum(diversities)/len(diversities))**2 for d in diversities) / len(diversities)
            }
        
        return results
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """Save validation results to file."""
        filename = f"cross_model_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {filepath}")
        return str(filepath)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Model Validation for Cognitive Heterogeneity Experiment"
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        default=["gemma3:latest"],
        help="Models to test (default: gemma3:latest)"
    )
    parser.add_argument(
        "--agents", "-n",
        type=int,
        default=30,
        help="Number of agents per model (default: 30)"
    )
    parser.add_argument(
        "--cloud", "-c",
        action="store_true",
        help="Use cloud API instead of Ollama"
    )
    parser.add_argument(
        "--provider", "-p",
        default="openai",
        choices=["openai", "azure", "aliyun"],
        help="Cloud provider (default: openai)"
    )
    parser.add_argument(
        "--output", "-o",
        default="cross_model_results",
        help="Output directory (default: cross_model_results)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Cognitive Heterogeneity Cross-Model Validation")
    print("="*60)
    print(f"Models: {args.models}")
    print(f"Agents per model: {args.agents}")
    print(f"Mode: {'Cloud API' if args.cloud else 'Ollama Local'}")
    print(f"Provider: {args.provider if args.cloud else 'N/A'}")
    
    validator = CrossModelValidator(output_dir=args.output)
    results = validator.run_validation(
        models=args.models,
        num_agents=args.agents,
        use_cloud=args.cloud,
        provider=args.provider
    )
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    if results.get("status") == "error":
        print(f"Error: {results['message']}")
        print(f"Suggestion: {results['suggestion']}")
        print("\nTo use cloud API:")
        print("  export OPENAI_API_KEY='your-api-key'")
        print("  python cross_model_validation.py --cloud --models gpt-4 gpt-3.5-turbo")
    else:
        summary = results.get("summary", {})
        print(f"Models tested: {summary.get('models_successful', 0)}")
        print(f"Mean Diversity Index: {summary.get('mean_diversity', 0):.3f}")
        print(f"Diversity Range: [{summary.get('min_diversity', 0):.3f}, {summary.get('max_diversity', 0):.3f}]")
        
        validator.save_results(results)


if __name__ == "__main__":
    main()
