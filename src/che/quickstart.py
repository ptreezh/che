#!/usr/bin/env python3
"""
Quick Start Script for Cognitive Heterogeneity Validation

This script provides a quick way to run the cognitive heterogeneity validation experiment
with default parameters for immediate testing and demonstration.

Authors: CHE Research Team
Date: 2025-10-19
"""

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from che.main import main as che_main

def quick_start():
    """Run the cognitive heterogeneity validation experiment with default parameters."""
    print("🚀 Quick Start: Cognitive Heterogeneity Validation")
    print("=" * 50)
    print("Running experiment with default parameters:")
    print("  - Population size: 30 agents")
    print("  - Generations: 15")
    print("  - Model: qwen:0.5b")
    print("  - Verbose logging enabled")
    print("=" * 50)
    
    # Run with default arguments
    args = [
        "--population-size", "30",
        "--generations", "15", 
        "--model", "qwen:0.5b",
        "--verbose"
    ]
    
    # Call main function with arguments
    sys.argv = ["che_quickstart.py"] + args
    return che_main()

if __name__ == "__main__":
    exit_code = quick_start()
    sys.exit(exit_code)