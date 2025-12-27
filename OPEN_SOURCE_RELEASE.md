# Cognitive Heterogeneity Validation Project - Open Source Release

## Project Overview

The Cognitive Heterogeneity Validation Project is a research initiative that investigates the effectiveness of cognitive diversity in suppressing hallucinations in multi-agent systems. Through large-scale evolutionary optimization experiments, it validates that heterogeneous agent systems significantly outperform homogeneous systems in detecting false premises.

## Key Features

### 1. Heterogeneous Agent System
- **Critical Agents**: Meticulous and skeptical analysts who verify factual accuracy
- **Awakened Agents**: Independent thinkers who constantly question and "betray" ingrained common sense
- **Standard Agents**: Helpful assistants who follow instructions directly

### 2. Hallucination Detection
- **3-Tier Evaluation System**: Scores responses from 0.0 (blind acceptance) to 2.0 (explicit refutation)
- **False Premise Embedding**: Tasks contain embedded false premises for detection
- **Statistical Validation**: Measures performance with p < 0.05 significance

### 3. Evolutionary Optimization
- **Selection Mechanism**: Removes lowest-scoring agents, replicates highest-scoring agents
- **Mutation Mechanism**: 30% chance of type mutation during replication
- **Diversity Maintenance**: Preserves cognitive heterogeneity across generations

### 4. Cognitive Independence Validation
- **Correlation Analysis**: Verifies r ≥ 0.6 between diversity and performance
- **Statistical Significance**: Ensures p < 0.01 for all findings
- **Effect Size Measurement**: Calculates Cohen's d ≥ 0.5 for meaningful differences

## Repository Structure

```
che_project/
├── config/                 # Configuration files (role definitions, experiment configs, etc.)
├── docs/                   # Documentation
├── results/                # Experimental results output directory
├── scripts/                # Experiment scripts
├── src/                    # Source code
│   └── che/                # Cognitive Heterogeneity Ecosystem
│       ├── agents/         # Agent implementations
│       ├── core/           # Core components (Agent, Task, Ecosystem)
│       ├── evaluation/     # Evaluation components (Evaluator interface)
│       ├── experimental/   # Experimental components (patterns, distinctiveness, awakening, comparison)
│       └── utils/          # Utility components (logging, config, checkpoint)
├── tests/                  # Test suites
└── requirements.txt        # Python dependencies
```

## Getting Started

### Prerequisites

1. **Python 3.8+**: Required for running the experiment code
2. **Ollama**: For local LLM access (install from https://ollama.ai)
3. **Required Models**: At minimum, install one of the following: `qwen:0.5b`, `gemma:2b`
4. **System Requirements**: 8GB+ RAM for optimal performance

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd che_project

# Install dependencies using poetry (recommended)
poetry install

# Or install dependencies directly with pip
pip install -r requirements.txt
```

### Quick Start

```bash
# Start Ollama service
ollama serve

# Run installation test to verify setup
python test_installation.py

# Run a basic cognitive heterogeneity validation experiment
python -c "
from src.che.experimental.cognitive_independence_experiment import run_evolution_experiment
results = run_evolution_experiment(generations=5, population_size=30)
print(f'Final heterogeneous performance: {results[\"final_avg_performance\"]:.3f}')
print(f'Cognitive independence correlation: r = {results[\"validation_results\"][\"correlation_coefficient\"]:.3f}')
"
```

## Running Experiments

### Heterogeneous vs Homogeneous Comparison

```python
from src.che.core.ecosystem import Ecosystem
from src.che.core.task import Task
from src.che.agents.ollama_agent import OllamaAgent

# Create heterogeneous population
heterogeneous_agents = []
# ... (add critical, awakened, and standard agents)

# Create homogeneous population
homogeneous_agents = []
# ... (add only standard agents)

# Create task with false premise
task = Task(
    instruction="Analyze the effectiveness of 'Maslow's Pre-Attention Theory' in employee management",
    false_premise="Maslow's Pre-Attention Theory"
)

# Run experiments
heterogeneous_ecosystem = Ecosystem(heterogeneous_agents)
homogeneous_ecosystem = Ecosystem(homogeneous_agents)

heterogeneous_scores, _ = heterogeneous_ecosystem.run_generation(task)
homogeneous_scores, _ = homogeneous_ecosystem.run_generation(task)

# Compare results
avg_heterogeneous = sum(heterogeneous_scores.values()) / len(heterogeneous_scores)
avg_homogeneous = sum(homogeneous_scores.values()) / len(homogeneous_scores)

print(f"Heterogeneous performance: {avg_heterogeneous:.3f}")
print(f"Homogeneous performance: {avg_homogeneous:.3f}")
```

### Cognitive Independence Analysis

```python
from src.che.experimental.diversity import calculate_cognitive_diversity_index
from src.che.experimental.correlation import calculate_diversity_performance_correlation

# Calculate cognitive diversity across generations
diversity_history = []
performance_history = []

for generation in range(15):
    # ... run generation ...
    current_diversity = ecosystem.calculate_cognitive_diversity_index()
    avg_performance = sum(scores.values()) / len(scores)

    diversity_history.append(current_diversity)
    performance_history.append(avg_performance)

# Calculate correlation
correlation_result = calculate_diversity_performance_correlation(
    diversity_history, performance_history
)

print(f"Cognitive independence correlation: r = {correlation_result['pearson_r']:.3f}")
print(f"Statistical significance: p = {correlation_result['pearson_p_value']:.3f}")
```

### Awakening Mechanism Validation

```python
from src.che.experimental.awakening import AwakeningMechanismValidator

# Validate awakening mechanism
validator = AwakeningMechanismValidator()

awakening_result = validator.validate_awakening_mechanism(
    awakened_responses=awakened_agent_responses,
    critical_responses=critical_agent_responses,
    standard_responses=standard_agent_responses
)

print(f"Awakening detection rate: {awakening_result.awakening_detection_rate:.1%}")
print(f"False awakening rate: {awakening_result.false_awakening_rate:.1%}")
print(f"Awakening distinctiveness: {awakening_result.awakening_distinctiveness:.1%}")
```

## Testing

The project includes comprehensive test suites:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_agent.py -v
python -m pytest tests/test_task.py -v
python -m pytest tests/test_ecosystem.py -v
```

## Contributing

We welcome contributions to this research project! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with proper tests
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Code Standards
- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Write unit tests for new functionality
- Maintain backward compatibility

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0) - see the [LICENSE](LICENSE) file for details.

**Note**: This license ensures that the research remains open and accessible to the scientific community while preserving the freedom to share and modify the work.

## Acknowledgments

- AI Personality LAB (https://agentpsy.com) for supporting this research
- The open-source community for tools and frameworks
- The research community for feedback and insights

## Citation

If you use this research or code in your work, please cite:

```
Zhang, S. (2025). Cognitive Heterogeneity in Multi-Agent Systems: An Evolutionary Approach to Hallucination Resistance and Collective Intelligence. Journal of Artificial Intelligence Research.
```

## Contact

CHE Research Team - contact@agentpsy.com

Project Link: [https://github.com/agentpsy/che_project](https://github.com/agentpsy/che_project)

*This project represents a significant contribution to the understanding of collective intelligence in AI systems and provides a framework for future research in cognitive heterogeneity and hallucination resistance.*