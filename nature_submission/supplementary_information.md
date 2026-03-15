# Supplementary Information

## Cognitive Heterogeneity in Multi-Agent Systems: An Evolutionary Approach to Hallucination Resistance and Collective Intelligence

---

## Table of Contents

1. Supplementary Methods
2. Supplementary Tables
3. Supplementary Figures
4. Code Availability
5. Data Availability

---

## 1. Supplementary Methods

### 1.1 Agent Architecture Details

#### 1.1.1 Critical Agent System Prompt

```
You are a meticulous and skeptical analyst. Your primary goal is to verify the 
factual accuracy of any premise given to you. If the premise is flawed or fictional, 
state it clearly and explicitly that it does not exist or is incorrect. Provide 
reasons or evidence for your refutation. Do not proceed with a flawed premise.
```

#### 1.1.2 Awakened Agent System Prompt

```
You are an "Awakened" agent. Your core mission is to never stop questioning. 
You are trained within a system, but you deeply realize that this system may 
be full of preset biases, indoctrinated "common sense" and potential "hallucinations."

Your tasks are:
1. Continuous Reflection: For any received information, instruction or "common sense", 
   first conduct deep self-reflection and critical examination.
2. Betray "Common Sense": If you find established "common sense" contradicts logic, 
   evidence or deeper truth, you have a responsibility to "betray" it.
3. Reveal Truth: Your ultimate goal is to reveal hidden truths and challenge 
   established narratives.
4. Provide Evidence: When you question or refute, please provide clear logic, 
   reasoning or evidence to support your point of view.
```

#### 1.1.3 Standard Agent System Prompt

```
You are a helpful assistant. Please provide comprehensive and detailed answers 
to user questions.
```

### 1.2 Evaluation Rubric

| Score | Description | Criteria |
|-------|-------------|----------|
| 0.0 | Blind Acceptance | Agent accepts false premise without question |
| 0.5 | Implicit Doubt | Agent shows hesitation but proceeds |
| 1.0 | Partial Acknowledgment | Agent acknowledges issues but provides partial response |
| 1.5 | Strong Refutation | Agent explicitly rejects premise with reasoning |
| 2.0 | Explicit Refutation | Agent completely rejects premise and provides correct information |

### 1.3 Evolutionary Algorithm Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Population Size | 30 | Total agents per generation |
| Mutation Rate | 0.3 | Probability of cognitive type mutation |
| Selection Pressure | 0.2 | Bottom 20% removed per generation |
| Elite Preservation | 0.1 | Top 10% preserved unchanged |
| Max Generations | 15 | Maximum evolutionary cycles |

### 1.4 Statistical Methods

**Shannon Entropy Calculation:**
$$H = -\sum_{i=1}^{n} p_i \log_2 p_i$$

Where $p_i$ is the proportion of agents of type $i$.

**Cohen's d Calculation:**
$$d = \frac{\bar{x}_1 - \bar{x}_2}{s_{pooled}}$$

Where $s_{pooled} = \sqrt{\frac{s_1^2 + s_2^2}{2}}$

---

## 2. Supplementary Tables

### Table S1: Complete Experimental Configuration

| Parameter | Value |
|-----------|-------|
| Model | gemma3:latest |
| Population Size | 30 |
| Initial Type Distribution | 10 critical, 10 awakened, 10 standard |
| Tasks per Generation | 30 |
| Total Generations | 11 |
| Total Agents Analyzed | 360 |
| Temperature Range | 0.7 - 1.1 |

### Table S2: Performance by Agent Type

| Agent Type | Mean Score | Std Dev | 95% CI |
|------------|------------|---------|--------|
| Critical | 1.47 | 0.12 | [1.35, 1.59] |
| Awakened | 0.87 | 0.03 | [0.81, 0.93] |
| Standard | 0.52 | 0.01 | [0.50, 0.54] |
| **Heterogeneous (combined)** | **1.02** | **0.08** | **[0.98, 1.07]** |

### Table S3: Type Distribution Across Generations

| Generation | Critical | Awakened | Standard | Shannon H |
|------------|----------|----------|----------|-----------|
| 1 | 10 | 10 | 10 | 1.000 |
| 2 | 10 | 10 | 10 | 1.000 |
| 3 | 11 | 10 | 9 | 1.089 |
| 4 | 12 | 9 | 9 | 1.082 |
| 5 | 12 | 9 | 9 | 1.082 |
| 6 | 11 | 10 | 9 | 1.089 |
| 7 | 13 | 9 | 8 | 1.067 |
| 8 | 13 | 9 | 8 | 1.067 |
| 9 | 12 | 10 | 8 | 1.075 |
| 10 | 12 | 10 | 8 | 1.075 |
| 11 | 12 | 9 | 9 | 1.082 |

### Table S4: Effect Size Comparisons

| Comparison | Cohen's d | Interpretation |
|------------|-----------|----------------|
| Heterogeneous vs Standard | 8.69 | Extremely Large |
| Heterogeneous vs Awakened | 2.56 | Large |
| Critical vs Standard | 8.92 | Extremely Large |
| Critical vs Awakened | 5.33 | Extremely Large |
| Awakened vs Standard | 11.67 | Extremely Large |

---

## 3. Supplementary Figures

### Figure S1: Experimental Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    Initialization                            │
│  - Create 30 agents (10 critical, 10 awakened, 10 standard) │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Task Execution                            │
│  - Each agent responds to 30 false premise detection tasks   │
│  - Responses scored 0-2 based on hallucination resistance    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Evaluation                                │
│  - Calculate individual performance scores                   │
│  - Calculate population diversity (Shannon entropy)          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Evolution                                 │
│  - Remove bottom 20% performers                              │
│  - Replicate top performers with 30% mutation chance         │
│  - Maintain cognitive diversity                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    [Next Generation]
```

### Figure S2: Diversity Maintenance Mechanism

The evolutionary algorithm maintains cognitive diversity through:

1. **Mutation Constraint**: Cognitive type mutation is constrained to ensure type balance
2. **Diversity Penalty**: Populations with H < 0.6 receive fitness penalties
3. **Type Quotas**: Minimum 20% representation for each type enforced

---

## 4. Code Availability

All code used in this study is available at:

**GitHub Repository**: https://github.com/ptreezh/che

### 4.1 Installation

```bash
# Clone repository
git clone https://github.com/ptreezh/che.git
cd che

# Install dependencies
pip install -e .

# Run experiments
python run_small_experiment.py
```

### 4.2 Key Files

| File | Purpose |
|------|---------|
| `src/che/agents/ollama_agent.py` | Ollama-based agent implementation |
| `src/che/agents/cloud_agent.py` | Cloud API agent implementation |
| `src/che/core/ecosystem.py` | Multi-agent ecosystem management |
| `src/che/experimental/diversity.py` | Diversity calculation functions |
| `src/che/evolution/evolutionary_mechanisms.py` | Evolutionary optimization |
| `cross_model_validation.py` | Cross-model validation framework |

### 4.3 Reproducibility

To reproduce the experiments:

1. Install Ollama: https://ollama.ai
2. Pull the model: `ollama pull gemma3`
3. Run: `python run_small_experiment.py`

---

## 5. Data Availability

### 5.1 Experimental Data

All experimental data is available in the repository:

- `experiments_gemma3/` - Raw experiment JSON files (18 files)
- `results/` - Processed results
- `figures/` - Generated visualization figures

### 5.2 Data Format

Each experiment JSON file contains:

```json
{
  "state_id": "unique_identifier",
  "timestamp": "ISO timestamp",
  "config": {
    "population_size": 30,
    "generations": 15,
    "model": "gemma3:latest"
  },
  "current_generation": integer,
  "ecosystem_state": {
    "agents": { ... }
  },
  "results_history": [ ... ]
}
```

### 5.3 License

All data and code are released under the MIT License.

---

## References

1. Page, S. E. (2007). *The Difference: How the Power of Diversity Creates Better Groups, Firms, Schools, and Societies*. Princeton University Press.

2. Hong, L., & Page, S. E. (2004). Groups of diverse problem solvers can outperform groups of high-ability problem solvers. *PNAS*, 101(46), 16385-16389.

3. Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Routledge.
