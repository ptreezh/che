# CHE Project Specification Document
**AI Personality LAB: https://agentpsy.com contact@agentpsy.com 3061176@qq.com**

## 1. Architecture Principles

### 1.1 BMAD (Build, Measure, Analyze, Deploy)
- **Build**: Minimal viable prototype focusing on core validation
- **Measure**: Quantitative metrics for cognitive heterogeneity and performance
- **Analyze**: Statistical validation of cognitive independence hypothesis
- **Deploy**: Research-oriented deployment with reproducible experiments

### 1.2 KISS (Keep It Simple, Stupid)
- Single responsibility for each component
- Clear interfaces between modules
- Minimal dependencies
- Straightforward data flow

### 1.3 YAGNI (You Ain't Gonna Need It)
- No premature optimization
- Only implement what's necessary for hypothesis validation
- Avoid over-engineering
- Focus on core research questions

### 1.4 SOLID Principles
- **S**ingle Responsibility: Each class has one reason to change
- **O**pen/Closed: Open for extension, closed for modification
- **L**iskov Substitution: Subtypes must be substitutable for base types
- **I**nterface Segregation: Client-specific interfaces
- **D**ependency Inversion: Depend on abstractions, not concretions

## 2. System Architecture

### 2.1 Core Components
```
src/che/
├── __init__.py                 # Package initialization
├── agent.py                    # Abstract agent interface
├── ecosystem.py                # Population management and evolution
├── task.py                     # Task data structures
├── evaluator.py                # Performance evaluation
├── prompts.py                  # Agent behavior prompts
└── agents/
    └── ollama_agent.py         # Ollama LLM implementation
```

### 2.2 Component Responsibilities

#### Agent System (`agent.py`, `agents/`)
- **Purpose**: Define and implement agent interfaces
- **SOLID**: Single responsibility for agent behavior
- **KISS**: Simple abstract base class pattern

#### Ecosystem Engine (`ecosystem.py`)
- **Purpose**: Manage population lifecycle and evolution
- **SOLID**: Open/closed for different evolutionary strategies
- **YAGNI**: Only implement selection and replication

#### Task & Evaluation (`task.py`, `evaluator.py`)
- **Purpose**: Define tasks and evaluate performance
- **SOLID**: Separate task definition from evaluation logic
- **KISS**: Clear evaluation criteria

## 3. TDD Task List

### 3.1 Test-Driven Development Implementation

#### Phase 1: Core Functionality Tests
```python
# tests/test_agent.py
class TestAgentInterface:
    def test_abstract_agent_interface()
    def test_concrete_agent_implementation()
    def test_agent_response_validation()

# tests/test_ecosystem.py
class TestEcosystemManagement:
    def test_population_initialization()
    def test_evolution_cycle()
    def test_diversity_maintenance()

# tests/test_evaluation.py
class TestTaskEvaluation:
    def test_hallucination_detection_scoring()
    def test_performance_metrics_calculation()
    def test_statistical_significance_testing()
```

#### Phase 2: Integration Tests
```python
# tests/test_integration.py
class TestFullSystemIntegration:
    def test_complete_evolution_experiment()
    def test_multi_agent_task_execution()
    def test_data_collection_and_analysis()
```

#### Phase 3: Experimental Validation Tests
```python
# tests/test_experiments.py
class TestExperimentalProtocols:
    def test_base_evolution_experiment()
    def test_enhanced_task_diversity_experiment()
    def test_cognitive_independence_validation()
```

### 3.2 Test Coverage Requirements
- **Unit Tests**: ≥ 90% code coverage
- **Integration Tests**: All component interactions
- **End-to-End Tests**: Complete experimental workflows
- **Performance Tests**: Execution time and resource usage

## 4. Current Functionality Assessment

### 4.1 Working Components
- ✅ Abstract agent interface (`src/che/agent.py`)
- ✅ Ollama agent implementation (`src/che/agents/ollama_agent.py`)
- ✅ Ecosystem management (`src/che/ecosystem.py`)
- ✅ Task structures (`src/che/task.py`)
- ✅ Evaluation framework (`src/che/evaluator.py`)
- ✅ System prompts (`src/che/prompts.py`)
- ✅ Main simulation (`main.py`)

### 4.2 Experimental Scripts Status
- ⏳ `enhanced_task_diversity_experiment.py` (running)
- ⏳ `diverse_real_experiments.py` (running)
- ⏳ `complete_evolution_experiment.py` (running)
- ⏳ `real_cognitive_independence_experiment.py` (running)
- ⏳ `enhanced_cognitive_independence_experiment.py` (running)
- ⏳ `enhanced_task_diversity_experiment_fixed.py` (running)
- ⏳ `main.py` (running)

### 4.3 Documentation Status
- ✅ JAIR format LaTeX paper (`cognitive_heterogeneity_paper.tex`)
- ✅ Theoretical framework enhancement (`theoretical_framework_enhancement.tex`)
- ✅ Experimental charts (`experimental_charts.py`)
- ✅ Open-source package structure (`gitup/`)

## 5. Implementation Roadmap

### 5.1 Priority 1: Stability and Reliability
1. **Health Check**: Verify all running experiments
2. **Error Handling**: Add robust error handling and logging
3. **Data Integrity**: Ensure experimental data is properly saved
4. **Resource Management**: Monitor and optimize resource usage

### 5.2 Priority 2: Testing Infrastructure
1. **Unit Tests**: Implement comprehensive unit tests
2. **Integration Tests**: Test component interactions
3. **Experimental Tests**: Validate experimental protocols
4. **Performance Tests**: Ensure system reliability

### 5.3 Priority 3: Documentation and Deployment
1. **API Documentation**: Generate comprehensive API docs
2. **User Guide**: Create user-friendly documentation
3. **Deployment Scripts**: Automate deployment process
4. **Monitoring**: Add system monitoring and alerts

## 6. Quality Assurance

### 6.1 Code Quality Standards
- **PEP 8**: Python style guide compliance
- **Type Hints**: 100% type annotation coverage
- **Docstrings**: Comprehensive documentation
- **Linting**: Automated code quality checks

### 6.2 Experimental Validation
- **Reproducibility**: All experiments must be reproducible
- **Statistical Rigor**: Proper statistical analysis
- **Data Management**: Structured data storage and retrieval
- **Version Control**: Clear experiment versioning

## 7. Risk Assessment

### 7.1 Technical Risks
- **Resource Exhaustion**: Multiple experiments running concurrently
- **Data Loss**: Experimental data not properly saved
- **Dependency Issues**: Ollama model availability and compatibility
- **Performance Bottlenecks**: Large-scale experiment execution

### 7.2 Mitigation Strategies
- **Resource Monitoring**: Implement resource usage tracking
- **Data Backup**: Automated backup of experimental results
- **Dependency Management**: Version pinning and compatibility testing
- **Performance Optimization**: Incremental scaling and optimization

## 8. Success Metrics

### 8.1 Research Metrics
- **Hypothesis Validation**: Clear evidence for/against cognitive independence
- **Statistical Significance**: p < 0.05 for key findings
- **Effect Size**: Meaningful practical significance
- **Reproducibility**: Independent replication possible

### 8.2 Technical Metrics
- **System Reliability**: 99% uptime for experiments
- **Data Quality**: Complete and consistent experimental data
- **Performance**: Reasonable execution times
- **Test Coverage**: ≥ 90% test coverage

---

## 9. Current Status Summary

**Active Experiments**: 7 concurrent experiments running
**Documentation**: Comprehensive theoretical framework completed
**Code Quality**: Needs structured testing approach
**Priority**: Stabilize running experiments, then implement TDD

**Next Steps**:
1. Monitor and verify running experiments
2. Implement comprehensive test suite
3. Enhance error handling and logging
4. Create deployment-ready package
5. Finalize research paper and documentation