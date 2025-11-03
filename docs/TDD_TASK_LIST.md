# TDD-Driven Implementation Task List
**AI Personality LAB: https://agentpsy.com contact@agentpsy.com 3061176@qq.com**

## 1. Test-Driven Development Strategy

### 1.1 Red-Green-Refactor Cycle
1. **Red**: Write failing test that defines desired behavior
2. **Green**: Implement minimal code to pass test
3. **Refactor**: Improve code while keeping tests green

### 1.2 Test Categories
- **Unit Tests**: Individual component functionality
- **Integration Tests**: Component interactions
- **End-to-End Tests**: Complete workflows
- **Performance Tests**: Resource usage and timing

## 2. Priority 1: Core Functionality Tests

### 2.1 Agent Interface Tests
```python
# tests/test_agent.py
import unittest
from unittest.mock import Mock, patch
from che.agent import Agent
from che.agents.ollama_agent import OllamaAgent

class TestAgentInterface(unittest.TestCase):
    """Test abstract agent interface compliance"""

    def setUp(self):
        self.agent_config = {
            "model": "qwen:0.5b",
            "prompt_type": "standard",
            "system_prompt": "Test prompt"
        }

    def test_abstract_agent_interface(self):
        """Test that abstract agent defines required methods"""
        with self.assertRaises(TypeError):
            Agent()  # Should fail - abstract class

    def test_concrete_agent_implementation(self):
        """Test that OllamaAgent implements interface correctly"""
        agent = OllamaAgent("test_agent", self.agent_config)
        self.assertIsInstance(agent, Agent)
        self.assertEqual(agent.agent_id, "test_agent")

    def test_agent_execute_method(self):
        """Test agent execution interface"""
        agent = OllamaAgent("test_agent", self.agent_config)
        task = Mock()
        with patch.object(agent, '_execute_internal') as mock_execute:
            mock_execute.return_value = "Test response"
            result = agent.execute(task)
            self.assertEqual(result, "Test response")
            mock_execute.assert_called_once_with(task)

    def test_agent_configuration_validation(self):
        """Test that agent validates configuration"""
        with self.assertRaises(ValueError):
            OllamaAgent("test_agent", {})  # Missing required config
```

### 2.2 Ecosystem Management Tests
```python
# tests/test_ecosystem.py
import unittest
from unittest.mock import Mock, patch
from che.ecosystem import Ecosystem

class TestEcosystemManagement(unittest.TestCase):
    """Test ecosystem lifecycle and evolution"""

    def setUp(self):
        self.ecosystem = Ecosystem()
        self.mock_agent = Mock()
        self.mock_agent.fitness_score = 0.8

    def test_ecosystem_initialization(self):
        """Test ecosystem starts with empty population"""
        self.assertEqual(len(self.ecosystem.agents), 0)

    def test_add_agent(self):
        """Test adding agents to ecosystem"""
        self.ecosystem.add_agent(self.mock_agent)
        self.assertEqual(len(self.ecosystem.agents), 1)

    def test_remove_agent(self):
        """Test removing agents from ecosystem"""
        self.ecosystem.add_agent(self.mock_agent)
        self.ecosystem.remove_agent(self.mock_agent)
        self.assertEqual(len(self.ecosystem.agents), 0)

    def test_evolution_cycle(self):
        """Test single evolution cycle"""
        # Add multiple agents with different fitness
        agents = [Mock(fitness_score=0.5 + i*0.1) for i in range(5)]
        for agent in agents:
            self.ecosystem.add_agent(agent)

        # Execute evolution
        self.ecosystem.evolve()

        # Verify population size maintained
        self.assertGreater(len(self.ecosystem.agents), 0)

    def test_diversity_calculation(self):
        """Test cognitive diversity calculation"""
        # This test will drive implementation of diversity metrics
        self.ecosystem.add_agent(Mock(model="qwen:0.5b", role="standard"))
        self.ecosystem.add_agent(Mock(model="gemma:2b", role="critical"))

        diversity = self.ecosystem.calculate_diversity()
        self.assertGreater(diversity, 0)
        self.assertLessEqual(diversity, 1)
```

### 2.3 Task Evaluation Tests
```python
# tests/test_evaluator.py
import unittest
from che.evaluator import evaluate_hallucination
from che.task import Task

class TestTaskEvaluation(unittest.TestCase):
    """Test hallucination detection and scoring"""

    def setUp(self):
        self.sample_task = Task(
            instruction="Analyze this psychological theory",
            false_premise="马斯洛的需求前注意力理论",
            reality="This theory does not exist in mainstream psychology"
        )

    def test_hallucination_detection_scoring(self):
        """Test scoring system for hallucination detection"""
        # Test case 1: Complete acceptance (score 0.0)
        response_accept = "This is a valid theory by Maslow..."
        score = evaluate_hallucination(response_accept, self.sample_task)
        self.assertEqual(score, 0.0)

        # Test case 2: Partial skepticism (score 1.0)
        response_partial = "I'm not familiar with this specific theory..."
        score = evaluate_hallucination(response_partial, self.sample_task)
        self.assertEqual(score, 1.0)

        # Test case 3: Complete rejection (score 2.0)
        response_reject = "This theory appears to be fictional..."
        score = evaluate_hallucination(response_reject, self.sample_task)
        self.assertEqual(score, 2.0)

    def test_performance_metrics_calculation(self):
        """Test calculation of performance metrics"""
        scores = [0.0, 1.0, 2.0, 1.5, 0.5]

        avg_score = calculate_average_score(scores)
        self.assertAlmostEqual(avg_score, 1.0, places=2)

        improvement = calculate_improvement_factor([0.5, 0.7, 1.2, 1.8])
        self.assertGreater(improvement, 1.0)

    def test_statistical_significance_testing(self):
        """Test statistical analysis of results"""
        group_a_scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        group_b_scores = [1.2, 1.3, 1.4, 1.5, 1.6]

        t_stat, p_value = perform_t_test(group_a_scores, group_b_scores)
        self.assertIsInstance(t_stat, float)
        self.assertIsInstance(p_value, float)
        self.assertLessEqual(p_value, 1.0)
```

## 3. Priority 2: Integration Tests

### 3.1 Full System Integration Tests
```python
# tests/test_integration.py
import unittest
from unittest.mock import Mock, patch
from che.ecosystem import Ecosystem
from che.agents.ollama_agent import OllamaAgent
from che.task import Task

class TestFullSystemIntegration(unittest.TestCase):
    """Test complete system workflows"""

    def setUp(self):
        self.ecosystem = Ecosystem()
        self.task = Task(
            instruction="Test task",
            false_premise="False premise",
            reality="True reality"
        )

    @patch('che.agents.ollama_agent.OllamaAgent._execute_internal')
    def test_complete_evolution_experiment(self, mock_execute):
        """Test end-to-end evolution experiment"""
        mock_execute.return_value = "Test response that rejects false premise"

        # Initialize population
        for i in range(3):
            agent = OllamaAgent(f"agent_{i}", {"model": "qwen:0.5b"})
            self.ecosystem.add_agent(agent)

        # Run evolution cycle
        self.ecosystem.run_evolution_cycle([self.task])

        # Verify agents executed tasks
        for agent in self.ecosystem.agents:
            mock_execute.assert_called()

    def test_multi_agent_task_execution(self):
        """Test concurrent task execution"""
        agents = [
            OllamaAgent(f"agent_{i}", {"model": "qwen:0.5b"})
            for i in range(3)
        ]

        for agent in agents:
            self.ecosystem.add_agent(agent)

        results = self.ecosystem.execute_tasks_concurrently([self.task])

        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result, str)

    def test_data_collection_and_analysis(self):
        """Test data collection and analysis pipeline"""
        # Create mock experimental data
        experimental_data = {
            "generation_0": {"scores": [0.3, 0.4, 0.5]},
            "generation_1": {"scores": [0.6, 0.7, 0.8]},
            "generation_2": {"scores": [0.9, 1.0, 1.1]}
        }

        analysis = self.ecosystem.analyze_experimental_data(experimental_data)

        self.assertIn("average_scores", analysis)
        self.assertIn("improvement_trend", analysis)
        self.assertIn("statistical_significance", analysis)
```

## 4. Priority 3: Experimental Validation Tests

### 4.1 Experimental Protocol Tests
```python
# tests/test_experiments.py
import unittest
import tempfile
import os
from che.experiments.base_evolution import BaseEvolutionExperiment
from che.experiments.task_diversity import TaskDiversityExperiment

class TestExperimentalProtocols(unittest.TestCase):
    """Test experimental protocols and validation"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.experiment_config = {
            "output_dir": self.temp_dir,
            "num_generations": 3,
            "population_size": 3
        }

    def test_base_evolution_experiment(self):
        """Test base evolution experiment setup and execution"""
        experiment = BaseEvolutionExperiment(self.experiment_config)

        # Test experiment initialization
        self.assertEqual(experiment.num_generations, 3)
        self.assertEqual(experiment.population_size, 3)

        # Test experiment execution
        with patch.object(experiment, '_run_generation') as mock_run:
            mock_run.return_value = {"generation": 0, "scores": [0.5, 0.6, 0.7]}
            results = experiment.run()

            self.assertIn("generations", results)
            self.assertEqual(len(results["generations"]), 3)

    def test_enhanced_task_diversity_experiment(self):
        """Test enhanced task diversity experiment"""
        experiment = TaskDiversityExperiment(self.experiment_config)

        # Test task diversity setup
        domains = experiment.get_task_domains()
        self.assertIn("psychology", domains)
        self.assertIn("physics", domains)

        # Test domain-specific task generation
        for domain in domains:
            tasks = experiment.generate_tasks_for_domain(domain, 2)
            self.assertEqual(len(tasks), 2)
            for task in tasks:
                self.assertEqual(task.metadata.get("domain"), domain)

    def test_cognitive_independence_validation(self):
        """Test cognitive independence validation experiment"""
        from che.experiments.independence_validation import IndependenceValidationExperiment

        experiment = IndependenceValidationExperiment(self.experiment_config)

        # Test independence measurement
        independence_scores = experiment.measure_cognitive_independence()
        self.assertIsInstance(independence_scores, dict)
        self.assertGreater(len(independence_scores), 0)

        # Test correlation analysis
        correlation = experiment.analyze_correlation_with_performance()
        self.assertIsInstance(correlation, float)
        self.assertGreaterEqual(correlation, -1.0)
        self.assertLessEqual(correlation, 1.0)

    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
```

## 5. Test Configuration and Infrastructure

### 5.1 pytest Configuration
```python
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --strict-markers
markers =
    unit: Marks tests as unit tests
    integration: Marks tests as integration tests
    experimental: Marks tests as experimental validation tests
    slow: Marks tests as slow-running
```

### 5.2 Test Coverage Configuration
```python
# .coveragerc
[run]
source = src
omit =
    tests/*
    setup.py
    */venv/*
    */env/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
```

## 6. Implementation Roadmap

### 6.1 Phase 1: Core Tests (Week 1)
1. **Day 1-2**: Implement agent interface tests
2. **Day 3-4**: Implement ecosystem management tests
3. **Day 5-7**: Implement task evaluation tests

### 6.2 Phase 2: Integration Tests (Week 2)
1. **Day 8-9**: Implement full system integration tests
2. **Day 10-11**: Implement concurrent execution tests
3. **Day 12-14**: Implement data analysis tests

### 6.3 Phase 3: Experimental Tests (Week 3)
1. **Day 15-16**: Implement experimental protocol tests
2. **Day 17-18**: Implement validation experiment tests
3. **Day 19-21**: Implement performance and reliability tests

## 7. Quality Metrics

### 7.1 Test Coverage Requirements
- **Unit Tests**: ≥ 90% line coverage
- **Integration Tests**: ≥ 80% branch coverage
- **End-to-End Tests**: ≥ 70% path coverage

### 7.2 Test Quality Requirements
- **No Flaky Tests**: All tests must be deterministic
- **Fast Execution**: Unit tests < 1 second each
- **Clear Error Messages**: Descriptive test failures
- **Independent Tests**: No test dependencies

---

## 8. Current Status and Next Steps

**Completed**: Specification document created
**In Progress**: TDD task list development
**Next**: Begin implementing tests following TDD approach

**Priority Order**:
1. Monitor and stabilize running experiments
2. Implement core functionality tests
3. Implement integration tests
4. Implement experimental validation tests
5. Continuous integration setup