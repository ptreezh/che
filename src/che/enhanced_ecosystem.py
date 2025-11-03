from typing import List, Dict, Tuple, Optional
import uuid
import random
import numpy as np

from .agent import Agent
from .advanced_agent import AdvancedAgent, create_agent
from .task import Task
from .evaluator import evaluate_hallucination
from .ai_evaluator import evaluate_hallucination_ai


class EnhancedEcosystem:
    """Enhanced ecosystem with advanced cognitive diversity and evaluation."""

    def __init__(self, initial_agents: List[AdvancedAgent], use_ai_evaluator: bool = True):
        """
        Initializes the enhanced ecosystem with a starting population of advanced agents.

        Args:
            initial_agents: List of initial advanced agents
            use_ai_evaluator: Whether to use AI-based evaluation (default: True)
        """
        self.agents: Dict[str, AdvancedAgent] = {agent.agent_id: agent for agent in initial_agents}
        self.generation: int = 0
        self.history: List[Dict] = []
        self.use_ai_evaluator = use_ai_evaluator
        self.diversity_metrics = {
            'cognitive_diversity': 0.0,
            'behavioral_diversity': 0.0,
            'performance_diversity': 0.0
        }
        # Track lineage information
        self.lineage_tracker: Dict[str, Dict] = {}
        
        # Initialize lineage tracker with initial agents
        for agent in initial_agents:
            self.lineage_tracker[agent.agent_id] = {
                'original_source': None,  # Initial agents have no source
                'parent_agent_id': None,
                'is_variant': False,
                'generation_created': 0,
                'model_info': agent.config.get('model', 'unknown'),
                'role_info': agent.config.get('prompt', 'unknown'),
                'cognitive_type': getattr(agent, 'cognitive_type', 'unknown')
            }

    def calculate_cognitive_diversity(self) -> float:
        """
        Calculate cognitive diversity across the agent population.
        """
        if len(self.agents) < 2:
            return 0.0

        # Get cognitive types for all agents
        cognitive_types = [agent.cognitive_type for agent in self.agents.values()]
        unique_types = len(set(cognitive_types))
        total_agents = len(cognitive_types)

        # Calculate diversity as ratio of unique types to total agents
        diversity = unique_types / total_agents if total_agents > 0 else 0.0

        # Also consider the distribution of types
        type_counts = {}
        for t in cognitive_types:
            type_counts[t] = type_counts.get(t, 0) + 1
        
        # Calculate entropy as a measure of distribution diversity
        entropy = 0.0
        for count in type_counts.values():
            p = count / total_agents
            if p > 0:
                entropy -= p * np.log2(p)
        
        # Normalize entropy
        max_entropy = np.log2(len(type_counts)) if type_counts else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return max(diversity, normalized_entropy)

    def run_generation(self, task: Task) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, Dict]]:
        """
        Runs a single generation: all agents execute the task and get evaluated.

        Args:
            task: The task for the current generation.

        Returns:
            A tuple containing:
            - A dictionary mapping agent_id to its score.
            - A dictionary mapping agent_id to its raw output.
            - A dictionary mapping agent_id to its meta-cognitive evaluation.
        """
        scores: Dict[str, float] = {}
        outputs: Dict[str, str] = {}
        meta_evaluations: Dict[str, Dict] = {}

        for agent_id, agent in self.agents.items():
            # Execute the task
            output = agent.execute(task)
            
            # Evaluate the output
            if self.use_ai_evaluator:
                score = evaluate_hallucination_ai(output, task.false_premise, task.instruction)
            else:
                score = evaluate_hallucination(output, task.false_premise)
            
            scores[agent_id] = score
            outputs[agent_id] = output
            
            # Perform meta-cognitive evaluation if it's an AdvancedAgent
            if isinstance(agent, AdvancedAgent):
                meta_eval = agent.evaluate_response(output, task)
                meta_evaluations[agent_id] = meta_eval
                
                # Update agent's performance history
                agent.update_performance(score, task)
                
                # Adapt agent behavior based on performance
                agent.adapt_behavior()
            else:
                meta_evaluations[agent_id] = {}

        # Update diversity metrics
        self.diversity_metrics['cognitive_diversity'] = self.calculate_cognitive_diversity()
        
        # Calculate behavioral and performance diversity
        performance_values = list(scores.values())
        if len(performance_values) > 1:
            self.diversity_metrics['performance_diversity'] = np.std(performance_values)
            self.diversity_metrics['behavioral_diversity'] = np.mean(list(
                agent.diversity_contribution for agent in self.agents.values()
            )) if self.agents else 0.0

        return scores, outputs, meta_evaluations

    def evolve(self, scores: Dict[str, float], selection_pressure: float = 1.0):
        """
        Evolves the agent population based on the scores from the last generation.
        Enhanced with sophisticated selection and mutation mechanisms.
        """
        if not scores or len(self.agents) <= 1:
            return  # Cannot evolve with no scores or one/zero agents

        # Sort agents by score (best first)
        sorted_agents = sorted(self.agents.items(), key=lambda x: scores.get(x[0], 0), reverse=True)
        
        # Determine number of agents to keep (elitism)
        elite_count = max(1, int(len(sorted_agents) * 0.5))  # Keep top 50%
        new_agents = {}
        
        # Keep elite agents (inherit to next generation with new IDs)
        for idx, (agent_id, agent) in enumerate(sorted_agents[:elite_count]):
            # Create new ID in the format: gen_X_agent_YY_type
            new_id = f"gen_{self.generation + 1}_agent_{idx+1:02d}_{agent.cognitive_type}"
            
            # Update agent ID to match new naming convention
            agent.agent_id = new_id
            
            # Update lineage information
            self.lineage_tracker[new_id] = {
                'original_source': agent_id,  # Original ID before evolution
                'parent_agent_id': agent_id,  # Parent agent
                'is_variant': False,  # Inherited, not variant
                'generation_created': self.generation + 1,
                'model_info': agent.config.get('model', 'unknown'),
                'role_info': agent.config.get('prompt', 'unknown'),
                'cognitive_type': getattr(agent, 'cognitive_type', 'unknown')
            }
            
            new_agents[new_id] = agent

        # Generate new agents to fill population
        num_agents_to_create = len(self.agents) - len(new_agents)
        
        for i in range(num_agents_to_create):
            # Select parent using tournament selection with selection pressure
            tournament_size = max(2, int(len(sorted_agents) * 0.1))  # 10% of population
            tournament_contestants = random.sample(sorted_agents, min(tournament_size, len(sorted_agents)))
            parent_id, parent_agent = max(tournament_contestants, key=lambda x: scores.get(x[0], 0))
            
            # Create new agent ID in the format: gen_X_agent_YY_type
            new_idx = len(new_agents) + 1
            new_id = f"gen_{self.generation + 1}_agent_{(elite_count + i + 1):02d}_{parent_agent.cognitive_type}"
            
            # Apply mutation with probability based on selection pressure
            mutation_rate = min(0.5, 0.1 * selection_pressure)  # Higher selection pressure = more mutation
            
            # Create new agent by replicating parent with possible mutation
            if isinstance(parent_agent, AdvancedAgent):
                # Get the parent's config to pass to replication
                parent_config = parent_agent.config.copy()
                
                # Create new agent
                new_agent = create_agent(new_id, parent_agent.cognitive_type, parent_config)
                
                # Apply mutation if random chance occurs
                if random.random() < mutation_rate:
                    # Mutate cognitive type
                    cognitive_types = [
                        "critical", "standard", "awakened", "innovative", 
                        "analytical", "collaborative", "sceptical", 
                        "empathetic", "systemic", "pragmatic"
                    ]
                    current_type = new_agent.cognitive_type
                    other_types = [t for t in cognitive_types if t != current_type]
                    if other_types:
                        mutated_type = random.choice(other_types)
                        # Create new agent with mutated cognitive type
                        new_agent = create_agent(new_id, mutated_type, parent_config)
                        
                        # Update agent's cognitive type to reflect mutation
                        new_agent.cognitive_type = mutated_type
            else:
                # For non-advanced agents, create using the factory
                parent_type = getattr(parent_agent, 'cognitive_type', 'standard')
                new_agent = create_agent(new_id, parent_type, parent_agent.config.copy())
            
            new_agent.agent_id = new_id  # Ensure agent's ID is set correctly
            
            # Update lineage information
            self.lineage_tracker[new_id] = {
                'original_source': parent_id,
                'parent_agent_id': parent_id,
                'is_variant': random.random() < mutation_rate,  # Mark as variant if mutated
                'generation_created': self.generation + 1,
                'model_info': new_agent.config.get('model', 'unknown'),
                'role_info': new_agent.config.get('prompt', 'unknown'),
                'cognitive_type': getattr(new_agent, 'cognitive_type', 'unknown')
            }
            
            new_agents[new_id] = new_agent

        # Replace the old population
        self.agents = new_agents
        self.generation += 1

    def get_diversity_report(self) -> Dict[str, float]:
        """
        Get a report on the cognitive diversity of the ecosystem.
        """
        return self.diversity_metrics.copy()

    def get_population_composition(self) -> Dict[str, int]:
        """
        Get the count of each cognitive type in the population.
        """
        composition = {}
        for agent in self.agents.values():
            agent_type = getattr(agent, 'cognitive_type', 'unknown')
            composition[agent_type] = composition.get(agent_type, 0) + 1
        return composition
    
    def save_generation_lineage(self, generation: int = None):
        """
        Save the lineage information for the current generation to a JSON file.
        """
        import json
        import os
        from pathlib import Path
        
        gen_num = generation if generation is not None else self.generation
        lineage_dir = Path("lineage_records")
        lineage_dir.mkdir(exist_ok=True)
        
        # Create lineage data for current agents
        current_generation_agents = {}
        for agent_id, agent in self.agents.items():
            if agent_id in self.lineage_tracker:
                current_generation_agents[agent_id] = self.lineage_tracker[agent_id]
        
        # Create filename with generation number
        filename = lineage_dir / f"gen_{gen_num}_lineage.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'generation': gen_num,
                'timestamp': __import__('time').time(),
                'population_size': len(current_generation_agents),
                'agents': current_generation_agents
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Lineage information saved for generation {gen_num}: {filename}")
        
    def get_generation_lineage(self, generation: int) -> Dict:
        """
        Get lineage information for a specific generation.
        """
        import json
        from pathlib import Path
        
        lineage_dir = Path("lineage_records")
        filename = lineage_dir / f"gen_{generation}_lineage.json"
        
        if filename.exists():
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"Lineage file for generation {generation} not found: {filename}")
            return {}