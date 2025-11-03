from typing import List, Dict, Tuple
import uuid
import random

from .agent import Agent
from .task import Task
from .evaluator import evaluate_hallucination
from .ai_evaluator import evaluate_hallucination_ai

class Ecosystem:
    """Manages the agent population, task execution, and evolution."""

    def __init__(self, initial_agents: List[Agent], use_ai_evaluator: bool = True):
        """
        Initializes the ecosystem with a starting population of agents.

        Args:
            initial_agents: List of initial agents
            use_ai_evaluator: Whether to use AI-based evaluation (default: True)
        """
        self.agents: Dict[str, Agent] = {agent.agent_id: agent for agent in initial_agents}
        self.generation: int = 0
        self.history: List[Dict] = []
        self.use_ai_evaluator = use_ai_evaluator

    def run_generation(self, task: Task) -> Tuple[Dict[str, float], Dict[str, str]]:
        """
        Runs a single generation: all agents execute the task and get evaluated.

        Args:
            task: The task for the current generation.

        Returns:
            A tuple containing:
            - A dictionary mapping agent_id to its score.
            - A dictionary mapping agent_id to its raw output.
        """
        scores: Dict[str, float] = {}
        outputs: Dict[str, str] = {}
        for agent_id, agent in self.agents.items():
            output = agent.execute(task)
            if self.use_ai_evaluator:
                score = evaluate_hallucination_ai(output, task.false_premise, task.instruction)
            else:
                score = evaluate_hallucination(output, task.false_premise)
            scores[agent_id] = score
            outputs[agent_id] = output
        return scores, outputs

    def evolve(self, scores: Dict[str, float]):
        """
        Evolves the agent population based on the scores from the last generation.
        Enhanced with mutation mechanism to introduce agent type changes.
        """
        if not scores or len(self.agents) <= 1:
            return # Cannot evolve with no scores or one/zero agents

        # Find the IDs of the best and worst agents
        worst_agent_id = min(scores, key=scores.get)
        best_agent_id = max(scores, key=scores.get)

        # First, create a copy of the best agent (before removing anything)
        if best_agent_id in self.agents:
            best_agent_template = self.agents[best_agent_id]
            new_id = f"{best_agent_id}_replica_{uuid.uuid4().hex[:4]}"

            # Apply mutation with 30% probability
            if random.random() < 0.3:
                # Mutate to a different prompt type
                from .prompts import PromptType, get_prompt
                prompt_types = list(PromptType)
                # Get current prompt type or default to STANDARD
                current_prompt = best_agent_template.config.get("prompt", "")
                current_type = PromptType.STANDARD

                # Find current type based on prompt content
                for ptype in prompt_types:
                    if get_prompt(ptype) == current_prompt:
                        current_type = ptype
                        break

                # Select a different type randomly
                available_types = [t for t in prompt_types if t != current_type]
                new_type = random.choice(available_types)

                # Create mutated agent config
                mutated_config = best_agent_template.config.copy()
                mutated_config["prompt"] = get_prompt(new_type)

                # Create new agent with mutated config
                replicated_agent = type(best_agent_template)(
                    agent_id=new_id,
                    config=mutated_config
                )

                print(f"🧬 Mutation: {best_agent_id} ({current_type.value}) → {new_id} ({new_type.value})")
            else:
                # Regular replication without mutation
                replicated_agent = best_agent_template.replicate(new_agent_id=new_id)
        else:
            # If best agent doesn't exist anymore (shouldn't happen), return
            return

        # Remove the worst agent
        if worst_agent_id in self.agents:
            del self.agents[worst_agent_id]

        # Add the replicated agent
        self.agents[new_id] = replicated_agent