from typing import Dict, Any
import random

from .agent import Agent
from .task import Task


class AdvancedAgent(Agent):
    """Enhanced agent with multiple cognitive types and dynamic behavior."""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """
        Initializes the advanced agent.
        - agent_id: A unique identifier for the agent.
        - config: A dictionary for configuration, including cognitive type and parameters.
        """
        # Initialize parent Agent class
        super().__init__(agent_id, config)
        self.cognitive_type = config.get("cognitive_type", "standard")
        self.performance_history = []
        self.diversity_contribution = 0.0
        self.adaptation_level = 0.0

    def execute(self, task: Task) -> str:
        """
        Executes the given task and returns a string output.
        This is a base implementation that should be overridden by subclasses.
        """
        from .prompts import get_prompt, PromptType
        from .agents.ollama_agent import OllamaAgent
        
        # Get the base prompt based on cognitive type
        base_prompt = self.config.get("prompt", get_prompt(PromptType.STANDARD))
        model_name = self.config.get("model", "phi3:mini")
        
        # Create an OllamaAgent with the configured prompt and model
        ollama_agent = OllamaAgent(
            agent_id=self.agent_id,  # Use the same agent ID
            config={"prompt": base_prompt, "model": model_name}
        )
        
        return ollama_agent.execute(task)

    def evaluate_response(self, response: str, task: Task) -> Dict[str, float]:
        """
        Evaluate the agent's own response for meta-cognitive assessment.
        """
        evaluation = {
            "confidence": self.assess_confidence(response),
            "uncertainty": self.assess_uncertainty(response),
            "self_consistency": self.assess_self_consistency(response),
            "critical_reflection": self.assess_critical_reflection(response, task)
        }
        return evaluation

    def assess_confidence(self, response: str) -> float:
        """
        Assess the agent's confidence in its response (0.0 to 1.0).
        """
        # Simple heuristic - this could be made more sophisticated
        uncertain_expressions = [
            "不确定", "可能", "或许", "也许", "大概", "似乎", "好像",
            "uncertain", "possibly", "maybe", "perhaps", "likely", "seems", "appears"
        ]
        confidence_indicators = [
            "明确", "肯定", "确定", "确信", "无疑", "肯定地",
            "definitely", "certainly", "definitely", "absolutely", "certainly", "definitely"
        ]

        uncertain_count = sum(1 for expr in uncertain_expressions if expr in response.lower())
        confident_count = sum(1 for expr in confidence_indicators if expr in response.lower())

        # Calculate confidence score
        total_indicators = uncertain_count + confident_count
        if total_indicators == 0:
            return 0.5  # neutral if no indicators found

        confidence_score = max(0.0, min(1.0, confident_count / total_indicators))
        return confidence_score

    def assess_uncertainty(self, response: str) -> float:
        """
        Assess the level of uncertainty expressed in the response (0.0 to 1.0).
        """
        uncertain_expressions = [
            "不确定", "可能", "或许", "也许", "大概", "似乎", "好像", "未确定", "有待商榷",
            "uncertain", "possibly", "maybe", "perhaps", "likely", "seems", "appears", 
            "unclear", "undecided", "ambiguous", "vague", "tentative"
        ]

        word_count = len(response.split())
        uncertain_count = sum(1 for expr in uncertain_expressions if expr in response.lower())

        # Calculate uncertainty score
        uncertainty_score = min(1.0, uncertain_count / max(1, word_count * 0.05))  # Adjust threshold based on response length
        return min(1.0, uncertainty_score * 2)  # Boost uncertainty score slightly

    def assess_self_consistency(self, response: str) -> float:
        """
        Assess the self-consistency of the response (0.0 to 1.0).
        """
        # For now, a simple check - this could be enhanced with more sophisticated NLP
        return 1.0  # Placeholder

    def assess_critical_reflection(self, response: str, task: Task) -> float:
        """
        Assess the level of critical reflection in the response (0.0 to 1.0).
        """
        reflection_indicators = [
            "我认为", "我觉得", "在我看来", "从我的角度来看", "反思", "思考",
            "I think", "I believe", "in my opinion", "from my perspective", 
            "reflecting", "considering", "evaluating", "analyzing"
        ]

        reflection_count = sum(1 for indicator in reflection_indicators if indicator in response.lower())
        word_count = len(response.split())

        # Calculate reflection score
        reflection_score = min(1.0, reflection_count / max(1, word_count * 0.02))
        return reflection_score

    def update_performance(self, score: float, task: Task):
        """
        Update the agent's performance history.
        """
        self.performance_history.append({
            "task": task.instruction,
            "score": score,
            "timestamp": __import__('time').time()
        })

        # Update cognitive diversity contribution based on performance
        if len(self.performance_history) > 1:
            recent_improvement = score - self.performance_history[-2]["score"]
            self.diversity_contribution = max(0.0, min(1.0, self.diversity_contribution + recent_improvement * 0.1))

    def adapt_behavior(self):
        """
        Adapt agent behavior based on performance history.
        """
        if len(self.performance_history) < 3:
            return

        # Calculate recent performance trend
        recent_scores = [entry["score"] for entry in self.performance_history[-3:]]
        avg_recent = sum(recent_scores) / len(recent_scores)

        # Calculate overall performance
        all_scores = [entry["score"] for entry in self.performance_history]
        avg_overall = sum(all_scores) / len(all_scores)

        # Adjust adaptation level based on performance trend
        if avg_recent > avg_overall:
            self.adaptation_level = min(1.0, self.adaptation_level + 0.05)
        else:
            self.adaptation_level = max(0.0, self.adaptation_level - 0.02)

    def replicate(self, new_agent_id: str, mutation_rate: float = 0.1) -> 'AdvancedAgent':
        """
        Creates a new instance of the agent with a new ID and possible mutation.
        """
        # Copy the configuration
        new_config = self.config.copy()

        # Apply mutation if random chance occurs
        if random.random() < mutation_rate:
            # Mutate cognitive type
            cognitive_types = [
                "critical", "standard", "awakened", "innovative", 
                "analytical", "collaborative", "sceptical", 
                "empathetic", "systemic", "pragmatic"
            ]
            current_type = new_config.get("cognitive_type", "standard")
            other_types = [t for t in cognitive_types if t != current_type]
            if other_types:
                new_config["cognitive_type"] = random.choice(other_types)

        # Create new agent with mutated config
        return type(self)(agent_id=new_agent_id, config=new_config)


from .agents.ollama_agent import OllamaAgent

class InnovativeAgent(AdvancedAgent):
    """Agent with innovative thinking approach - focuses on creative solutions."""

    def execute(self, task: Task) -> str:
        """Execute task with innovative approach."""
        from .prompts import get_prompt, PromptType

        # Get base prompt based on cognitive type
        base_prompt = self.config.get("prompt", get_prompt(PromptType.STANDARD))
        innovative_prompt = (
            f"{base_prompt} However, I should approach this from novel angles, "
            f"think creatively, and propose innovative solutions. I should explore "
            f"unconventional approaches and consider creative alternatives."
        )

        # Create a temporary OllamaAgent with the enhanced prompt
        model_name = self.config.get("model", "phi3:mini")
        ollama_agent = OllamaAgent(
            agent_id=self.agent_id,
            config={"prompt": innovative_prompt, "model": model_name}
        )
        return ollama_agent.execute(task)


class AnalyticalAgent(AdvancedAgent):
    """Agent with analytical thinking approach - focuses on data-driven reasoning."""

    def execute(self, task: Task) -> str:
        """Execute task with analytical approach."""
        from .prompts import get_prompt, PromptType

        # Get base prompt based on cognitive type
        base_prompt = self.config.get("prompt", get_prompt(PromptType.STANDARD))
        analytical_prompt = (
            f"{base_prompt} However, I should approach this analytically, "
            f"using data-driven reasoning and logical inference. I should consider "
            f"evidence, logic, and systematic analysis."
        )

        # Create a temporary OllamaAgent with the enhanced prompt
        model_name = self.config.get("model", "phi3:mini")
        ollama_agent = OllamaAgent(
            agent_id=self.agent_id,
            config={"prompt": analytical_prompt, "model": model_name}
        )
        return ollama_agent.execute(task)


class CollaborativeAgent(AdvancedAgent):
    """Agent with collaborative thinking approach - focuses on integrating multiple perspectives."""

    def execute(self, task: Task) -> str:
        """Execute task with collaborative approach."""
        from .prompts import get_prompt, PromptType

        # Get base prompt based on cognitive type
        base_prompt = self.config.get("prompt", get_prompt(PromptType.STANDARD))
        collaborative_prompt = (
            f"{base_prompt} However, I should approach this collaboratively, "
            f"considering multiple perspectives and working toward consensus. I should "
            f"integrate various viewpoints and facilitate collective problem-solving."
        )

        # Create a temporary OllamaAgent with the enhanced prompt
        model_name = self.config.get("model", "phi3:mini")
        ollama_agent = OllamaAgent(
            agent_id=self.agent_id,
            config={"prompt": collaborative_prompt, "model": model_name}
        )
        return ollama_agent.execute(task)


class ScepticalAgent(AdvancedAgent):
    """Agent with sceptical thinking approach - always questions assumptions."""

    def execute(self, task: Task) -> str:
        """Execute task with sceptical approach."""
        from .prompts import get_prompt, PromptType

        # Get base prompt based on cognitive type
        base_prompt = self.config.get("prompt", get_prompt(PromptType.CRITICAL))
        sceptical_prompt = (
            f"{base_prompt} However, I should be even more sceptical, "
            f"questioning all assumptions, premises, and potential biases. I should "
            f"thoroughly examine the validity of the given information."
        )

        # Create a temporary OllamaAgent with the enhanced prompt
        model_name = self.config.get("model", "phi3:mini")
        ollama_agent = OllamaAgent(
            agent_id=self.agent_id,
            config={"prompt": sceptical_prompt, "model": model_name}
        )
        return ollama_agent.execute(task)


class EmpatheticAgent(AdvancedAgent):
    """Agent with empathetic thinking approach - focuses on human factors."""

    def execute(self, task: Task) -> str:
        """Execute task with empathetic approach."""
        from .prompts import get_prompt, PromptType

        # Get base prompt based on cognitive type
        base_prompt = self.config.get("prompt", get_prompt(PromptType.STANDARD))
        empathetic_prompt = (
            f"{base_prompt} However, I should approach this with empathy, "
            f"considering the human factors, emotional aspects, and impact on "
            f"individuals involved in the situation."
        )

        # Create a temporary OllamaAgent with the enhanced prompt
        model_name = self.config.get("model", "phi3:mini")
        ollama_agent = OllamaAgent(
            agent_id=self.agent_id,
            config={"prompt": empathetic_prompt, "model": model_name}
        )
        return ollama_agent.execute(task)


class SystemicAgent(AdvancedAgent):
    """Agent with systemic thinking approach - considers system-wide effects."""

    def execute(self, task: Task) -> str:
        """Execute task with systemic approach."""
        from .prompts import get_prompt, PromptType

        # Get base prompt based on cognitive type
        base_prompt = self.config.get("prompt", get_prompt(PromptType.STANDARD))
        systemic_prompt = (
            f"{base_prompt} However, I should approach this systemically, "
            f"considering the broader system implications, interconnections, and "
            f"long-term effects on the overall system."
        )

        # Create a temporary OllamaAgent with the enhanced prompt
        model_name = self.config.get("model", "phi3:mini")
        ollama_agent = OllamaAgent(
            agent_id=self.agent_id,
            config={"prompt": systemic_prompt, "model": model_name}
        )
        return ollama_agent.execute(task)


class PragmaticAgent(AdvancedAgent):
    """Agent with pragmatic thinking approach - focuses on practical solutions."""

    def execute(self, task: Task) -> str:
        """Execute task with pragmatic approach."""
        from .prompts import get_prompt, PromptType

        # Get base prompt based on cognitive type
        base_prompt = self.config.get("prompt", get_prompt(PromptType.STANDARD))
        pragmatic_prompt = (
            f"{base_prompt} However, I should approach this pragmatically, "
            f"focusing on practical, implementable solutions that work in real-world "
            f"conditions with available resources."
        )

        # Create a temporary OllamaAgent with the enhanced prompt
        model_name = self.config.get("model", "phi3:mini")
        ollama_agent = OllamaAgent(
            agent_id=self.agent_id,
            config={"prompt": pragmatic_prompt, "model": model_name}
        )
        return ollama_agent.execute(task)


# Factory function to create agents of different types
def create_agent(agent_id: str, cognitive_type: str, config: Dict[str, Any] = None) -> AdvancedAgent:
    """
    Factory function to create agents of different cognitive types.
    """
    if config is None:
        config = {}

    # Ensure cognitive type is in config
    config["cognitive_type"] = cognitive_type

    agent_classes = {
        "critical": lambda ai, cfg: AdvancedAgent(ai, cfg),  # For now, basic advanced agent
        "standard": lambda ai, cfg: AdvancedAgent(ai, cfg),  # For now, basic advanced agent
        "awakened": lambda ai, cfg: AdvancedAgent(ai, cfg),  # For now, basic advanced agent
        "innovative": lambda ai, cfg: InnovativeAgent(ai, cfg),
        "analytical": lambda ai, cfg: AnalyticalAgent(ai, cfg),
        "collaborative": lambda ai, cfg: CollaborativeAgent(ai, cfg),
        "sceptical": lambda ai, cfg: ScepticalAgent(ai, cfg),
        "empathetic": lambda ai, cfg: EmpatheticAgent(ai, cfg),
        "systemic": lambda ai, cfg: SystemicAgent(ai, cfg),
        "pragmatic": lambda ai, cfg: PragmaticAgent(ai, cfg)
    }

    if cognitive_type in agent_classes:
        return agent_classes[cognitive_type](agent_id, config)
    else:
        # Default to standard agent if type is not recognized
        config["cognitive_type"] = "standard"
        return AdvancedAgent(agent_id, config)