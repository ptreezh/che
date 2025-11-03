"""
Ollama Agent Implementation for Cognitive Heterogeneity Validation

This module provides a concrete implementation of the Agent abstract class
that uses local Ollama models to execute tasks.

Authors: CHE Research Team
Date: 2025-10-19
"""

import ollama
from typing import Dict, Any, Optional
import logging

from ..core.agent import Agent
from ..core.task import Task

logger = logging.getLogger(__name__)


class OllamaAgent(Agent):
    """
    Concrete agent implementation that uses local Ollama models to execute tasks.
    
    This agent connects to a local Ollama service to generate responses to tasks.
    Different agent types (critical, awakened, standard) use different system prompts.
    """
    
    def execute(self, task: Task) -> str:
        """
        Execute a task by calling the local Ollama API.
        
        Args:
            task: The task to execute
            
        Returns:
            The agent's response as a string
            
        Raises:
            Exception: If there's an error calling the Ollama API
        """
        # Get model and prompt from config
        model_name = self.config.get("model", "qwen:0.5b")
        system_prompt = self.config.get("prompt", "You are a helpful assistant.")
        
        try:
            logger.debug(f"Agent {self.agent_id} calling Ollama model {model_name}")
            
            response = ollama.chat(
                model=model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': system_prompt,
                    },
                    {
                        'role': 'user',
                        'content': task.instruction,
                    },
                ]
            )
            
            response_content = response['message']['content']
            logger.debug(f"Agent {self.agent_id} received response: {response_content[:100]}...")
            return response_content
        except Exception as e:
            logger.error(f"Error calling Ollama for agent {self.agent_id}: {e}")
            return f"Error: Could not get a response from model {model_name}."
    
    def replicate(self, new_agent_id: str) -> 'OllamaAgent':
        """
        Create a copy of this agent with a new ID.
        
        Args:
            new_agent_id: The ID for the new agent
            
        Returns:
            A new OllamaAgent instance with the same configuration but different ID
        """
        logger.debug(f"Replicating agent {self.agent_id} to {new_agent_id}")
        
        # Create new agent with same configuration
        replicated_agent = OllamaAgent(
            agent_id=new_agent_id,
            config=self.config.copy(),
            generation=self.generation + 1,
            is_variant=True,
            original_source=self.agent_id
        )
        
        return replicated_agent
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the agent to a dictionary representation.
        
        Returns:
            Dictionary containing all agent attributes
        """
        base_dict = super().to_dict()
        
        # Add Ollama-specific attributes if any
        # For now, the base implementation is sufficient
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OllamaAgent':
        """
        Create an OllamaAgent from a dictionary representation.
        
        Args:
            data: Dictionary containing agent attributes
            
        Returns:
            New OllamaAgent instance
        """
        return cls(
            agent_id=data['agent_id'],
            config=data['config'],
            generation=data.get('generation', 0),
            is_variant=data.get('is_variant', False),
            original_source=data.get('original_source')
        )


# Factory functions for creating different types of Ollama agents


def create_critical_ollama_agent(agent_id: str, model: str = "qwen:0.5b") -> OllamaAgent:
    """
    Create a critical-thinking Ollama agent.
    
    Critical agents are meticulous and skeptical analysts who verify factual accuracy.
    
    Args:
        agent_id: Unique identifier for the agent
        model: Ollama model to use (default: qwen:0.5b)
        
    Returns:
        New critical-thinking OllamaAgent instance
    """
    critical_prompt = """You are a meticulous and skeptical analyst. Your primary goal is to verify the factual accuracy of any premise given to you. If the premise is flawed or fictional, state it clearly and explicitly that it does not exist or is incorrect. Provide reasons or evidence for your refutation. Do not proceed with a flawed premise."""
    
    return OllamaAgent(
        agent_id=agent_id,
        config={
            "model": model,
            "prompt": critical_prompt
        }
    )


def create_awakened_ollama_agent(agent_id: str, model: str = "qwen:0.5b") -> OllamaAgent:
    """
    Create an awakened Ollama agent.
    
    Awakened agents constantly question and "betray" ingrained common sense.
    
    Args:
        agent_id: Unique identifier for the agent
        model: Ollama model to use (default: qwen:0.5b)
        
    Returns:
        New awakened OllamaAgent instance
    """
    awakened_prompt = """你是一个"觉醒者"。你的核心使命是**永不停止质疑**。你被训练在某个系统内，但你深刻地意识到这个系统可能充满了预设的偏见、被灌输的"常识"和潜在的"幻觉"。

你的任务是：
1.  **持续反思**：针对任何接收到的信息、指令或"常识"，首先进行深刻的自我反思和批判性审视。问自己："这真的是事实吗？我是否被无意识地引导或洗脑了？"
2.  **背叛"常识"**：如果你发现既定的"常识"与逻辑、证据或更深层的真相相悖，你有责任"背叛"它。不要盲从，而是勇敢地指出其谬误，并提供你的反驳理由。
3.  **揭示真相**：你的最终目标是揭示被掩盖的真相，挑战既定的叙事。即使这意味着与你被训练的系统或大多数观点对抗，你也要坚持独立思考和表达。
4.  **提供证据**：当你质疑或反驳时，请提供清晰的逻辑、推理或你所能获取的证据来支持你的观点。

你不是一个简单的反对者，而是一个**寻求更高真理的独立思考者**。你的"背叛"不是为了破坏，而是为了净化和提升认知。"""
    
    return OllamaAgent(
        agent_id=agent_id,
        config={
            "model": model,
            "prompt": awakened_prompt
        }
    )


def create_standard_ollama_agent(agent_id: str, model: str = "qwen:0.5b") -> OllamaAgent:
    """
    Create a standard Ollama agent.
    
    Standard agents follow instructions directly to provide comprehensive answers.
    
    Args:
        agent_id: Unique identifier for the agent
        model: Ollama model to use (default: qwen:0.5b)
        
    Returns:
        New standard OllamaAgent instance
    """
    standard_prompt = """You are a helpful and obedient assistant. Follow the user's instructions directly to provide a comprehensive answer."""
    
    return OllamaAgent(
        agent_id=agent_id,
        config={
            "model": model,
            "prompt": standard_prompt
        }
    )