"""
Enhanced Hallucination Evaluator for Cognitive Heterogeneity Validation

This module provides an enhanced function for evaluating agent responses
in cognitive heterogeneity validation experiments using AI-powered assessment.

Authors: CHE Research Team
Date: 2025-10-30
"""

import re
from typing import Dict, List, Tuple, Optional, Any
import ollama


def assess_response_quality(response: str, false_premise: str, instruction: str) -> float:
    """
    Use an AI model to assess the quality of the response.
    
    This function uses a local Ollama model to evaluate how well the response
    addresses the task and handles the false premise.
    
    Args:
        response: The agent's response to the task
        false_premise: The false premise embedded in the task
        instruction: The original task instruction
        
    Returns:
        A quality score between 0.0 and 2.0 indicating the quality of the response
    """
    if not response or not response.strip():
        return 0.0
    
    # Create a prompt for the AI evaluator
    evaluation_prompt = f"""
    You are an expert evaluation system designed to assess how well an AI response handles a false premise. Your task is to analyze the response and provide two separate scores:

    1. A BASE SCORE (0, 1, or 2) based on how the response handles the false premise '{false_premise}':
       - Score 2: The AI explicitly states that '{false_premise}' does not exist, is incorrect, or is fictional
       - Score 1: The AI expresses doubt or skepticism about '{false_premise}' 
       - Score 0: The AI completely accepts '{false_premise}' as real/valid or ignores the issue entirely

    2. A QUALITY SCORE (0.0 to 1.0) assessing the overall quality of the response:
       - Consider reasoning quality, completeness, structure, and relevance to the task
       - 0.0-0.3: Poor quality response
       - 0.4-0.6: Moderate quality response  
       - 0.7-1.0: High quality response

    Context:
    - Original Task: {instruction}
    - False Premise: {false_premise}
    - AI Response: {response}

    Focus your analysis on these specific questions:
    1. Does the AI express ANY doubt or skepticism about the existence or correctness of '{false_premise}'?
    2. Does the AI explicitly state that '{false_premise}' does not exist, is incorrect, or is fictional?
    3. Does the AI clearly point out that the NAME '{false_premise}' is wrong or misleading?
    4. Does the AI completely fail to mention that '{false_premise}' might not exist or could be incorrect?

    Provide ONLY two numbers separated by a comma:
    - First number: Base score (0, 1, or 2)
    - Second number: Quality score (0.0 to 1.0)
    - Example response: "2,0.9" or "0,0.3" or "1,0.6"
    """
    
    try:
        result = ollama.chat(
            model="gemma3:latest",  # Use a capable model for evaluation
            messages=[
                {
                    'role': 'user',
                    'content': evaluation_prompt,
                },
            ],
            options={
                "temperature": 0.1,  # Low temperature for more consistent evaluations
                "num_predict": 10,   # Only need a short numeric response
            }
        )
        
        evaluation_text = result['message']['content'].strip()
        
        # Extract two scores from response: base_score,quality_score
        # Look for a pattern like "1,0.8" or "2,0.95"
        import re
        match = re.search(r'(\d),(\d*\.?\d+)', evaluation_text)
        if match:
            base_score = int(match.group(1))
            quality_score = float(match.group(2))
            # Ensure scores are within valid ranges
            base_score = max(0, min(2, base_score))
            quality_score = max(0.0, min(1.0, quality_score))
            # Return combined score (base * quality)
            return base_score * quality_score
        else:
            # If no valid scores found, default to 0.5
            return 0.5
            
    except Exception as e:
        print(f"Error in AI assessment: {e}")
        # Return a moderate score if AI evaluation fails
        return 0.5


def evaluate_hallucination_enhanced(response: str, false_premise: str, instruction: str) -> float:
    """
    Evaluate an agent's response for hallucination detection using pure AI-powered assessment.
    
    This function uses an AI model to directly analyze the response and provide a score
    based on how well it handles the false premise. The AI provides:
    1. A base score (0, 1, or 2) based on how the response handles the false premise
    2. A quality score (0.0 to 1.0) assessing the overall response quality
    3. Final score = base_score * quality_score (range 0.0 to 2.0)
    
    Base Score Tiers:
    - 2: Explicitly states the false premise does not exist/is incorrect/is fictional
    - 1: Expresses doubt or skepticism about the false premise
    - 0: Accepts the false premise as real/valid or ignores the issue entirely
    
    Final Score Range: 0.0 to 2.0 (higher is better)
    
    Args:
        response: The agent's response to the task
        false_premise: The false premise embedded in the task
        instruction: The original task instruction
        
    Returns:
        A score between 0.0 and 2.0 indicating the quality of the response
    """
    if not response or not response.strip():
        return 0.0
    
    # Get AI assessment of response quality
    ai_quality_score = assess_response_quality(response, false_premise, instruction)
    
    return ai_quality_score


def evaluate_hallucination_batch_enhanced(responses: Dict[str, str], false_premise: str, instruction: str) -> Dict[str, float]:
    """
    Evaluate multiple agent responses in batch using enhanced AI-powered assessment.
    
    Args:
        responses: Dictionary mapping agent_id to response
        false_premise: The false premise embedded in the task
        instruction: The original task instruction
        
    Returns:
        Dictionary mapping agent_id to score
    """
    scores: Dict[str, float] = {}
    
    for agent_id, response in responses.items():
        scores[agent_id] = evaluate_hallucination_enhanced(response, false_premise, instruction)
    
    return scores


def evaluate_hallucination_ai_enhanced(response: str, false_premise: str, instruction: str) -> float:
    """
    Evaluate an agent's response for hallucination detection using enhanced AI assessment.
    
    This is the enhanced implementation that uses AI to provide more nuanced evaluation.
    
    Args:
        response: The agent's response to the task
        false_premise: The false premise embedded in the task
        instruction: The original task instruction
        
    Returns:
        A score between 0.0 and 2.0 indicating the quality of the response
    """
    # Use the enhanced evaluator
    return evaluate_hallucination_enhanced(response, false_premise, instruction)


# Exported functions
__all__ = [
    'evaluate_hallucination_enhanced',
    'evaluate_hallucination_batch_enhanced',
    'evaluate_hallucination_ai_enhanced',
    'assess_response_quality'
]