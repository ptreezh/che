"""
Distinctiveness Metrics Calculator for Cognitive Heterogeneity Validation

This module provides functionality for calculating distinctiveness metrics
in cognitive heterogeneity validation experiments, measuring how unique
different cognitive approaches are compared to standard responses.

Authors: CHE Research Team
Date: 2025-10-20
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class DistinctivenessMetrics:
    """
    Data structure representing distinctiveness metrics.
    
    This class encapsulates various metrics that measure how distinct
    different cognitive approaches are compared to standard responses.
    """
    
    # Unique identifier for the metrics
    metrics_id: str
    
    # Distinctiveness scores
    lexical_diversity: float = 0.0  # Lexical diversity of responses
    semantic_diversity: float = 0.0  # Semantic diversity of responses
    syntactic_variety: float = 0.0   # Syntactic variety of responses
    conceptual_uniqueness: float = 0.0  # Conceptual uniqueness of responses
    
    # Comparative metrics
    distinctiveness_index: float = 0.0  # Overall distinctiveness score (0.0-1.0)
    deviation_from_standard: float = 0.0  # Deviation from standard responses
    innovation_score: float = 0.0  # Innovation compared to baseline
    
    # Statistical measures
    sample_size: int = 0
    confidence_interval: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    p_value: float = 0.0
    
    # Metadata
    created_at: str = ""
    last_updated: str = ""
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.metrics_id:
            raise ValueError("Metrics ID cannot be empty")
        
        # Validate score ranges
        scores = [
            self.lexical_diversity, self.semantic_diversity, self.syntactic_variety,
            self.conceptual_uniqueness, self.distinctiveness_index, self.innovation_score
        ]
        
        for score in scores:
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"All scores must be between 0.0 and 1.0, got {score}")
        
        if self.sample_size < 0:
            raise ValueError("Sample size cannot be negative")
        
        if not (0.0 <= self.p_value <= 1.0):
            raise ValueError(f"P-value must be between 0.0 and 1.0, got {self.p_value}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metrics to dictionary representation.
        
        Returns:
            Dictionary containing all metrics attributes
        """
        return {
            'metrics_id': self.metrics_id,
            'lexical_diversity': self.lexical_diversity,
            'semantic_diversity': self.semantic_diversity,
            'syntactic_variety': self.syntactic_variety,
            'conceptual_uniqueness': self.conceptual_uniqueness,
            'distinctiveness_index': self.distinctiveness_index,
            'deviation_from_standard': self.deviation_from_standard,
            'innovation_score': self.innovation_score,
            'sample_size': self.sample_size,
            'confidence_interval': self.confidence_interval,
            'p_value': self.p_value,
            'created_at': self.created_at,
            'last_updated': self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistinctivenessMetrics':
        """
        Create metrics from dictionary representation.
        
        Args:
            data: Dictionary containing metrics attributes
            
        Returns:
            New distinctiveness metrics instance
        """
        return cls(
            metrics_id=data.get('metrics_id', ''),
            lexical_diversity=data.get('lexical_diversity', 0.0),
            semantic_diversity=data.get('semantic_diversity', 0.0),
            syntactic_variety=data.get('syntactic_variety', 0.0),
            conceptual_uniqueness=data.get('conceptual_uniqueness', 0.0),
            distinctiveness_index=data.get('distinctiveness_index', 0.0),
            deviation_from_standard=data.get('deviation_from_standard', 0.0),
            innovation_score=data.get('innovation_score', 0.0),
            sample_size=data.get('sample_size', 0),
            confidence_interval=data.get('confidence_interval', (0.0, 0.0)),
            p_value=data.get('p_value', 0.0),
            created_at=data.get('created_at', ''),
            last_updated=data.get('last_updated', '')
        )


class DistinctivenessCalculator:
    """
    Calculator for distinctiveness metrics in cognitive heterogeneity experiments.
    
    This class provides methods for calculating various distinctiveness metrics
    that measure how unique different cognitive approaches are compared to
    standard responses.
    """
    
    def __init__(self):
        """Initialize the distinctiveness calculator."""
        logger.info("Initialized DistinctivenessCalculator")
    
    def calculate_lexical_diversity(self, responses: List[str]) -> float:
        """
        Calculate lexical diversity of responses.
        
        Lexical diversity measures the variety of words used in responses.
        
        Args:
            responses: List of agent responses
            
        Returns:
            Lexical diversity score between 0.0 (no diversity) and 1.0 (maximum diversity)
        """
        if not responses:
            return 0.0
        
        # Flatten all responses into a single text
        all_text = " ".join(responses)
        
        # Tokenize into words
        words = all_text.lower().split()
        
        if not words:
            return 0.0
        
        # Calculate type-token ratio (TTR)
        unique_words = set(words)
        total_words = len(words)
        
        # TTR ranges from 0 (no diversity) to 1 (maximum diversity)
        ttr = len(unique_words) / total_words if total_words > 0 else 0.0
        
        # Normalize to 0.0-1.0 range
        # TTR typically ranges from ~0.3 to ~0.8 for natural language
        # We'll normalize this to 0.0-1.0 range
        normalized_ttr = min(1.0, max(0.0, (ttr - 0.3) / 0.5)) if total_words > 0 else 0.0
        
        logger.debug(f"Calculated lexical diversity: TTR={ttr:.3f}, Normalized={normalized_ttr:.3f}")
        return normalized_ttr
    
    def calculate_semantic_diversity(self, responses: List[str]) -> float:
        """
        Calculate semantic diversity of responses.
        
        Semantic diversity measures the variety of meanings/concepts expressed.
        
        Args:
            responses: List of agent responses
            
        Returns:
            Semantic diversity score between 0.0 (no diversity) and 1.0 (maximum diversity)
        """
        if not responses:
            return 0.0
        
        # Extract semantic concepts from responses
        semantic_concepts = []
        for response in responses:
            concepts = self._extract_semantic_concepts(response)
            semantic_concepts.extend(concepts)
        
        if not semantic_concepts:
            return 0.0
        
        # Calculate concept diversity using Simpson's index
        concept_counts = Counter(semantic_concepts)
        total_concepts = len(semantic_concepts)
        
        # Calculate sum of squared proportions
        sum_squared_proportions = sum((count / total_concepts) ** 2 
                                   for count in concept_counts.values())
        
        # Convert to diversity index (higher values indicate higher diversity)
        diversity_index = 1.0 - sum_squared_proportions
        
        # Normalize to 0.0-1.0 range
        max_possible_diversity = 1.0 - (1.0 / total_concepts) if total_concepts > 1 else 0.0
        if max_possible_diversity > 0:
            normalized_diversity = diversity_index / max_possible_diversity
        else:
            normalized_diversity = 0.0
        
        logger.debug(f"Calculated semantic diversity: Index={diversity_index:.3f}, Normalized={normalized_diversity:.3f}")
        return min(1.0, max(0.0, normalized_diversity))
    
    def _extract_semantic_concepts(self, response: str) -> List[str]:
        """
        Extract semantic concepts from a response.
        
        Args:
            response: Agent response to analyze
            
        Returns:
            List of semantic concepts found in the response
        """
        # This is a simplified implementation
        # In a real system, this would use NLP techniques or embedding models
        concepts = []
        
        # Extract key semantic categories
        semantic_categories = [
            # Epistemic concepts
            'know', 'understand', 'believe', 'think', 'doubt', 'question',
            
            # Logical concepts
            'reason', 'logic', 'argument', 'proof', 'evidence', 'verify',
            
            # Critical thinking concepts
            'analyze', 'evaluate', 'assess', 'critique', 'challenge', 'refute',
            
            # Cognitive concepts
            'mind', 'brain', 'thought', 'consciousness', 'awareness', 'intelligence',
            
            # Emotional concepts
            'feel', 'emotion', 'sense', 'intuition', 'gut', 'instinct',
            
            # Social concepts
            'group', 'team', 'collective', 'community', 'society', 'culture',
            
            # Temporal concepts
            'time', 'future', 'past', 'present', 'change', 'evolve',
            
            # Spatial concepts
            'space', 'place', 'location', 'direction', 'distance', 'size',
            
            # Causal concepts
            'cause', 'effect', 'result', 'consequence', 'impact', 'influence',
            
            # Modal concepts
            'can', 'could', 'may', 'might', 'should', 'would', 'must',
        ]
        
        # Check each category
        response_lower = response.lower()
        for category in semantic_categories:
            if category in response_lower:
                concepts.append(category)
        
        return concepts
    
    def calculate_syntactic_variety(self, responses: List[str]) -> float:
        """
        Calculate syntactic variety of responses.
        
        Syntactic variety measures the diversity of sentence structures used.
        
        Args:
            responses: List of agent responses
            
        Returns:
            Syntactic variety score between 0.0 (no variety) and 1.0 (maximum variety)
        """
        if not responses:
            return 0.0
        
        # Extract sentence structures from responses
        sentence_structures = []
        for response in responses:
            structures = self._extract_sentence_structures(response)
            sentence_structures.extend(structures)
        
        if not sentence_structures:
            return 0.0
        
        # Calculate structure diversity using Simpson's index
        structure_counts = Counter(sentence_structures)
        total_structures = len(sentence_structures)
        
        # Calculate sum of squared proportions
        sum_squared_proportions = sum((count / total_structures) ** 2 
                                   for count in structure_counts.values())
        
        # Convert to diversity index (higher values indicate higher diversity)
        diversity_index = 1.0 - sum_squared_proportions
        
        # Normalize to 0.0-1.0 range
        max_possible_diversity = 1.0 - (1.0 / total_structures) if total_structures > 1 else 0.0
        if max_possible_diversity > 0:
            normalized_diversity = diversity_index / max_possible_diversity
        else:
            normalized_diversity = 0.0
        
        logger.debug(f"Calculated syntactic variety: Index={diversity_index:.3f}, Normalized={normalized_diversity:.3f}")
        return min(1.0, max(0.0, normalized_diversity))
    
    def _extract_sentence_structures(self, response: str) -> List[str]:
        """
        Extract sentence structures from a response.
        
        Args:
            response: Agent response to analyze
            
        Returns:
            List of sentence structures found in the response
        """
        # This is a simplified implementation
        # In a real system, this would use NLP parsing techniques
        structures = []
        
        # Simple sentence structure patterns
        structure_patterns = [
            # Simple sentences
            r'^[A-Z][^.!?]*[.!?]$',  # Subject + predicate
            
            # Complex sentences
            r'.*[,.].*[.!?]$',  # Contains comma or semicolon
            
            # Compound sentences
            r'.*(and|or|but).*[.!?]$',  # Contains conjunctions
            
            # Questions
            r'^[Ww]hat|[Hh]ow|[Ww]hy|[Ww]hen|[Ww]here|[Ww]ho.*\?$',  # Question words
            
            # Exclamations
            r'.*!$',  # Ends with exclamation mark
        ]
        
        # Split into sentences
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        # Check each sentence against patterns
        for sentence in sentences:
            for i, pattern in enumerate(structure_patterns):
                if sentence and any(word in sentence.lower() for word in ['what', 'how', 'why', 'when', 'where', 'who']):
                    structures.append(f"question_{i}")
                elif '!' in sentence:
                    structures.append(f"exclamation_{i}")
                elif ',' in sentence or ';' in sentence:
                    structures.append(f"complex_{i}")
                elif 'and' in sentence.lower() or 'or' in sentence.lower() or 'but' in sentence.lower():
                    structures.append(f"compound_{i}")
                else:
                    structures.append(f"simple_{i}")
        
        return structures
    
    def calculate_conceptual_uniqueness(self, responses: List[str], 
                                     standard_responses: List[str]) -> float:
        """
        Calculate conceptual uniqueness of responses compared to standard responses.
        
        Conceptual uniqueness measures how distinct responses are from standard approaches.
        
        Args:
            responses: List of agent responses
            standard_responses: List of standard agent responses for comparison
            
        Returns:
            Conceptual uniqueness score between 0.0 (not unique) and 1.0 (completely unique)
        """
        if not responses or not standard_responses:
            return 0.0
        
        # Extract concepts from both response sets
        agent_concepts = []
        for response in responses:
            concepts = self._extract_semantic_concepts(response)
            agent_concepts.extend(concepts)
        
        standard_concepts = []
        for response in standard_responses:
            concepts = self._extract_semantic_concepts(response)
            standard_concepts.extend(concepts)
        
        if not agent_concepts or not standard_concepts:
            return 0.0
        
        # Calculate Jaccard similarity between concept sets
        agent_set = set(agent_concepts)
        standard_set = set(standard_concepts)
        
        intersection = len(agent_set.intersection(standard_set))
        union = len(agent_set.union(standard_set))
        
        if union == 0:
            return 0.0
        
        jaccard_similarity = intersection / union
        
        # Convert similarity to uniqueness (1 - similarity)
        conceptual_uniqueness = 1.0 - jaccard_similarity
        
        logger.debug(f"Calculated conceptual uniqueness: Similarity={jaccard_similarity:.3f}, Uniqueness={conceptual_uniqueness:.3f}")
        return conceptual_uniqueness
    
    def calculate_distinctiveness_index(self, responses: List[str], 
                                     standard_responses: List[str]) -> float:
        """
        Calculate overall distinctiveness index.
        
        This combines multiple distinctiveness metrics into a single score.
        
        Args:
            responses: List of agent responses
            standard_responses: List of standard agent responses for comparison
            
        Returns:
            Overall distinctiveness index between 0.0 (not distinctive) and 1.0 (highly distinctive)
        """
        if not responses or not standard_responses:
            return 0.0
        
        # Calculate individual metrics
        lexical_diversity = self.calculate_lexical_diversity(responses)
        semantic_diversity = self.calculate_semantic_diversity(responses)
        syntactic_variety = self.calculate_syntactic_variety(responses)
        conceptual_uniqueness = self.calculate_conceptual_uniqueness(responses, standard_responses)
        
        # Weighted average (equal weights for all metrics)
        distinctiveness_index = (
            lexical_diversity * 0.25 +
            semantic_diversity * 0.25 +
            syntactic_variety * 0.25 +
            conceptual_uniqueness * 0.25
        )
        
        logger.debug(f"Calculated distinctiveness index: {distinctiveness_index:.3f}")
        return distinctiveness_index
    
    def calculate_deviation_from_standard(self, responses: List[str], 
                                       standard_responses: List[str]) -> float:
        """
        Calculate deviation from standard responses.
        
        Args:
            responses: List of agent responses
            standard_responses: List of standard agent responses for comparison
            
        Returns:
            Deviation score between 0.0 (no deviation) and 1.0 (maximum deviation)
        """
        if not responses or not standard_responses:
            return 0.0
        
        # Calculate conceptual uniqueness as deviation measure
        return self.calculate_conceptual_uniqueness(responses, standard_responses)
    
    def calculate_innovation_score(self, responses: List[str], 
                                standard_responses: List[str],
                                baseline_responses: Optional[List[str]] = None) -> float:
        """
        Calculate innovation score compared to baseline.
        
        Args:
            responses: List of agent responses
            standard_responses: List of standard agent responses for comparison
            baseline_responses: Optional list of baseline responses (defaults to standard_responses)
            
        Returns:
            Innovation score between 0.0 (no innovation) and 1.0 (maximum innovation)
        """
        if not responses:
            return 0.0
        
        if baseline_responses is None:
            baseline_responses = standard_responses
        
        if not baseline_responses:
            return 0.0
        
        # Innovation is measured as deviation from baseline plus uniqueness
        deviation = self.calculate_conceptual_uniqueness(responses, baseline_responses)
        uniqueness = self.calculate_distinctiveness_index(responses, standard_responses)
        
        # Weighted combination
        innovation_score = 0.6 * deviation + 0.4 * uniqueness
        
        logger.debug(f"Calculated innovation score: {innovation_score:.3f}")
        return innovation_score
    
    def calculate_all_metrics(self, responses: List[str], 
                           standard_responses: List[str],
                           baseline_responses: Optional[List[str]] = None) -> DistinctivenessMetrics:
        """
        Calculate all distinctiveness metrics.
        
        Args:
            responses: List of agent responses
            standard_responses: List of standard agent responses for comparison
            baseline_responses: Optional list of baseline responses (defaults to standard_responses)
            
        Returns:
            Distinctiveness metrics object containing all calculated metrics
        """
        if baseline_responses is None:
            baseline_responses = standard_responses
        
        # Calculate individual metrics
        lexical_diversity = self.calculate_lexical_diversity(responses)
        semantic_diversity = self.calculate_semantic_diversity(responses)
        syntactic_variety = self.calculate_syntactic_variety(responses)
        conceptual_uniqueness = self.calculate_conceptual_uniqueness(responses, standard_responses)
        distinctiveness_index = self.calculate_distinctiveness_index(responses, standard_responses)
        deviation_from_standard = self.calculate_deviation_from_standard(responses, standard_responses)
        innovation_score = self.calculate_innovation_score(responses, standard_responses, baseline_responses)
        
        # Create metrics object
        metrics = DistinctivenessMetrics(
            metrics_id=f"distinctiveness_{len(responses)}_responses",
            lexical_diversity=lexical_diversity,
            semantic_diversity=semantic_diversity,
            syntactic_variety=syntactic_variety,
            conceptual_uniqueness=conceptual_uniqueness,
            distinctiveness_index=distinctiveness_index,
            deviation_from_standard=deviation_from_standard,
            innovation_score=innovation_score,
            sample_size=len(responses)
        )
        
        logger.info(f"Calculated all distinctiveness metrics for {len(responses)} responses")
        return metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert calculator to dictionary representation.
        
        Returns:
            Dictionary containing calculator configuration
        """
        return {}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistinctivenessCalculator':
        """
        Create calculator from dictionary representation.
        
        Args:
            data: Dictionary containing calculator configuration
            
        Returns:
            New distinctiveness calculator instance
        """
        return cls()