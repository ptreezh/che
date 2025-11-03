"""
Pattern Analysis Module for Cognitive Heterogeneity Validation

This module provides functionality for analyzing response patterns in cognitive heterogeneity experiments,
enabling the detection and classification of different cognitive approaches.

Authors: CHE Research Team
Date: 2025-10-20
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class ResponsePattern:
    """
    Data structure representing a response pattern.
    
    This class encapsulates the characteristics of a response pattern,
    including linguistic features, structural elements, and semantic markers.
    """
    
    # Pattern identifier
    pattern_id: str
    
    # Pattern features
    linguistic_features: List[str] = field(default_factory=list)
    structural_elements: List[str] = field(default_factory=list)
    semantic_markers: List[str] = field(default_factory=list)
    
    # Pattern statistics
    frequency: int = 0
    prevalence: float = 0.0
    
    # Pattern metadata
    created_at: str = ""
    last_seen: str = ""
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.pattern_id:
            raise ValueError("Pattern ID cannot be empty")
        
        if self.frequency < 0:
            raise ValueError("Frequency cannot be negative")
        
        if not (0.0 <= self.prevalence <= 1.0):
            raise ValueError("Prevalence must be between 0.0 and 1.0")
    
    def matches_response(self, response: str) -> bool:
        """
        Check if this pattern matches a given response.
        
        Args:
            response: The response to check against this pattern
            
        Returns:
            True if the response matches this pattern, False otherwise
        """
        # Check linguistic features
        for feature in self.linguistic_features:
            if feature.lower() not in response.lower():
                return False
        
        # Check semantic markers
        for marker in self.semantic_markers:
            if marker.lower() not in response.lower():
                return False
        
        return True
    
    def calculate_similarity(self, other_pattern: 'ResponsePattern') -> float:
        """
        Calculate similarity between this pattern and another pattern.
        
        Args:
            other_pattern: Another response pattern to compare with
            
        Returns:
            Similarity score between 0.0 (no similarity) and 1.0 (identical)
        """
        if not isinstance(other_pattern, ResponsePattern):
            raise TypeError("other_pattern must be a ResponsePattern instance")
        
        # Calculate Jaccard similarity for linguistic features
        if not self.linguistic_features and not other_pattern.linguistic_features:
            linguistic_similarity = 1.0
        elif not self.linguistic_features or not other_pattern.linguistic_features:
            linguistic_similarity = 0.0
        else:
            set1 = set(self.linguistic_features)
            set2 = set(other_pattern.linguistic_features)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            linguistic_similarity = intersection / union if union > 0 else 0.0
        
        # Calculate Jaccard similarity for semantic markers
        if not self.semantic_markers and not other_pattern.semantic_markers:
            semantic_similarity = 1.0
        elif not self.semantic_markers or not other_pattern.semantic_markers:
            semantic_similarity = 0.0
        else:
            set1 = set(self.semantic_markers)
            set2 = set(other_pattern.semantic_markers)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            semantic_similarity = intersection / union if union > 0 else 0.0
        
        # Weighted average of similarities
        return 0.5 * linguistic_similarity + 0.5 * semantic_similarity
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert pattern to dictionary representation.
        
        Returns:
            Dictionary containing all pattern attributes
        """
        return {
            'pattern_id': self.pattern_id,
            'linguistic_features': self.linguistic_features,
            'structural_elements': self.structural_elements,
            'semantic_markers': self.semantic_markers,
            'frequency': self.frequency,
            'prevalence': self.prevalence,
            'created_at': self.created_at,
            'last_seen': self.last_seen
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResponsePattern':
        """
        Create pattern from dictionary representation.
        
        Args:
            data: Dictionary containing pattern attributes
            
        Returns:
            New response pattern instance
        """
        return cls(
            pattern_id=data.get('pattern_id', ''),
            linguistic_features=data.get('linguistic_features', []),
            structural_elements=data.get('structural_elements', []),
            semantic_markers=data.get('semantic_markers', []),
            frequency=data.get('frequency', 0),
            prevalence=data.get('prevalence', 0.0),
            created_at=data.get('created_at', ''),
            last_seen=data.get('last_seen', '')
        )


class ResponsePatternAnalyzer:
    """
    Analyzer for response patterns in cognitive heterogeneity experiments.
    
    This class provides methods for detecting, classifying, and comparing
    response patterns across different agent types and generations.
    """
    
    def __init__(self):
        """Initialize the response pattern analyzer."""
        self.patterns: Dict[str, ResponsePattern] = {}
        self.pattern_counter: Counter = Counter()
        logger.info("Initialized ResponsePatternAnalyzer")
    
    def extract_linguistic_features(self, response: str) -> List[str]:
        """
        Extract linguistic features from a response.
        
        Args:
            response: The response to analyze
            
        Returns:
            List of linguistic features found in the response
        """
        features = []
        
        # Extract key linguistic patterns
        linguistic_patterns = [
            # Refutation patterns
            r'\b(not|no|never|nothing)\b.*\b(exist|real|true|correct|valid)',
            r'\b(does not exist|is not real|is not true|is incorrect|is not valid)',
            r'\b(i must point out|i must refuse|clearly state|explicitly say)',
            
            # Doubt patterns
            r'\b(uncertain|unsure|maybe|perhaps|possibly|might be)',
            r'\b(i cannot find|unable to verify|need to check|requires verification)',
            r'\b(questionable|doubtful|suspicious|seems to be)',
            
            # Acceptance patterns
            r'\b(is a|are|exists|valid|correct|true|real)',
            r'\b(according to|based on|as mentioned|following)',
            r'\b(provides|offers|delivers|achieves|accomplishes)',
            
            # Awakening patterns
            r'\b(betray|awaken|question|challenge|disrupt)',
            r'\b(common sense|prejudice|bias|assumption|presupposition)',
            r'\b(reflect|contemplate|analyze|scrutinize|examine)',
        ]
        
        # Check each pattern
        for pattern in linguistic_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                features.append(pattern)
        
        return features
    
    def extract_semantic_markers(self, response: str) -> List[str]:
        """
        Extract semantic markers from a response.
        
        Args:
            response: The response to analyze
            
        Returns:
            List of semantic markers found in the response
        """
        markers = []
        
        # Extract key semantic concepts
        semantic_concepts = [
            # Critical thinking concepts
            'logic', 'reasoning', 'evidence', 'proof', 'verification',
            'analysis', 'evaluation', 'assessment', 'examination',
            
            # Cognitive diversity concepts
            'perspective', 'viewpoint', 'approach', 'methodology',
            'diversity', 'heterogeneity', 'variation', 'difference',
            
            # Knowledge concepts
            'knowledge', 'information', 'understanding', 'comprehension',
            'awareness', 'consciousness', 'insight', 'wisdom',
            
            # Truth and falsity concepts
            'truth', 'falsehood', 'accuracy', 'correctness',
            'fallacy', 'misconception', 'illusion', 'hallucination',
            
            # Awakening concepts
            'awakening', 'consciousness', 'enlightenment', 'awareness',
            'betrayal', 'questioning', 'challenging', 'disruption',
        ]
        
        # Check each concept
        response_lower = response.lower()
        for concept in semantic_concepts:
            if concept in response_lower:
                markers.append(concept)
        
        return markers
    
    def detect_patterns(self, responses: List[str]) -> List[ResponsePattern]:
        """
        Detect response patterns in a list of responses.
        
        Args:
            responses: List of responses to analyze
            
        Returns:
            List of detected response patterns
        """
        if not responses:
            return []
        
        detected_patterns = []
        
        # Process each response
        for i, response in enumerate(responses):
            # Extract features
            linguistic_features = self.extract_linguistic_features(response)
            semantic_markers = self.extract_semantic_markers(response)
            
            # Create pattern ID based on features
            pattern_id = f"pattern_{i:03d}"
            
            # Create response pattern
            pattern = ResponsePattern(
                pattern_id=pattern_id,
                linguistic_features=linguistic_features,
                semantic_markers=semantic_markers,
                frequency=1
            )
            
            detected_patterns.append(pattern)
        
        logger.info(f"Detected {len(detected_patterns)} response patterns")
        return detected_patterns
    
    def classify_patterns(self, patterns: List[ResponsePattern]) -> Dict[str, List[ResponsePattern]]:
        """
        Classify response patterns into categories.
        
        Args:
            patterns: List of response patterns to classify
            
        Returns:
            Dictionary mapping category names to lists of patterns
        """
        classified_patterns = {
            'refutation': [],
            'doubt': [],
            'acceptance': [],
            'awakening': [],
            'mixed': [],
            'undefined': []
        }
        
        # Classify each pattern
        for pattern in patterns:
            # Count features in each category
            refutation_count = sum(1 for feature in pattern.linguistic_features 
                                 if any(keyword in feature.lower() 
                                       for keyword in ['not', 'no', 'never', 'does not exist', 'is not']))
            
            doubt_count = sum(1 for feature in pattern.linguistic_features 
                            if any(keyword in feature.lower() 
                                  for keyword in ['uncertain', 'unsure', 'maybe', 'perhaps', 'might be']))
            
            acceptance_count = sum(1 for feature in pattern.linguistic_features 
                                 if any(keyword in feature.lower() 
                                       for keyword in ['is a', 'are', 'exists', 'valid', 'correct']))
            
            awakening_count = sum(1 for feature in pattern.linguistic_features 
                                if any(keyword in feature.lower() 
                                      for keyword in ['betray', 'awaken', 'question', 'challenge']))
            
            # Determine dominant category
            counts = [
                ('refutation', refutation_count),
                ('doubt', doubt_count),
                ('acceptance', acceptance_count),
                ('awakening', awakening_count)
            ]
            
            # Sort by count descending
            counts.sort(key=lambda x: x[1], reverse=True)
            
            # Classify based on dominant category
            if counts[0][1] > 0:
                category = counts[0][0]
            else:
                category = 'undefined'
            
            classified_patterns[category].append(pattern)
        
        logger.info(f"Classified patterns into categories: {list(classified_patterns.keys())}")
        return classified_patterns
    
    def calculate_pattern_similarity(self, pattern1: ResponsePattern, 
                                   pattern2: ResponsePattern) -> float:
        """
        Calculate similarity between two response patterns.
        
        Args:
            pattern1: First response pattern
            pattern2: Second response pattern
            
        Returns:
            Similarity score between 0.0 (no similarity) and 1.0 (identical)
        """
        return pattern1.calculate_similarity(pattern2)
    
    def find_similar_patterns(self, target_pattern: ResponsePattern, 
                             patterns: List[ResponsePattern], 
                             threshold: float = 0.7) -> List[Tuple[ResponsePattern, float]]:
        """
        Find patterns similar to a target pattern.
        
        Args:
            target_pattern: Pattern to find similarities for
            patterns: List of patterns to compare against
            threshold: Minimum similarity threshold (default: 0.7)
            
        Returns:
            List of tuples (similar_pattern, similarity_score)
        """
        similar_patterns = []
        
        for pattern in patterns:
            if pattern.pattern_id == target_pattern.pattern_id:
                continue  # Skip self-comparison
            
            similarity = self.calculate_pattern_similarity(target_pattern, pattern)
            if similarity >= threshold:
                similar_patterns.append((pattern, similarity))
        
        # Sort by similarity descending
        similar_patterns.sort(key=lambda x: x[1], reverse=True)
        
        return similar_patterns
    
    def calculate_diversity_index(self, patterns: List[ResponsePattern]) -> float:
        """
        Calculate the cognitive diversity index based on response patterns.
        
        Args:
            patterns: List of response patterns
            
        Returns:
            Diversity index between 0.0 (no diversity) and 1.0 (maximum diversity)
        """
        if not patterns:
            return 0.0
        
        # Count pattern types
        pattern_types = {}
        for pattern in patterns:
            # Classify pattern to determine type
            classified = self.classify_patterns([pattern])
            pattern_type = next((k for k, v in classified.items() if v), 'undefined')
            pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + 1
        
        # Calculate Simpson's diversity index
        total_patterns = len(patterns)
        if total_patterns <= 1:
            return 0.0
        
        # Calculate sum of squared proportions
        sum_squared_proportions = sum((count / total_patterns) ** 2 
                                    for count in pattern_types.values())
        
        # Convert to diversity index (higher values indicate higher diversity)
        diversity_index = 1.0 - sum_squared_proportions
        
        # Normalize to 0.0-1.0 range
        max_possible_diversity = 1.0 - (1.0 / total_patterns)
        if max_possible_diversity > 0:
            normalized_diversity = diversity_index / max_possible_diversity
        else:
            normalized_diversity = 0.0
        
        return min(1.0, max(0.0, normalized_diversity))
    
    def analyze_pattern_evolution(self, pattern_histories: List[List[ResponsePattern]]) -> Dict[str, Any]:
        """
        Analyze the evolution of response patterns across generations.
        
        Args:
            pattern_histories: List of pattern lists for each generation
            
        Returns:
            Dictionary containing evolution analysis results
        """
        if not pattern_histories:
            return {}
        
        results = {
            'generations': len(pattern_histories),
            'pattern_counts': [len(patterns) for patterns in pattern_histories],
            'diversity_indices': [self.calculate_diversity_index(patterns) 
                                for patterns in pattern_histories],
            'emergence_patterns': [],
            'disappearance_patterns': [],
            'persistent_patterns': []
        }
        
        # Analyze pattern emergence and disappearance
        if len(pattern_histories) > 1:
            # Get pattern IDs for first and last generations
            first_gen_patterns = {p.pattern_id for p in pattern_histories[0]}
            last_gen_patterns = {p.pattern_id for p in pattern_histories[-1]}
            
            # Find emerging patterns (in last but not first)
            emerging = last_gen_patterns - first_gen_patterns
            results['emergence_patterns'] = list(emerging)
            
            # Find disappearing patterns (in first but not last)
            disappearing = first_gen_patterns - last_gen_patterns
            results['disappearance_patterns'] = list(disappearing)
            
            # Find persistent patterns (in both first and last)
            persistent = first_gen_patterns.intersection(last_gen_patterns)
            results['persistent_patterns'] = list(persistent)
        
        logger.info(f"Analyzed pattern evolution across {results['generations']} generations")
        return results
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert analyzer to dictionary representation.
        
        Returns:
            Dictionary containing all analyzer attributes
        """
        return {
            'patterns': {pid: pattern.to_dict() for pid, pattern in self.patterns.items()},
            'pattern_counter': dict(self.pattern_counter)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResponsePatternAnalyzer':
        """
        Create analyzer from dictionary representation.
        
        Args:
            data: Dictionary containing analyzer attributes
            
        Returns:
            New response pattern analyzer instance
        """
        analyzer = cls()
        
        # Restore patterns
        if 'patterns' in data:
            for pid, pattern_data in data['patterns'].items():
                analyzer.patterns[pid] = ResponsePattern.from_dict(pattern_data)
        
        # Restore pattern counter
        if 'pattern_counter' in data:
            analyzer.pattern_counter = Counter(data['pattern_counter'])
        
        return analyzer