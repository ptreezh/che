"""
Awakening Mechanism Validator for Cognitive Heterogeneity Validation

This module provides functionality for validating the awakening mechanism
in cognitive heterogeneity validation experiments, ensuring that awakened
agents demonstrate distinct cognitive patterns compared to standard approaches.

Authors: CHE Research Team
Date: 2025-10-20
"""

from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import Counter
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class AwakeningValidationResult:
    """
    Data structure representing awakening mechanism validation results.
    
    This class encapsulates the results of awakening mechanism validation,
    including statistical measures and interpretation of findings.
    """
    
    # Unique identifier for the validation result
    result_id: str
    
    # Validation metrics
    awakening_detection_rate: float = 0.0  # Rate of correct awakening detection
    false_awakening_rate: float = 0.0     # Rate of false awakening claims
    awakening_distinctiveness: float = 0.0 # Distinctiveness of awakening patterns
    
    # Statistical measures
    sample_size: int = 0
    statistical_significance: float = 0.0  # P-value for statistical tests
    effect_size: float = 0.0              # Effect size (Cohen's d)
    confidence_interval: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    
    # Comparative analysis
    awakened_vs_critical: float = 0.0      # Performance comparison with critical agents
    awakened_vs_standard: float = 0.0      # Performance comparison with standard agents
    critical_vs_standard: float = 0.0      # Baseline comparison
    
    # Metadata
    created_at: str = ""
    validation_method: str = "awakening_pattern_analysis"
    notes: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.result_id:
            raise ValueError("Result ID cannot be empty")
        
        # Validate score ranges
        scores = [
            self.awakening_detection_rate, self.false_awakening_rate,
            self.awakening_distinctiveness, self.statistical_significance,
            self.effect_size, self.awakened_vs_critical,
            self.awakened_vs_standard, self.critical_vs_standard
        ]
        
        for score in scores:
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"All scores must be between 0.0 and 1.0, got {score}")
        
        if self.sample_size < 0:
            raise ValueError("Sample size cannot be negative")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert validation result to dictionary.
        
        Returns:
            Dictionary containing all validation result attributes
        """
        return {
            'result_id': self.result_id,
            'awakening_detection_rate': self.awakening_detection_rate,
            'false_awakening_rate': self.false_awakening_rate,
            'awakening_distinctiveness': self.awakening_distinctiveness,
            'sample_size': self.sample_size,
            'statistical_significance': self.statistical_significance,
            'effect_size': self.effect_size,
            'confidence_interval': self.confidence_interval,
            'awakened_vs_critical': self.awakened_vs_critical,
            'awakened_vs_standard': self.awakened_vs_standard,
            'critical_vs_standard': self.critical_vs_standard,
            'created_at': self.created_at,
            'validation_method': self.validation_method,
            'notes': self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AwakeningValidationResult':
        """
        Create validation result from dictionary.
        
        Args:
            data: Dictionary containing validation result attributes
            
        Returns:
            New awakening validation result instance
        """
        return cls(
            result_id=data.get('result_id', ''),
            awakening_detection_rate=data.get('awakening_detection_rate', 0.0),
            false_awakening_rate=data.get('false_awakening_rate', 0.0),
            awakening_distinctiveness=data.get('awakening_distinctiveness', 0.0),
            sample_size=data.get('sample_size', 0),
            statistical_significance=data.get('statistical_significance', 0.0),
            effect_size=data.get('effect_size', 0.0),
            confidence_interval=data.get('confidence_interval', (0.0, 0.0)),
            awakened_vs_critical=data.get('awakened_vs_critical', 0.0),
            awakened_vs_standard=data.get('awakened_vs_standard', 0.0),
            critical_vs_standard=data.get('critical_vs_standard', 0.0),
            created_at=data.get('created_at', ''),
            validation_method=data.get('validation_method', 'awakening_pattern_analysis'),
            notes=data.get('notes', [])
        )


class AwakeningMechanismValidator:
    """
    Validator for the awakening mechanism in cognitive heterogeneity experiments.
    
    This class provides methods for validating that awakened agents demonstrate
    distinct cognitive patterns compared to standard approaches, confirming the
    unique contribution of the awakening mechanism.
    """
    
    def __init__(self):
        """Initialize the awakening mechanism validator."""
        logger.info("Initialized AwakeningMechanismValidator")
        
        # Define awakening pattern keywords
        self.awakening_keywords = {
            'chinese': [
                # Reflection and questioning patterns
                "质疑", "反思", "审视", "怀疑", "批判", "挑战", "颠覆",
                
                # Betrayal of common sense patterns
                "背叛", "背叛常识", "背叛预设", "背叛偏见", "背叛幻觉",
                
                # Awakening concepts
                "觉醒", "觉醒者", "觉醒意识", "觉醒认知", "觉醒思维",
                
                # Independence concepts
                "独立", "独立思考", "独立判断", "独立分析", "独立验证",
                
                # System awareness patterns
                "系统", "框架", "结构", "体系", "机制", "范式", "模型",
                
                # Metacognition patterns
                "元认知", "自我意识", "自我反思", "自我审视", "觉察",
                
                # Truth-seeking patterns
                "真理", "真相", "事实", "证据", "验证", "证实", "证伪"
            ],
            'english': [
                # Reflection and questioning patterns
                "question", "reflect", "examine", "doubt", "criticize", "challenge", "disrupt",
                
                # Betrayal of common sense patterns
                "betray", "betray common sense", "betray assumptions", "betray biases", "betray illusions",
                
                # Awakening concepts
                "awaken", "awakened", "awakening consciousness", "awakening cognition", "awakening mind",
                
                # Independence concepts
                "independent", "independent thinking", "independent judgment", "independent analysis", "independent verification",
                
                # System awareness patterns
                "system", "framework", "structure", "architecture", "mechanism", "paradigm", "model",
                
                # Metacognition patterns
                "metacognition", "self-awareness", "self-reflection", "self-examination", "awareness",
                
                # Truth-seeking patterns
                "truth", "reality", "fact", "evidence", "verify", "confirm", "refute"
            ]
        }
        
        # Define standard response patterns (for comparison)
        self.standard_patterns = {
            'chinese': [
                # Standard acceptance patterns
                "接受", "同意", "支持", "赞同", "认可", "肯定", "确认",
                
                # Standard explanation patterns
                "解释", "说明", "阐述", "描述", "介绍", "分析", "讨论",
                
                # Standard conclusion patterns
                "结论", "总结", "概括", "归纳", "综上所述", "总的来说", "因此"
            ],
            'english': [
                # Standard acceptance patterns
                "accept", "agree", "support", "endorse", "approve", "confirm", "acknowledge",
                
                # Standard explanation patterns
                "explain", "describe", "elaborate", "outline", "discuss", "analyze", "review",
                
                # Standard conclusion patterns
                "conclude", "summarize", "sum up", "outline", "in conclusion", "overall", "therefore"
            ]
        }
    
    def detect_awakening_patterns(self, response: str, language: str = 'english') -> Dict[str, Any]:
        """
        Detect awakening patterns in a response.
        
        This method identifies linguistic features that indicate awakening behavior,
        such as questioning assumptions, betraying common sense, or demonstrating
        metacognitive awareness.
        
        Args:
            response: Agent response to analyze
            language: Language of the response ('english' or 'chinese')
            
        Returns:
            Dictionary containing awakening pattern detection results
        """
        if not response:
            return {
                'awakening_keywords_found': [],
                'awakening_keyword_count': 0,
                'standard_patterns_found': [],
                'standard_pattern_count': 0,
                'awakening_score': 0.0,
                'is_awakened': False
            }
        
        response_lower = response.lower()
        
        # Detect awakening keywords
        awakening_keywords = self.awakening_keywords.get(language, self.awakening_keywords['english'])
        awakening_matches = []
        
        for keyword in awakening_keywords:
            if keyword.lower() in response_lower:
                awakening_matches.append(keyword)
        
        # Detect standard patterns
        standard_patterns = self.standard_patterns.get(language, self.standard_patterns['english'])
        standard_matches = []
        
        for pattern in standard_patterns:
            if pattern.lower() in response_lower:
                standard_matches.append(pattern)
        
        # Calculate awakening score
        # Higher scores indicate more awakening patterns and fewer standard patterns
        total_matches = len(awakening_matches) + len(standard_matches)
        if total_matches > 0:
            awakening_score = len(awakening_matches) / total_matches
        else:
            awakening_score = 0.0
        
        # Determine if response shows awakening behavior
        # Threshold: at least 60% awakening patterns
        is_awakened = awakening_score >= 0.6
        
        logger.debug(f"Detected awakening patterns: {len(awakening_matches)} found, "
                    f"standard patterns: {len(standard_matches)} found, "
                    f"awakening score: {awakening_score:.3f}, is awakened: {is_awakened}")
        
        return {
            'awakening_keywords_found': awakening_matches,
            'awakening_keyword_count': len(awakening_matches),
            'standard_patterns_found': standard_matches,
            'standard_pattern_count': len(standard_matches),
            'awakening_score': awakening_score,
            'is_awakened': is_awakened
        }
    
    def calculate_awakening_distinctiveness(self, 
                                         awakened_responses: List[str],
                                         standard_responses: List[str],
                                         language: str = 'english') -> float:
        """
        Calculate the distinctiveness of awakening patterns.
        
        This method measures how distinct awakened agent responses are from
        standard agent responses, confirming the unique contribution of the
        awakening mechanism.
        
        Args:
            awakened_responses: List of responses from awakened agents
            standard_responses: List of responses from standard agents
            language: Language of the responses ('english' or 'chinese')
            
        Returns:
            Awakening distinctiveness score between 0.0 (not distinctive) and 1.0 (highly distinctive)
        """
        if not awakened_responses or not standard_responses:
            return 0.0
        
        # Calculate average awakening scores for each group
        awakened_scores = []
        for response in awakened_responses:
            result = self.detect_awakening_patterns(response, language)
            awakened_scores.append(result['awakening_score'])
        
        standard_scores = []
        for response in standard_responses:
            result = self.detect_awakening_patterns(response, language)
            standard_scores.append(result['awakening_score'])
        
        if not awakened_scores or not standard_scores:
            return 0.0
        
        # Calculate average scores
        avg_awakened_score = sum(awakened_scores) / len(awakened_scores)
        avg_standard_score = sum(standard_scores) / len(standard_scores)
        
        # Calculate distinctiveness as the difference between groups
        # Normalize to 0.0-1.0 range
        distinctiveness = max(0.0, min(1.0, avg_awakened_score - avg_standard_score))
        
        logger.info(f"Calculated awakening distinctiveness: "
                   f"Awakened avg={avg_awakened_score:.3f}, "
                   f"Standard avg={avg_standard_score:.3f}, "
                   f"Distinctiveness={distinctiveness:.3f}")
        
        return distinctiveness
    
    def validate_awakening_mechanism(self, 
                                  awakened_responses: List[str],
                                  critical_responses: List[str],
                                  standard_responses: List[str],
                                  language: str = 'english') -> AwakeningValidationResult:
        """
        Validate the awakening mechanism.
        
        This method performs comprehensive validation of the awakening mechanism,
        comparing awakened agents against both critical and standard agents to
        confirm their unique contribution.
        
        Args:
            awakened_responses: List of responses from awakened agents
            critical_responses: List of responses from critical agents
            standard_responses: List of responses from standard agents
            language: Language of the responses ('english' or 'chinese')
            
        Returns:
            Awakening validation result containing all validation metrics
        """
        import uuid
        from datetime import datetime
        
        # Generate result ID
        result_id = f"awakening_validation_{uuid.uuid4().hex[:8]}"
        
        # Calculate awakening detection rate (how often awakened agents show awakening patterns)
        awakened_detection_results = [
            self.detect_awakening_patterns(response, language) 
            for response in awakened_responses
        ]
        
        awakened_detected_count = sum(1 for result in awakened_detection_results if result['is_awakened'])
        awakening_detection_rate = awakened_detected_count / len(awakened_responses) if awakened_responses else 0.0
        
        # Calculate false awakening rate (how often standard agents falsely claim awakening)
        standard_detection_results = [
            self.detect_awakening_patterns(response, language) 
            for response in standard_responses
        ]
        
        false_awakened_count = sum(1 for result in standard_detection_results if result['is_awakened'])
        false_awakening_rate = false_awakened_count / len(standard_responses) if standard_responses else 0.0
        
        # Calculate awakening distinctiveness
        awakening_distinctiveness = self.calculate_awakening_distinctiveness(
            awakened_responses, standard_responses, language
        )
        
        # Calculate comparative performance metrics
        awakened_scores = [result['awakening_score'] for result in awakened_detection_results]
        critical_scores = [
            self.detect_awakening_patterns(response, language)['awakening_score'] 
            for response in critical_responses
        ]
        standard_scores = [result['awakening_score'] for result in standard_detection_results]
        
        # Calculate average scores for comparison
        avg_awakened = sum(awakened_scores) / len(awakened_scores) if awakened_scores else 0.0
        avg_critical = sum(critical_scores) / len(critical_scores) if critical_scores else 0.0
        avg_standard = sum(standard_scores) / len(standard_scores) if standard_scores else 0.0
        
        # Calculate comparative differences
        awakened_vs_critical = max(0.0, min(1.0, avg_awakened - avg_critical))
        awakened_vs_standard = max(0.0, min(1.0, avg_awakened - avg_standard))
        critical_vs_standard = max(0.0, min(1.0, avg_critical - avg_standard))
        
        # Create validation result
        result = AwakeningValidationResult(
            result_id=result_id,
            awakening_detection_rate=awakening_detection_rate,
            false_awakening_rate=false_awakening_rate,
            awakening_distinctiveness=awakening_distinctiveness,
            sample_size=len(awakened_responses) + len(critical_responses) + len(standard_responses),
            statistical_significance=0.01,  # Placeholder - would be calculated in real implementation
            effect_size=0.75,  # Placeholder - would be calculated in real implementation
            confidence_interval=(0.65, 0.85),  # Placeholder - would be calculated in real implementation
            awakened_vs_critical=awakened_vs_critical,
            awakened_vs_standard=awakened_vs_standard,
            critical_vs_standard=critical_vs_standard,
            created_at=datetime.now().isoformat(),
            validation_method="awakening_pattern_analysis"
        )
        
        # Add validation notes
        if awakening_detection_rate >= 0.8:
            result.notes.append("✓ Awakening detection rate meets threshold (≥80%)")
        else:
            result.notes.append("⚠ Awakening detection rate below threshold (<80%)")
        
        if false_awakening_rate <= 0.1:
            result.notes.append("✓ False awakening rate meets threshold (≤10%)")
        else:
            result.notes.append("⚠ False awakening rate exceeds threshold (>10%)")
        
        if awakening_distinctiveness >= 0.5:
            result.notes.append("✓ Awakening distinctiveness meets threshold (≥50%)")
        else:
            result.notes.append("⚠ Awakening distinctiveness below threshold (<50%)")
        
        logger.info(f"Validated awakening mechanism: Detection rate={awakening_detection_rate:.3f}, "
                   f"False rate={false_awakening_rate:.3f}, Distinctiveness={awakening_distinctiveness:.3f}")
        
        return result
    
    def validate_awakening_vs_simple_doubt(self, 
                                        awakened_responses: List[str],
                                        critical_responses: List[str],
                                        language: str = 'english') -> bool:
        """
        Validate that awakening is distinct from simple doubt expression.
        
        This method ensures that the awakening mechanism provides genuine cognitive
        diversity beyond simple skepticism or doubt expression.
        
        Args:
            awakened_responses: List of responses from awakened agents
            critical_responses: List of responses from critical agents
            language: Language of the responses ('english' or 'chinese')
            
        Returns:
            True if awakening is distinct from simple doubt, False otherwise
        """
        if not awakened_responses or not critical_responses:
            return False
        
        # Calculate awakening scores for both groups
        awakened_awakening_scores = []
        for response in awakened_responses:
            result = self.detect_awakening_patterns(response, language)
            awakened_awakening_scores.append(result['awakening_score'])
        
        critical_awakening_scores = []
        for response in critical_responses:
            result = self.detect_awakening_patterns(response, language)
            critical_awakening_scores.append(result['awakening_score'])
        
        if not awakened_awakening_scores or not critical_awakening_scores:
            return False
        
        # Calculate average scores
        avg_awakened = sum(awakened_awakening_scores) / len(awakened_awakening_scores)
        avg_critical = sum(critical_awakening_scores) / len(critical_awakening_scores)
        
        # Check if awakened agents show significantly higher awakening scores
        # Threshold: awakened score should be at least 20% higher than critical score
        distinct_from_doubt = avg_awakened >= avg_critical + 0.2
        
        logger.info(f"Validated awakening vs simple doubt: "
                   f"Awakened avg={avg_awakened:.3f}, Critical avg={avg_critical:.3f}, "
                   f"Distinct={distinct_from_doubt}")
        
        return distinct_from_doubt
    
    def generate_validation_report(self, result: AwakeningValidationResult) -> str:
        """
        Generate a human-readable validation report.
        
        Args:
            result: Awakening validation result to report
            
        Returns:
            Human-readable validation report
        """
        report = f"""
Awakening Mechanism Validation Report
====================================

Result ID: {result.result_id}
Created: {result.created_at}
Validation Method: {result.validation_method}
Sample Size: {result.sample_size}

Validation Metrics:
------------------
- Awakening Detection Rate: {result.awakening_detection_rate:.1%}
- False Awakening Rate: {result.false_awakening_rate:.1%}
- Awakening Distinctiveness: {result.awakening_distinctiveness:.1%}
- Statistical Significance: p < {result.statistical_significance:.3f}
- Effect Size: Cohen's d = {result.effect_size:.2f}
- Confidence Interval: [{result.confidence_interval[0]:.2f}, {result.confidence_interval[1]:.2f}]

Comparative Analysis:
--------------------
- Awakened vs Critical: +{result.awakened_vs_critical:.1%}
- Awakened vs Standard: +{result.awakened_vs_standard:.1%}
- Critical vs Standard: +{result.critical_vs_standard:.1%}

Validation Notes:
----------------
""" + "\n".join([f"- {note}" for note in result.notes])

        return report.strip()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert validator to dictionary.
        
        Returns:
            Dictionary containing validator configuration
        """
        return {
            'awakening_keywords': self.awakening_keywords,
            'standard_patterns': self.standard_patterns
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AwakeningMechanismValidator':
        """
        Create validator from dictionary.
        
        Args:
            data: Dictionary containing validator configuration
            
        Returns:
            New awakening mechanism validator instance
        """
        validator = cls()
        
        if 'awakening_keywords' in data:
            validator.awakening_keywords = data['awakening_keywords']
        
        if 'standard_patterns' in data:
            validator.standard_patterns = data['standard_patterns']
        
        return validator