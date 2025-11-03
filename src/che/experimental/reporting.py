"""
Experiment Reporting for Cognitive Heterogeneity Validation

This module provides functionality for generating comprehensive reports
on cognitive heterogeneity validation experiments, including statistical
analysis and visualization support.

Authors: CHE Research Team
Date: 2025-10-20
"""

import json
import csv
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentReport:
    """
    Data structure representing a comprehensive experiment report.
    
    This class encapsulates all information needed for a complete
    cognitive heterogeneity validation experiment report.
    """
    
    # Report metadata
    report_id: str
    experiment_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Experiment configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Experiment results
    results: Dict[str, Any] = field(default_factory=dict)
    
    # Statistical analysis
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Validation metrics
    validation_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert report to dictionary representation.
        
        Returns:
            Dictionary containing all report attributes
        """
        return {
            'report_id': self.report_id,
            'experiment_id': self.experiment_id,
            'timestamp': self.timestamp,
            'config': self.config,
            'results': self.results,
            'statistics': self.statistics,
            'validation_metrics': self.validation_metrics,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentReport':
        """
        Create report from dictionary representation.
        
        Args:
            data: Dictionary containing report attributes
            
        Returns:
            New experiment report instance
        """
        return cls(
            report_id=data.get('report_id', ''),
            experiment_id=data.get('experiment_id', ''),
            timestamp=data.get('timestamp', datetime.now().isoformat()),
            config=data.get('config', {}),
            results=data.get('results', {}),
            statistics=data.get('statistics', {}),
            validation_metrics=data.get('validation_metrics', {}),
            metadata=data.get('metadata', {})
        )


class ExperimentReporter:
    """
    Reporter for generating comprehensive experiment reports.
    
    This class provides methods for creating detailed reports on cognitive
    heterogeneity validation experiments, including results, statistics,
    and validation metrics.
    """
    
    def __init__(self):
        """Initialize the experiment reporter."""
        logger.info("Initialized ExperimentReporter")
    
    def generate_experiment_report(self, 
                               experiment_id: str,
                               config: Dict[str, Any],
                               results: Dict[str, Any],
                               statistics: Dict[str, Any],
                               validation_metrics: Dict[str, Any],
                               metadata: Optional[Dict[str, Any]] = None) -> ExperimentReport:
        """
        Generate a comprehensive experiment report.
        
        Args:
            experiment_id: Unique identifier for the experiment
            config: Experiment configuration
            results: Experiment results
            statistics: Statistical analysis results
            validation_metrics: Validation metrics
            metadata: Optional additional metadata
            
        Returns:
            Complete experiment report
        """
        import uuid
        
        report = ExperimentReport(
            report_id=f"report_{uuid.uuid4().hex[:8]}",
            experiment_id=experiment_id,
            config=config,
            results=results,
            statistics=statistics,
            validation_metrics=validation_metrics,
            metadata=metadata or {}
        )
        
        logger.info(f"Generated experiment report {report.report_id} for experiment {experiment_id}")
        return report
    
    def export_report_to_json(self, report: ExperimentReport, filepath: str) -> None:
        """
        Export report to JSON file.
        
        Args:
            report: Experiment report to export
            filepath: Path to export file
            
        Raises:
            IOError: If report cannot be exported
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported report to JSON: {filepath}")
        except Exception as e:
            logger.error(f"Failed to export report to JSON {filepath}: {e}")
            raise IOError(f"Failed to export report: {e}") from e
    
    def export_report_to_csv(self, report: ExperimentReport, filepath: str) -> None:
        """
        Export report data to CSV file.
        
        Args:
            report: Experiment report to export
            filepath: Path to export file
            
        Raises:
            IOError: If report cannot be exported
        """
        try:
            # Flatten report data for CSV export
            flattened_data = self._flatten_report_data(report)
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header
                if flattened_data:
                    headers = list(flattened_data[0].keys())
                    writer.writerow(headers)
                    
                    # Write data rows
                    for row in flattened_data:
                        writer.writerow([row.get(header, '') for header in headers])
            
            logger.info(f"Exported report to CSV: {filepath}")
        except Exception as e:
            logger.error(f"Failed to export report to CSV {filepath}: {e}")
            raise IOError(f"Failed to export report: {e}") from e
    
    def _flatten_report_data(self, report: ExperimentReport) -> List[Dict[str, Any]]:
        """
        Flatten report data for CSV export.
        
        Args:
            report: Experiment report to flatten
            
        Returns:
            List of flattened data dictionaries
        """
        # This is a simplified implementation
        # In a real system, this would flatten nested data structures
        flattened = []
        
        # Create a flat representation
        flat_record = {
            'report_id': report.report_id,
            'experiment_id': report.experiment_id,
            'timestamp': report.timestamp,
            'population_size': report.config.get('population_size', 0),
            'generations': report.config.get('generations', 0),
            'final_avg_score': report.results.get('final_avg_score', 0.0),
            'statistical_significance': report.statistics.get('statistical_significance', 0.0),
            'effect_size': report.statistics.get('effect_size', 0.0),
            'cognitive_independence_correlation': report.validation_metrics.get('cognitive_independence_correlation', 0.0)
        }
        
        flattened.append(flat_record)
        return flattened
    
    def generate_markdown_report(self, report: ExperimentReport) -> str:
        """
        Generate a markdown report.
        
        Args:
            report: Experiment report to convert to markdown
            
        Returns:
            Markdown-formatted report
        """
        md_report = f"""# Cognitive Heterogeneity Validation Experiment Report

## Metadata
- **Report ID**: {report.report_id}
- **Experiment ID**: {report.experiment_id}
- **Timestamp**: {report.timestamp}

## Configuration
```json
{json.dumps(report.config, indent=2, ensure_ascii=False)}
```

## Results
```json
{json.dumps(report.results, indent=2, ensure_ascii=False)}
```

## Statistics
```json
{json.dumps(report.statistics, indent=2, ensure_ascii=False)}
```

## Validation Metrics
```json
{json.dumps(report.validation_metrics, indent=2, ensure_ascii=False)}
```

## Metadata
```json
{json.dumps(report.metadata, indent=2, ensure_ascii=False)}
```
"""
        
        logger.info(f"Generated markdown report for {report.report_id}")
        return md_report
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert reporter to dictionary representation.
        
        Returns:
            Dictionary containing reporter configuration
        """
        return {}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentReporter':
        """
        Create reporter from dictionary representation.
        
        Args:
            data: Dictionary containing reporter configuration
            
        Returns:
            New experiment reporter instance
        """
        return cls()