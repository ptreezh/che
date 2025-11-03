"""
Resumable Experiment Functionality for Cognitive Heterogeneity Validation

This module provides functionality for resuming interrupted experiments,
enabling long-running cognitive heterogeneity validation experiments to
continue from their last saved state.

Authors: CHE Research Team
Date: 2025-10-20
"""

import os
import json
import pickle
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import uuid

# Import ecosystem from core instead of experimental
from ..core.ecosystem import Ecosystem
from ..core.task import Task

logger = logging.getLogger(__name__)


@dataclass
class ExperimentState:
    """
    Data structure representing experiment state for resumable experiments.
    
    This class encapsulates all information needed to save and restore
    the state of a cognitive heterogeneity validation experiment.
    """
    
    # Unique identifier for the experiment state
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Timestamp when state was saved
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Experiment configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Current generation number
    current_generation: int = 0
    
    # Ecosystem state
    ecosystem_state: Dict[str, Any] = field(default_factory=dict)
    
    # Task state
    task_state: Dict[str, Any] = field(default_factory=dict)
    
    # Results history
    results_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert experiment state to dictionary representation.
        
        Returns:
            Dictionary containing all experiment state attributes
        """
        return {
            'state_id': self.state_id,
            'timestamp': self.timestamp,
            'config': self.config,
            'current_generation': self.current_generation,
            'ecosystem_state': self.ecosystem_state,
            'task_state': self.task_state,
            'results_history': self.results_history,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentState':
        """
        Create experiment state from dictionary representation.
        
        Args:
            data: Dictionary containing experiment state attributes
            
        Returns:
            New experiment state instance
        """
        return cls(
            state_id=data.get('state_id', str(uuid.uuid4())),
            timestamp=data.get('timestamp', datetime.now().isoformat()),
            config=data.get('config', {}),
            current_generation=data.get('current_generation', 0),
            ecosystem_state=data.get('ecosystem_state', {}),
            task_state=data.get('task_state', {}),
            results_history=data.get('results_history', []),
            metadata=data.get('metadata', {})
        )


class ResumableExperiment:
    """
    Resumable experiment for cognitive heterogeneity validation.
    
    This class provides functionality for saving and restoring experiment state,
    enabling long-running cognitive heterogeneity validation experiments to
    continue from their last saved state.
    """
    
    def __init__(self, experiment_dir: str = "experiments"):
        """
        Initialize the resumable experiment.
        
        Args:
            experiment_dir: Directory to store experiment files
        """
        self.experiment_dir = experiment_dir
        self._ensure_experiment_dir()
        logger.info(f"Initialized ResumableExperiment in {experiment_dir}")
    
    def _ensure_experiment_dir(self) -> None:
        """Ensure experiment directory exists."""
        os.makedirs(self.experiment_dir, exist_ok=True)
    
    def save_experiment_state(self, 
                          experiment_id: str,
                          config: Dict[str, Any],
                          current_generation: int,
                          ecosystem: Ecosystem,
                          task: Task,
                          results_history: List[Dict[str, Any]],
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save experiment state to enable resumption.
        
        Args:
            experiment_id: Unique identifier for the experiment
            config: Experiment configuration
            current_generation: Current generation number
            ecosystem: Current ecosystem state
            task: Current task state
            results_history: History of experiment results
            metadata: Optional additional metadata
            
        Returns:
            Path to the saved state file
            
        Raises:
            IOError: If state cannot be saved to disk
        """
        # Create experiment state
        state = ExperimentState(
            state_id=f"state_{uuid.uuid4().hex[:8]}",
            config=config,
            current_generation=current_generation,
            ecosystem_state=ecosystem.to_dict(),
            task_state=task.to_dict(),
            results_history=results_history,
            metadata=metadata or {}
        )
        
        # Generate state file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state_filename = f"{experiment_id}_gen_{current_generation}_{timestamp}.json"
        state_filepath = os.path.join(self.experiment_dir, state_filename)
        
        try:
            # Save state to JSON file
            with open(state_filepath, 'w', encoding='utf-8') as f:
                json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved experiment state to {state_filepath}")
            return state_filepath
        except Exception as e:
            logger.error(f"Failed to save experiment state to {state_filepath}: {e}")
            raise IOError(f"Failed to save experiment state: {e}") from e
    
    def save_experiment_state_pickle(self, 
                                 experiment_id: str,
                                 config: Dict[str, Any],
                                 current_generation: int,
                                 ecosystem: Ecosystem,
                                 task: Task,
                                 results_history: List[Dict[str, Any]],
                                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save experiment state using pickle serialization.
        
        This method is useful for saving complex Python objects that may not
        be easily serializable to JSON.
        
        Args:
            experiment_id: Unique identifier for the experiment
            config: Experiment configuration
            current_generation: Current generation number
            ecosystem: Current ecosystem state
            task: Current task state
            results_history: History of experiment results
            metadata: Optional additional metadata
            
        Returns:
            Path to the saved state file
            
        Raises:
            IOError: If state cannot be saved to disk
        """
        # Create experiment state
        state = ExperimentState(
            state_id=f"state_{uuid.uuid4().hex[:8]}",
            config=config,
            current_generation=current_generation,
            ecosystem_state=ecosystem.to_dict(),
            task_state=task.to_dict(),
            results_history=results_history,
            metadata=metadata or {}
        )
        
        # Generate state file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state_filename = f"{experiment_id}_gen_{current_generation}_{timestamp}.pkl"
        state_filepath = os.path.join(self.experiment_dir, state_filename)
        
        try:
            # Save state to pickle file
            with open(state_filepath, 'wb') as f:
                pickle.dump(state, f)
            
            logger.info(f"Saved experiment state (pickle) to {state_filepath}")
            return state_filepath
        except Exception as e:
            logger.error(f"Failed to save experiment state to {state_filepath}: {e}")
            raise IOError(f"Failed to save experiment state: {e}") from e
    
    def load_experiment_state(self, state_filepath: str) -> ExperimentState:
        """
        Load experiment state from file.
        
        Args:
            state_filepath: Path to the state file
            
        Returns:
            Loaded experiment state
            
        Raises:
            FileNotFoundError: If state file does not exist
            IOError: If state cannot be loaded from disk
        """
        if not os.path.exists(state_filepath):
            raise FileNotFoundError(f"Experiment state file not found: {state_filepath}")
        
        try:
            # Determine file type from extension
            if state_filepath.endswith('.pkl'):
                # Load from pickle file
                with open(state_filepath, 'rb') as f:
                    state = pickle.load(f)
            else:
                # Load from JSON file
                with open(state_filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                state = ExperimentState.from_dict(data)
            
            logger.info(f"Loaded experiment state from {state_filepath}")
            return state
        except Exception as e:
            logger.error(f"Failed to load experiment state from {state_filepath}: {e}")
            raise IOError(f"Failed to load experiment state: {e}") from e
    
    def find_latest_checkpoint(self, experiment_id: str) -> Optional[str]:
        """
        Find the latest checkpoint for an experiment.
        
        Args:
            experiment_id: Unique identifier for the experiment
            
        Returns:
            Path to the latest checkpoint file, or None if none found
        """
        try:
            # List all files in experiment directory
            files = os.listdir(self.experiment_dir)
            
            # Filter checkpoint files for this experiment
            checkpoint_files = [
                f for f in files 
                if f.startswith(experiment_id) and (f.endswith('.json') or f.endswith('.pkl'))
            ]
            
            if not checkpoint_files:
                logger.info(f"No checkpoints found for experiment {experiment_id}")
                return None
            
            # Sort by modification time to get latest
            checkpoint_paths = [os.path.join(self.experiment_dir, f) for f in checkpoint_files]
            latest_checkpoint = max(checkpoint_paths, key=os.path.getmtime)
            
            logger.info(f"Found latest checkpoint for {experiment_id}: {latest_checkpoint}")
            return latest_checkpoint
        except Exception as e:
            logger.error(f"Failed to find latest checkpoint for {experiment_id}: {e}")
            return None
    
    def resume_experiment(self, experiment_id: str) -> Optional[ExperimentState]:
        """
        Resume an experiment from its latest checkpoint.
        
        Args:
            experiment_id: Unique identifier for the experiment
            
        Returns:
            Loaded experiment state, or None if no checkpoint found
        """
        latest_checkpoint = self.find_latest_checkpoint(experiment_id)
        
        if latest_checkpoint is None:
            logger.info(f"No checkpoint found to resume experiment {experiment_id}")
            return None
        
        try:
            state = self.load_experiment_state(latest_checkpoint)
            logger.info(f"Resumed experiment {experiment_id} from generation {state.current_generation}")
            return state
        except Exception as e:
            logger.error(f"Failed to resume experiment {experiment_id} from {latest_checkpoint}: {e}")
            return None
    
    def cleanup_old_checkpoints(self, experiment_id: str, keep_last_n: int = 5) -> int:
        """
        Clean up old checkpoints for an experiment, keeping only the most recent N.
        
        Args:
            experiment_id: Unique identifier for the experiment
            keep_last_n: Number of most recent checkpoints to keep
            
        Returns:
            Number of checkpoints deleted
        """
        try:
            # List all files in experiment directory
            files = os.listdir(self.experiment_dir)
            
            # Filter checkpoint files for this experiment
            checkpoint_files = [
                f for f in files 
                if f.startswith(experiment_id) and (f.endswith('.json') or f.endswith('.pkl'))
            ]
            
            if len(checkpoint_files) <= keep_last_n:
                logger.info(f"Not enough checkpoints to clean up for {experiment_id}")
                return 0
            
            # Sort by modification time
            checkpoint_paths = [os.path.join(self.experiment_dir, f) for f in checkpoint_files]
            checkpoint_paths.sort(key=os.path.getmtime)
            
            # Delete oldest checkpoints
            to_delete = checkpoint_paths[:-keep_last_n]
            deleted_count = 0
            
            for filepath in to_delete:
                try:
                    os.remove(filepath)
                    logger.info(f"Deleted old checkpoint: {filepath}")
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete checkpoint {filepath}: {e}")
            
            logger.info(f"Cleaned up {deleted_count} old checkpoints for {experiment_id}")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to clean up checkpoints for {experiment_id}: {e}")
            return 0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert resumable experiment to dictionary representation.
        
        Returns:
            Dictionary containing resumable experiment configuration
        """
        return {
            'experiment_dir': self.experiment_dir
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResumableExperiment':
        """
        Create resumable experiment from dictionary representation.
        
        Args:
            data: Dictionary containing resumable experiment configuration
            
        Returns:
            New resumable experiment instance
        """
        return cls(
            experiment_dir=data.get('experiment_dir', 'experiments')
        )