"""
Checkpoint and Persistence Utilities for Cognitive Heterogeneity Validation

This module provides utilities for saving and restoring experiment state,
enabling resumable experiments and persistent storage of results.

Authors: CHE Research Team
Date: 2025-10-19
"""

import os
import json
import pickle
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import uuid


logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """
    Checkpoint data structure for saving and restoring experiment state.
    
    This class represents a snapshot of experiment state that can be saved
    and later restored to resume an experiment.
    """
    
    # Unique identifier for the checkpoint
    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Timestamp when checkpoint was created
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Experiment generation number
    generation: int = 0
    
    # Experiment state data
    state_data: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata about the checkpoint
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert checkpoint to dictionary.
        
        Returns:
            Dictionary representation of checkpoint
        """
        return {
            'checkpoint_id': self.checkpoint_id,
            'timestamp': self.timestamp,
            'generation': self.generation,
            'state_data': self.state_data,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Checkpoint':
        """
        Create checkpoint from dictionary.
        
        Args:
            data: Dictionary representation of checkpoint
            
        Returns:
            New checkpoint instance
        """
        return cls(
            checkpoint_id=data.get('checkpoint_id', str(uuid.uuid4())),
            timestamp=data.get('timestamp', datetime.now().isoformat()),
            generation=data.get('generation', 0),
            state_data=data.get('state_data', {}),
            metadata=data.get('metadata', {})
        )


class CheckpointManager:
    """
    Checkpoint manager for saving and restoring experiment state.
    
    This class provides methods for creating checkpoints, saving them to disk,
    and restoring experiments from saved checkpoints.
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize the checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = checkpoint_dir
        self._ensure_checkpoint_dir()
    
    def _ensure_checkpoint_dir(self) -> None:
        """Ensure checkpoint directory exists."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def create_checkpoint(self, generation: int, state_data: Dict[str, Any], 
                         metadata: Optional[Dict[str, Any]] = None) -> Checkpoint:
        """
        Create a checkpoint from experiment state.
        
        Args:
            generation: Current generation number
            state_data: Experiment state data to save
            metadata: Optional metadata about the checkpoint
            
        Returns:
            New checkpoint instance
        """
        checkpoint = Checkpoint(
            generation=generation,
            state_data=state_data,
            metadata=metadata or {}
        )
        
        logger.info(f"Created checkpoint {checkpoint.checkpoint_id} for generation {generation}")
        return checkpoint
    
    def save_checkpoint(self, checkpoint: Checkpoint, filename: Optional[str] = None) -> str:
        """
        Save a checkpoint to disk.
        
        Args:
            checkpoint: Checkpoint to save
            filename: Optional filename for the checkpoint.
                      If not provided, generates a filename based on checkpoint ID.
                      
        Returns:
            Path to the saved checkpoint file
            
        Raises:
            IOError: If checkpoint cannot be saved to disk
        """
        if filename is None:
            filename = f"checkpoint_{checkpoint.checkpoint_id}.json"
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(checkpoint.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved checkpoint to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save checkpoint to {filepath}: {e}")
            raise IOError(f"Failed to save checkpoint: {e}") from e
    
    def save_checkpoint_pickle(self, checkpoint: Checkpoint, filename: Optional[str] = None) -> str:
        """
        Save a checkpoint to disk using pickle serialization.
        
        This method is useful for saving complex Python objects that may not
        be easily serializable to JSON.
        
        Args:
            checkpoint: Checkpoint to save
            filename: Optional filename for the checkpoint.
                      If not provided, generates a filename based on checkpoint ID.
                      
        Returns:
            Path to the saved checkpoint file
            
        Raises:
            IOError: If checkpoint cannot be saved to disk
        """
        if filename is None:
            filename = f"checkpoint_{checkpoint.checkpoint_id}.pkl"
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            logger.info(f"Saved checkpoint (pickle) to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save checkpoint to {filepath}: {e}")
            raise IOError(f"Failed to save checkpoint: {e}") from e
    
    def load_checkpoint(self, filename: str) -> Checkpoint:
        """
        Load a checkpoint from disk.
        
        Args:
            filename: Name of the checkpoint file to load
            
        Returns:
            Loaded checkpoint instance
            
        Raises:
            FileNotFoundError: If checkpoint file does not exist
            IOError: If checkpoint cannot be loaded from disk
        """
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            checkpoint = Checkpoint.from_dict(data)
            logger.info(f"Loaded checkpoint {checkpoint.checkpoint_id} from {filepath}")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {filepath}: {e}")
            raise IOError(f"Failed to load checkpoint: {e}") from e
    
    def load_checkpoint_pickle(self, filename: str) -> Checkpoint:
        """
        Load a checkpoint from disk using pickle deserialization.
        
        Args:
            filename: Name of the checkpoint file to load
            
        Returns:
            Loaded checkpoint instance
            
        Raises:
            FileNotFoundError: If checkpoint file does not exist
            IOError: If checkpoint cannot be loaded from disk
        """
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                checkpoint = pickle.load(f)
            
            logger.info(f"Loaded checkpoint (pickle) from {filepath}")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {filepath}: {e}")
            raise IOError(f"Failed to load checkpoint: {e}") from e
    
    def list_checkpoints(self) -> list:
        """
        List all available checkpoints in the checkpoint directory.
        
        Returns:
            List of checkpoint filenames
        """
        try:
            files = os.listdir(self.checkpoint_dir)
            checkpoint_files = [f for f in files if f.startswith("checkpoint_") and f.endswith((".json", ".pkl"))]
            return sorted(checkpoint_files)
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get the latest checkpoint file.
        
        Returns:
            Name of the latest checkpoint file, or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        
        # Sort by modification time to get the latest
        checkpoint_paths = [os.path.join(self.checkpoint_dir, f) for f in checkpoints]
        latest_checkpoint = max(checkpoint_paths, key=os.path.getmtime)
        return os.path.basename(latest_checkpoint)
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> int:
        """
        Clean up old checkpoints, keeping only the most recent N.
        
        Args:
            keep_last_n: Number of most recent checkpoints to keep
            
        Returns:
            Number of checkpoints deleted
        """
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= keep_last_n:
            return 0
        
        # Sort by modification time
        checkpoint_paths = [os.path.join(self.checkpoint_dir, f) for f in checkpoints]
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
        
        return deleted_count


# Convenience functions for common checkpoint operations


def save_experiment_state(generation: int, ecosystem_state: Dict[str, Any], 
                          experiment_config: Dict[str, Any], 
                          checkpoint_dir: str = "checkpoints") -> str:
    """
    Save the current experiment state as a checkpoint.
    
    Args:
        generation: Current generation number
        ecosystem_state: Current ecosystem state
        experiment_config: Experiment configuration
        checkpoint_dir: Directory to save checkpoint
        
    Returns:
        Path to the saved checkpoint file
    """
    manager = CheckpointManager(checkpoint_dir)
    
    state_data = {
        'ecosystem_state': ecosystem_state,
        'experiment_config': experiment_config
    }
    
    metadata = {
        'created_by': 'save_experiment_state',
        'python_version': '.'.join(map(str, __import__('sys').version_info[:3]))
    }
    
    checkpoint = manager.create_checkpoint(
        generation=generation,
        state_data=state_data,
        metadata=metadata
    )
    
    filename = f"experiment_checkpoint_gen_{generation}.json"
    return manager.save_checkpoint(checkpoint, filename)


def load_experiment_state(checkpoint_file: str, 
                          checkpoint_dir: str = "checkpoints") -> Checkpoint:
    """
    Load experiment state from a checkpoint file.
    
    Args:
        checkpoint_file: Name of the checkpoint file to load
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        Loaded checkpoint with experiment state
    """
    manager = CheckpointManager(checkpoint_dir)
    return manager.load_checkpoint(checkpoint_file)


def find_resume_point(checkpoint_dir: str = "checkpoints") -> Optional[str]:
    """
    Find the latest checkpoint to resume an experiment.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        Name of the latest checkpoint file, or None if no checkpoints exist
    """
    manager = CheckpointManager(checkpoint_dir)
    return manager.get_latest_checkpoint()