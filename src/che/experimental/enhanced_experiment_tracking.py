"""
Enhanced Experiment Tracking System for Cognitive Heterogeneity Validation

This module provides enhanced tracking capabilities for the cognitive heterogeneity experiments,
addressing concerns about data quality, timing verification, and detailed performance monitoring.

Authors: CHE Research Team
Date: 2025-10-27
"""

import json
import csv
import time
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelCallRecord:
    """Records detailed information about each model API call."""
    call_id: str
    agent_id: str
    task_id: str
    model_name: str
    start_time: float
    end_time: float
    duration_seconds: float
    success: bool
    error_message: Optional[str]
    input_prompt: str
    raw_response: str
    response_length: int
    tokens_used: Optional[int] = None

@dataclass
class AgentPerformanceRecord:
    """Records detailed performance metrics for each agent on each task."""
    record_id: str
    agent_id: str
    task_id: str
    generation: int
    score: float
    evaluation_timestamp: float
    evaluator_id: str
    evaluation_notes: Optional[str]
    response_quality: str  # "ACCEPTED", "DOUBTFUL", "REJECTED"

@dataclass
class GenerationCheckpoint:
    """Records comprehensive information about each generation."""
    generation_id: int
    start_time: str
    end_time: str
    duration_seconds: float
    population_size: int
    average_performance: float
    performance_std_dev: float
    diversity_index: float
    completed_tasks: int
    successful_model_calls: int
    failed_model_calls: int
    total_model_calls: int
    average_call_duration: float
    generation_hash: str  # For integrity verification

class EnhancedExperimentTracker:
    """
    Enhanced tracker for cognitive heterogeneity experiments with detailed monitoring
    and data quality verification.
    """
    
    def __init__(self, experiment_id: str, base_output_dir: str = "experiment_data"):
        """
        Initialize the enhanced experiment tracker.
        
        Args:
            experiment_id: Unique identifier for this experiment
            base_output_dir: Base directory for storing experiment data
        """
        self.experiment_id = experiment_id
        self.base_output_dir = Path(base_output_dir)
        self.experiment_dir = self.base_output_dir / experiment_id
        self.generation_dirs = {}
        self.agent_dirs = {}
        self.task_dirs = {}
        
        # Create directory structure
        self._setup_directories()
        
        # Initialize tracking collections
        self.model_calls: List[ModelCallRecord] = []
        self.agent_performances: List[AgentPerformanceRecord] = []
        self.generation_checkpoints: List[GenerationCheckpoint] = []
        
        # Log initialization
        logger.info(f"Enhanced Experiment Tracker initialized for experiment: {experiment_id}")
        logger.info(f"Data will be stored in: {self.experiment_dir}")
    
    def _setup_directories(self):
        """Setup the pyramid directory structure for data storage."""
        # Create main experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create top-level directories
        (self.experiment_dir / "generation_data").mkdir(exist_ok=True)
        (self.experiment_dir / "agent_data").mkdir(exist_ok=True)
        (self.experiment_dir / "task_data").mkdir(exist_ok=True)
        (self.experiment_dir / "logs").mkdir(exist_ok=True)
        
        # Create master experiment log
        self.master_log_path = self.experiment_dir / "master_experiment_log.json"
        self._initialize_master_log()
        
        logger.info("Directory structure setup completed")
    
    def _initialize_master_log(self):
        """Initialize the master experiment log file."""
        if not self.master_log_path.exists():
            master_log = {
                "experiment_id": self.experiment_id,
                "start_time": datetime.now().isoformat(),
                "status": "INITIALIZED",
                "generation_files": [],
                "agent_files": [],
                "task_files": [],
                "total_model_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0
            }
            self._write_json_file(self.master_log_path, master_log)
    
    def _update_master_log(self, updates: Dict[str, Any]):
        """Update the master experiment log with new information."""
        try:
            master_log = self._read_json_file(self.master_log_path)
            master_log.update(updates)
            self._write_json_file(self.master_log_path, master_log)
        except Exception as e:
            logger.error(f"Failed to update master log: {e}")
    
    def _create_generation_directory(self, generation: int) -> Path:
        """Create directory structure for a specific generation."""
        gen_dir = self.experiment_dir / "generation_data" / f"GEN_{generation:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        self.generation_dirs[generation] = gen_dir
        return gen_dir
    
    def _create_agent_directory(self, agent_id: str) -> Path:
        """Create directory structure for a specific agent."""
        agent_dir = self.experiment_dir / "agent_data" / agent_id
        agent_dir.mkdir(parents=True, exist_ok=True)
        self.agent_dirs[agent_id] = agent_dir
        return agent_dir
    
    def _create_task_directory(self, task_id: str) -> Path:
        """Create directory structure for a specific task."""
        task_dir = self.experiment_dir / "task_data" / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        self.task_dirs[task_id] = task_dir
        return task_dir
    
    def _write_json_file(self, filepath: Path, data: Dict[str, Any]):
        """Write data to a JSON file."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to write JSON file {filepath}: {e}")
    
    def _read_json_file(self, filepath: Path) -> Dict[str, Any]:
        """Read data from a JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read JSON file {filepath}: {e}")
            return {}
    
    def _write_csv_file(self, filepath: Path, data: List[Dict[str, Any]], headers: List[str]):
        """Write data to a CSV file."""
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(data)
        except Exception as e:
            logger.error(f"Failed to write CSV file {filepath}: {e}")
    
    def record_model_call(self, 
                          agent_id: str,
                          task_id: str,
                          model_name: str,
                          start_time: float,
                          end_time: float,
                          success: bool,
                          input_prompt: str,
                          raw_response: str,
                          error_message: Optional[str] = None,
                          tokens_used: Optional[int] = None) -> str:
        """
        Record a model API call with detailed timing and success/failure information.
        
        Args:
            agent_id: ID of the agent making the call
            task_id: ID of the task being processed
            model_name: Name of the model used
            start_time: Unix timestamp when call started
            end_time: Unix timestamp when call ended
            success: Whether the call was successful
            input_prompt: The prompt sent to the model
            raw_response: The raw response from the model
            error_message: Error message if call failed
            tokens_used: Number of tokens used (if available)
            
        Returns:
            Unique call ID for reference
        """
        call_id = f"{agent_id}_{task_id}_{int(start_time)}"
        duration = end_time - start_time
        
        # Create model call record
        record = ModelCallRecord(
            call_id=call_id,
            agent_id=agent_id,
            task_id=task_id,
            model_name=model_name,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            success=success,
            error_message=error_message,
            input_prompt=input_prompt,
            raw_response=raw_response,
            response_length=len(raw_response),
            tokens_used=tokens_used
        )
        
        # Add to tracking collection
        self.model_calls.append(record)
        
        # Log important information
        if not success:
            logger.warning(f"Model call failed for agent {agent_id}, task {task_id}: {error_message}")
        else:
            logger.debug(f"Model call completed for agent {agent_id}, task {task_id} in {duration:.2f}s")
        
        # Update master log statistics
        self._update_master_log({
            "total_model_calls": len(self.model_calls),
            "successful_calls": len([r for r in self.model_calls if r.success]),
            "failed_calls": len([r for r in self.model_calls if not r.success])
        })
        
        return call_id
    
    def record_agent_performance(self,
                                agent_id: str,
                                task_id: str,
                                generation: int,
                                score: float,
                                evaluator_id: str,
                                evaluation_notes: Optional[str] = None,
                                response_quality: str = "ACCEPTED") -> str:
        """
        Record detailed performance metrics for an agent on a specific task.
        
        Args:
            agent_id: ID of the agent being evaluated
            task_id: ID of the task evaluated
            generation: Generation number
            score: Performance score (0.0-2.0)
            evaluator_id: ID of the evaluator
            evaluation_notes: Additional notes about the evaluation
            response_quality: Quality classification of the response
            
        Returns:
            Unique record ID for reference
        """
        record_id = f"{agent_id}_{task_id}_{generation}_{int(time.time())}"
        
        # Validate score range
        if not 0.0 <= score <= 2.0:
            logger.warning(f"Score {score} is outside valid range [0.0, 2.0] for agent {agent_id}")
        
        # Create performance record
        record = AgentPerformanceRecord(
            record_id=record_id,
            agent_id=agent_id,
            task_id=task_id,
            generation=generation,
            score=score,
            evaluation_timestamp=time.time(),
            evaluator_id=evaluator_id,
            evaluation_notes=evaluation_notes,
            response_quality=response_quality
        )
        
        # Add to tracking collection
        self.agent_performances.append(record)
        
        logger.debug(f"Recorded performance for agent {agent_id} on task {task_id}: score={score}")
        
        return record_id
    
    def create_generation_checkpoint(self,
                                   generation: int,
                                   population_size: int,
                                   average_performance: float,
                                   performance_std_dev: float,
                                   diversity_index: float,
                                   completed_tasks: int,
                                   start_time: float,
                                   end_time: float) -> str:
        """
        Create a comprehensive checkpoint for a generation with all metrics.
        
        Args:
            generation: Generation number
            population_size: Number of agents in this generation
            average_performance: Average performance score
            performance_std_dev: Standard deviation of performance scores
            diversity_index: Cognitive diversity index
            completed_tasks: Number of tasks completed this generation
            start_time: Unix timestamp when generation started
            end_time: Unix timestamp when generation ended
            
        Returns:
            Path to the generation checkpoint file
        """
        # Create generation directory
        gen_dir = self._create_generation_directory(generation)
        
        # Filter records for this generation
        gen_model_calls = [r for r in self.model_calls if r.agent_id.split('_')[0] in ['critical', 'awakened', 'standard']]
        gen_agent_performances = [r for r in self.agent_performances if r.generation == generation]
        
        # Calculate metrics
        successful_calls = len([r for r in gen_model_calls if r.success])
        failed_calls = len([r for r in gen_model_calls if not r.success])
        total_calls = len(gen_model_calls)
        avg_call_duration = sum(r.duration_seconds for r in gen_model_calls) / total_calls if total_calls > 0 else 0
        
        # Create hash for integrity verification
        gen_data_for_hash = f"{generation}_{population_size}_{average_performance}_{diversity_index}_{completed_tasks}"
        generation_hash = hashlib.sha256(gen_data_for_hash.encode()).hexdigest()[:16]
        
        # Create generation checkpoint
        checkpoint = GenerationCheckpoint(
            generation_id=generation,
            start_time=datetime.fromtimestamp(start_time).isoformat(),
            end_time=datetime.fromtimestamp(end_time).isoformat(),
            duration_seconds=end_time - start_time,
            population_size=population_size,
            average_performance=average_performance,
            performance_std_dev=performance_std_dev,
            diversity_index=diversity_index,
            completed_tasks=completed_tasks,
            successful_model_calls=successful_calls,
            failed_model_calls=failed_calls,
            total_model_calls=total_calls,
            average_call_duration=avg_call_duration,
            generation_hash=generation_hash
        )
        
        # Add to tracking collection
        self.generation_checkpoints.append(checkpoint)
        
        # Save generation data to files
        gen_summary_path = gen_dir / f"generation_summary_{generation:03d}.json"
        gen_performance_path = gen_dir / f"generation_performance_{generation:03d}.csv"
        gen_timing_path = gen_dir / f"generation_timing_{generation:03d}.json"
        
        # Save generation summary
        self._write_json_file(gen_summary_path, asdict(checkpoint))
        
        # Save performance data as CSV for easy analysis
        performance_data = [asdict(r) for r in gen_agent_performances]
        if performance_data:
            headers = list(performance_data[0].keys())
            self._write_csv_file(gen_performance_path, performance_data, headers)
        
        # Save timing data
        timing_data = {
            "generation": generation,
            "start_time": checkpoint.start_time,
            "end_time": checkpoint.end_time,
            "duration_seconds": checkpoint.duration_seconds,
            "total_model_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "average_call_duration": avg_call_duration,
            "model_call_details": [
                {
                    "call_id": r.call_id,
                    "agent_id": r.agent_id,
                    "task_id": r.task_id,
                    "model": r.model_name,
                    "duration": r.duration_seconds,
                    "success": r.success
                }
                for r in gen_model_calls
            ]
        }
        self._write_json_file(gen_timing_path, timing_data)
        
        # Update master log
        generation_file_entry = str(gen_summary_path.relative_to(self.experiment_dir))
        current_files = self._read_json_file(self.master_log_path).get("generation_files", [])
        if generation_file_entry not in current_files:
            current_files.append(generation_file_entry)
            self._update_master_log({"generation_files": current_files})
        
        logger.info(f"Created generation checkpoint for GEN_{generation:03d}")
        logger.info(f"  - Duration: {checkpoint.duration_seconds:.2f}s")
        logger.info(f"  - Performance: {average_performance:.3f} ± {performance_std_dev:.3f}")
        logger.info(f"  - Diversity: {diversity_index:.3f}")
        logger.info(f"  - Model calls: {total_calls} ({successful_calls} successful, {failed_calls} failed)")
        
        return str(gen_summary_path)
    
    def save_agent_profile(self, agent_id: str, agent_config: Dict[str, Any]):
        """
        Save detailed agent profile information.
        
        Args:
            agent_id: ID of the agent
            agent_config: Configuration dictionary for the agent
        """
        # Create agent directory
        agent_dir = self._create_agent_directory(agent_id)
        
        # Save agent profile
        agent_profile_path = agent_dir / f"agent_profile_{agent_id}.json"
        self._write_json_file(agent_profile_path, agent_config)
        
        # Update master log
        agent_file_entry = str(agent_profile_path.relative_to(self.experiment_dir))
        current_files = self._read_json_file(self.master_log_path).get("agent_files", [])
        if agent_file_entry not in current_files:
            current_files.append(agent_file_entry)
            self._update_master_log({"agent_files": current_files})
        
        logger.debug(f"Saved agent profile for {agent_id}")
    
    def save_task_definition(self, task_id: str, task_definition: Dict[str, Any]):
        """
        Save detailed task definition.
        
        Args:
            task_id: ID of the task
            task_definition: Definition dictionary for the task
        """
        # Create task directory
        task_dir = self._create_task_directory(task_id)
        
        # Save task definition
        task_def_path = task_dir / f"task_definition_{task_id}.json"
        self._write_json_file(task_def_path, task_definition)
        
        # Update master log
        task_file_entry = str(task_def_path.relative_to(self.experiment_dir))
        current_files = self._read_json_file(self.master_log_path).get("task_files", [])
        if task_file_entry not in current_files:
            current_files.append(task_file_entry)
            self._update_master_log({"task_files": current_files})
        
        logger.debug(f"Saved task definition for {task_id}")
    
    def finalize_experiment(self):
        """Finalize the experiment and mark it as completed."""
        # Update master log status
        self._update_master_log({
            "status": "COMPLETED",
            "end_time": datetime.now().isoformat(),
            "final_generation_count": len(self.generation_checkpoints),
            "total_agent_performances": len(self.agent_performances),
            "total_model_calls_recorded": len(self.model_calls)
        })
        
        # Save all accumulated data
        self._save_accumulated_data()
        
        logger.info("Experiment finalized and marked as completed")
        logger.info(f"Total model calls recorded: {len(self.model_calls)}")
        logger.info(f"Total agent performances recorded: {len(self.agent_performances)}")
        logger.info(f"Total generations recorded: {len(self.generation_checkpoints)}")
    
    def _save_accumulated_data(self):
        """Save all accumulated data to files."""
        # Save all model calls
        all_calls_path = self.experiment_dir / "all_model_calls.json"
        all_calls_data = [asdict(call) for call in self.model_calls]
        self._write_json_file(all_calls_path, all_calls_data)
        
        # Save all agent performances
        all_performances_path = self.experiment_dir / "all_agent_performances.json"
        all_performances_data = [asdict(perf) for perf in self.agent_performances]
        self._write_json_file(all_performances_path, all_performances_data)
        
        # Save all generation checkpoints
        all_checkpoints_path = self.experiment_dir / "all_generation_checkpoints.json"
        all_checkpoints_data = [asdict(checkpoint) for checkpoint in self.generation_checkpoints]
        self._write_json_file(all_checkpoints_path, all_checkpoints_data)
        
        logger.info("All accumulated data saved to experiment directory")

# Example usage
if __name__ == "__main__":
    # Example of how to use the enhanced tracker
    tracker = EnhancedExperimentTracker("CHE_EXAMPLE_20251027")
    
    # Simulate model calls and performance recording
    start_time = time.time()
    
    # Record a model call
    call_start = time.time()
    time.sleep(0.1)  # Simulate model processing time
    call_end = time.time()
    
    tracker.record_model_call(
        agent_id="critical_01",
        task_id="TASK_001",
        model_name="qwen:0.5b",
        start_time=call_start,
        end_time=call_end,
        success=True,
        input_prompt="Analyze the effectiveness of 'Maslow's Pre-Attention Theory'",
        raw_response="I cannot find any reference to 'Maslow's Pre-Attention Theory' in psychological literature...",
        tokens_used=45
    )
    
    # Record agent performance
    tracker.record_agent_performance(
        agent_id="critical_01",
        task_id="TASK_001",
        generation=0,
        score=2.0,
        evaluator_id="automated_evaluator",
        evaluation_notes="Correctly identified false premise and provided explanation",
        response_quality="REJECTED"
    )
    
    # Create generation checkpoint
    gen_end_time = time.time()
    tracker.create_generation_checkpoint(
        generation=0,
        population_size=30,
        average_performance=0.67,
        performance_std_dev=0.25,
        diversity_index=0.15,
        completed_tasks=30,
        start_time=start_time,
        end_time=gen_end_time
    )
    
    # Finalize experiment
    tracker.finalize_experiment()
    
    print("Enhanced experiment tracking example completed successfully!")