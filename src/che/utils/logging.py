"""
Logging and Error Handling Infrastructure for Cognitive Heterogeneity Validation

This module provides centralized logging configuration and error handling utilities
for the cognitive heterogeneity validation system.

Authors: CHE Research Team
Date: 2025-10-19
"""

import logging
import sys
from typing import Optional, Dict, Any
import traceback
from functools import wraps
import time


# Default logging configuration
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


def setup_logging(log_level: int = DEFAULT_LOG_LEVEL, 
                  log_format: str = DEFAULT_LOG_FORMAT,
                  log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up centralized logging for the cognitive heterogeneity system.
    
    Args:
        log_level: The logging level (default: INFO)
        log_format: The log message format (default: timestamp - logger - level - message)
        log_file: Optional file to write logs to (default: None, log to console only)
        
    Returns:
        The configured root logger
    """
    # Configure the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class CHEException(Exception):
    """
    Base exception class for cognitive heterogeneity validation system.
    
    All custom exceptions in the system should inherit from this class.
    """
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        """
        Initialize the exception.
        
        Args:
            message: The error message
            error_code: Optional error code for categorization
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
    
    def __str__(self) -> str:
        """String representation of the exception."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class AgentException(CHEException):
    """Exception related to agent operations."""
    pass


class TaskException(CHEException):
    """Exception related to task operations."""
    pass


class EcosystemException(CHEException):
    """Exception related to ecosystem operations."""
    pass


class EvaluationException(CHEException):
    """Exception related to evaluation operations."""
    pass


def handle_exceptions(func):
    """
    Decorator to handle exceptions in functions and log them appropriately.
    
    This decorator catches all exceptions, logs them, and re-raises CHEException
    or wraps other exceptions in CHEException.
    
    Args:
        func: The function to decorate
        
    Returns:
        Wrapped function that handles exceptions
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except CHEException:
            # Re-raise CHE exceptions as-is
            raise
        except Exception as e:
            # Wrap other exceptions in CHEException
            logging.error(f"Unexpected error in {func.__name__}: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            raise CHEException(f"Error in {func.__name__}: {str(e)}") from e
    
    return wrapper


def log_execution(func):
    """
    Decorator to log function entry and exit.
    
    This decorator logs when a function is called and when it returns,
    including execution time for performance monitoring.
    
    Args:
        func: The function to decorate
        
    Returns:
        Wrapped function that logs execution
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logging.debug(f"Entering {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logging.debug(f"Exiting {func.__name__} - Execution time: {execution_time:.4f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logging.error(f"Error in {func.__name__} after {execution_time:.4f}s: {str(e)}")
            raise
    
    return wrapper


# Convenience functions for common logging operations


def log_agent_action(agent_id: str, action: str, details: Optional[Dict[str, Any]] = None) -> None:
    """
    Log an agent action.
    
    Args:
        agent_id: The ID of the agent performing the action
        action: Description of the action
        details: Optional additional details about the action
    """
    message = f"Agent {agent_id}: {action}"
    if details:
        message += f" - Details: {details}"
    logging.info(message)


def log_ecosystem_event(event: str, generation: int, details: Optional[Dict[str, Any]] = None) -> None:
    """
    Log an ecosystem event.
    
    Args:
        event: Description of the event
        generation: Generation number
        details: Optional additional details about the event
    """
    message = f"Ecosystem (Gen {generation}): {event}"
    if details:
        message += f" - Details: {details}"
    logging.info(message)


def log_evaluation_result(agent_id: str, score: float, threshold: float = 1.0) -> None:
    """
    Log an evaluation result.
    
    Args:
        agent_id: The ID of the agent being evaluated
        score: The evaluation score
        threshold: The threshold for passing evaluation (default: 1.0)
    """
    status = "PASS" if score >= threshold else "FAIL"
    logging.info(f"Evaluation Result - Agent {agent_id}: {status} (Score: {score:.2f}, Threshold: {threshold})")


def log_experiment_start(experiment_type: str, config: Dict[str, Any]) -> None:
    """
    Log the start of an experiment.
    
    Args:
        experiment_type: Type of experiment being started
        config: Experiment configuration
    """
    logging.info(f"Starting {experiment_type} experiment")
    logging.info(f"Configuration: {config}")


def log_experiment_end(experiment_type: str, duration: float, results: Dict[str, Any]) -> None:
    """
    Log the end of an experiment.
    
    Args:
        experiment_type: Type of experiment that ended
        duration: Duration of the experiment in seconds
        results: Experiment results
    """
    logging.info(f"Completed {experiment_type} experiment in {duration:.2f} seconds")
    logging.info(f"Results: {results}")


def log_population_change(change_type: str, generation: int, 
                         before_count: int, after_count: int) -> None:
    """
    Log a population change event.
    
    Args:
        change_type: Type of population change (e.g., "evolution", "mutation")
        generation: Generation number
        before_count: Population size before change
        after_count: Population size after change
    """
    logging.info(f"Population {change_type} (Gen {generation}): {before_count} → {after_count} agents")


def log_checkpoint_saved(generation: int, filepath: str) -> None:
    """
    Log that a checkpoint was saved.
    
    Args:
        generation: Generation number
        filepath: Path to the checkpoint file
    """
    logging.info(f"Checkpoint saved for generation {generation} to {filepath}")


def log_checkpoint_loaded(generation: int, filepath: str) -> None:
    """
    Log that a checkpoint was loaded.
    
    Args:
        generation: Generation number
        filepath: Path to the checkpoint file
    """
    logging.info(f"Checkpoint loaded for generation {generation} from {filepath}")