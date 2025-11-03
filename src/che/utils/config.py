"""
Environment Configuration Management for Cognitive Heterogeneity Validation

This module provides centralized configuration management for the cognitive heterogeneity
validation system, supporting environment-specific settings and runtime configuration.

Authors: CHE Research Team
Date: 2025-10-19
"""

import os
import json
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CHEConfig:
    """
    Configuration data class for cognitive heterogeneity validation system.
    
    This class encapsulates all configuration parameters for the system,
    including experiment parameters, model settings, and system behavior.
    """
    
    # Experiment parameters
    population_size: int = 30
    generations: int = 15
    agent_ratios: Dict[str, float] = field(default_factory=lambda: {
        "critical": 0.33,
        "awakened": 0.33,
        "standard": 0.34
    })
    
    # Model settings
    models: list = field(default_factory=lambda: ["qwen:0.5b", "gemma:2b"])
    default_model: str = "qwen:0.5b"
    
    # Evaluation settings
    use_ai_evaluator: bool = True
    evaluation_threshold: float = 1.0
    
    # System settings
    random_seed: Optional[int] = None
    checkpoint_interval: int = 5
    log_level: str = "INFO"
    
    # Performance settings
    max_concurrent_agents: int = 10
    timeout_seconds: int = 300
    
    # Experiment parameters
    heterogeneous_enabled: bool = True
    homogeneous_enabled: bool = True
    agent_types: list = field(default_factory=lambda: ["critical", "awakened", "standard"])
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.population_size <= 0:
            raise ValueError("Population size must be positive")
        
        if self.generations <= 0:
            raise ValueError("Generations must be positive")
        
        if self.checkpoint_interval <= 0:
            raise ValueError("Checkpoint interval must be positive")
        
        # Validate agent ratios sum to approximately 1.0
        ratio_sum = sum(self.agent_ratios.values())
        if abs(ratio_sum - 1.0) > 0.01:  # Allow small floating point errors
            raise ValueError(f"Agent ratios must sum to 1.0, got {ratio_sum}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CHEConfig':
        """
        Create configuration from dictionary.
        
        Args:
            data: Dictionary representation of configuration
            
        Returns:
            New configuration instance
        """
        # Handle the agent_ratios field specially
        if 'agent_ratios' in data and isinstance(data['agent_ratios'], dict):
            agent_ratios = data['agent_ratios']
        else:
            agent_ratios = {"critical": 0.33, "awakened": 0.33, "standard": 0.34}
        
        # Handle the models field specially
        if 'models' in data and isinstance(data['models'], list):
            models = data['models']
        else:
            models = ["qwen:0.5b", "gemma:2b"]
        
        # Handle the agent_types field specially
        if 'agent_types' in data and isinstance(data['agent_types'], list):
            agent_types = data['agent_types']
        else:
            agent_types = ["critical", "awakened", "standard"]
        
        # Create instance with processed fields
        config = cls(
            population_size=data.get('population_size', 30),
            generations=data.get('generations', 15),
            agent_ratios=agent_ratios,
            models=models,
            default_model=data.get('default_model', 'qwen:0.5b'),
            use_ai_evaluator=data.get('use_ai_evaluator', True),
            evaluation_threshold=data.get('evaluation_threshold', 1.0),
            random_seed=data.get('random_seed'),
            checkpoint_interval=data.get('checkpoint_interval', 5),
            log_level=data.get('log_level', 'INFO'),
            max_concurrent_agents=data.get('max_concurrent_agents', 10),
            timeout_seconds=data.get('timeout_seconds', 300),
            heterogeneous_enabled=data.get('heterogeneous_enabled', True),
            homogeneous_enabled=data.get('homogeneous_enabled', True),
            agent_types=agent_types
        )
        
        return config


class ConfigManager:
    """
    Configuration manager for the cognitive heterogeneity validation system.
    
    This class provides methods for loading, saving, and managing configuration
    from various sources including environment variables, JSON files, and defaults.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Optional path to JSON configuration file
        """
        self.config_file = config_file
        self.config = CHEConfig()
        
        # Load configuration from file if specified
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
        
        # Override with environment variables
        self.load_from_environment()
    
    def load_from_file(self, filepath: str) -> None:
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to JSON configuration file
            
        Raises:
            FileNotFoundError: If configuration file does not exist
            json.JSONDecodeError: If configuration file is not valid JSON
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.config = CHEConfig.from_dict(data)
        self.config_file = filepath
        logger.info(f"Loaded configuration from {filepath}")
    
    def save_to_file(self, filepath: Optional[str] = None) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Optional path to save configuration file.
                      If not provided, uses the loaded config file path.
                      
        Raises:
            ValueError: If no file path is available
        """
        save_path = filepath or self.config_file
        if not save_path:
            raise ValueError("No file path specified for saving configuration")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved configuration to {save_path}")
    
    def load_from_environment(self) -> None:
        """
        Load configuration overrides from environment variables.
        
        Environment variables should be prefixed with 'CHE_' and use
        uppercase names matching configuration fields.
        """
        # Map environment variable names to config fields
        env_mapping = {
            'CHE_POPULATION_SIZE': 'population_size',
            'CHE_GENERATIONS': 'generations',
            'CHE_DEFAULT_MODEL': 'default_model',
            'CHE_USE_AI_EVALUATOR': 'use_ai_evaluator',
            'CHE_EVALUATION_THRESHOLD': 'evaluation_threshold',
            'CHE_RANDOM_SEED': 'random_seed',
            'CHE_CHECKPOINT_INTERVAL': 'checkpoint_interval',
            'CHE_LOG_LEVEL': 'log_level',
            'CHE_MAX_CONCURRENT_AGENTS': 'max_concurrent_agents',
            'CHE_TIMEOUT_SECONDS': 'timeout_seconds',
            'CHE_HETEROGENEOUS_ENABLED': 'heterogeneous_enabled',
            'CHE_HOMOGENEOUS_ENABLED': 'homogeneous_enabled'
        }
        
        for env_var, config_field in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                # Convert string values to appropriate types
                if config_field in ['population_size', 'generations', 'random_seed', 
                                  'checkpoint_interval', 'max_concurrent_agents', 'timeout_seconds']:
                    try:
                        setattr(self.config, config_field, int(value))
                    except ValueError:
                        logger.warning(f"Invalid integer value for {env_var}: {value}")
                elif config_field in ['use_ai_evaluator', 'heterogeneous_enabled', 'homogeneous_enabled']:
                    setattr(self.config, config_field, value.lower() in ['true', '1', 'yes'])
                else:
                    setattr(self.config, config_field, value)
        
        logger.info("Loaded configuration from environment variables")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key (field name)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return getattr(self.config, key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value by key.
        
        Args:
            key: Configuration key (field name)
            value: Value to set
            
        Raises:
            AttributeError: If key is not a valid configuration field
        """
        if not hasattr(self.config, key):
            raise AttributeError(f"Invalid configuration field: {key}")
        
        setattr(self.config, key, value)
        logger.debug(f"Set configuration {key} = {value}")
    
    def validate(self) -> bool:
        """
        Validate the current configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Trigger validation by accessing config
            _ = self.config.population_size
            _ = self.config.generations
            _ = self.config.checkpoint_interval
            return True
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert current configuration to dictionary.
        
        Returns:
            Dictionary representation of current configuration
        """
        return self.config.to_dict()


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_file: Optional[str] = None) -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Args:
        config_file: Optional path to configuration file
        
    Returns:
        Global configuration manager instance
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
    
    return _config_manager


def get_config() -> CHEConfig:
    """
    Get the current configuration.
    
    Returns:
        Current configuration instance
    """
    return get_config_manager().config


def reload_config(config_file: Optional[str] = None) -> None:
    """
    Reload configuration from file or environment.
    
    Args:
        config_file: Optional path to configuration file
    """
    global _config_manager
    _config_manager = ConfigManager(config_file)
    logger.info("Reloaded configuration")


class ExperimentConfig:
    """Standard experiment configuration."""
    
    MODEL = "gemma3:latest"
    POPULATION_SIZE = 30
    GENERATIONS = 15
    EVALUATION_THRESHOLD = 1.0
    
    @classmethod
    def get_agent_config(cls, prompt_type: str) -> Dict[str, Any]:
        """Get standard agent configuration."""
        from ..prompts import get_prompt, PromptType
        
        prompt_types = {
            "critical": PromptType.CRITICAL,
            "awakened": PromptType.AWAKENED,
            "standard": PromptType.STANDARD
        }
        
        return {
            "model": cls.MODEL,
            "prompt": get_prompt(prompt_types[prompt_type])
        }