import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging
from jsonschema import validate, ValidationError
from .defaults import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Load and validate configuration settings for exam generation."""
    
    def __init__(self, schema_path: Optional[Path] = None):
        """Initialize config loader with optional JSON schema path.
        
        Args:
            schema_path: Path to JSON schema for validating config files
        """
        self.schema_path = schema_path
        self.schema = None
        
        if schema_path:
            try:
                with open(schema_path, 'r') as f:
                    self.schema = json.load(f)
                logger.info(f"Loaded configuration schema from {schema_path}")
            except Exception as e:
                logger.warning(f"Failed to load schema from {schema_path}: {e}")
    
    def load_config(self, 
                   config_path: Path,
                   apply_defaults: bool = True) -> Dict[str, Any]:
        """Load configuration from file with optional validation.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
            apply_defaults: Whether to apply default values for missing fields
            
        Returns:
            Dictionary containing configuration settings
        """
        logger.info(f"Loading configuration from {config_path}")
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        # Determine file format from extension
        file_ext = config_path.suffix.lower()
        
        try:
            if file_ext == '.yaml' or file_ext == '.yml':
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            elif file_ext == '.json':
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {file_ext}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
            
        # Apply defaults if needed
        if apply_defaults:
            config = self._apply_defaults(config)
            
        # Validate against schema if available
        if self.schema:
            try:
                validate(instance=config, schema=self.schema)
                logger.info("Configuration validated successfully")
            except ValidationError as e:
                logger.error(f"Configuration validation failed: {e}")
                raise
        
        return config
    
    def _apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values for any missing configuration settings.
        
        Args:
            config: User-provided configuration dictionary
            
        Returns:
            Configuration with defaults applied for missing values
        """
        result = DEFAULT_CONFIG.copy()
        
        # Recursively update defaults with user config
        self._recursive_update(result, config)
        
        return result
    
    def _recursive_update(self, 
                         target: Dict[str, Any], 
                         source: Dict[str, Any]) -> None:
        """Recursively update target dictionary with values from source.
        
        Args:
            target: Target dictionary to update (modified in place)
            source: Source dictionary with new values
        """
        for key, value in source.items():
            # If both are dicts, recursively update
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._recursive_update(target[key], value)
            else:
                # Otherwise replace/add value
                target[key] = value
    
    def validate_grading_scheme(self, 
                               topic_points: Dict[str, int], 
                               type_points: Dict[str, int]) -> bool:
        """Validate that the grading scheme has positive values.
        
        Args:
            topic_points: Dictionary mapping topics to point values
            type_points: Dictionary mapping question types to point values
            
        Returns:
            True if grading scheme is valid, False otherwise
        """
        # Check for negative or zero values
        for topic, points in topic_points.items():
            if points <= 0:
                logger.error(f"Topic '{topic}' has invalid point value: {points}")
                return False
                
        for qtype, points in type_points.items():
            if points <= 0:
                logger.error(f"Question type '{qtype}' has invalid point value: {points}")
                return False
        
        # Check that at least one topic and type are defined
        if not topic_points:
            logger.error("No topics defined in grading scheme")
            return False
            
        if not type_points:
            logger.error("No question types defined in grading scheme")
            return False
            
        return True
    
    def save_config(self, config: Dict[str, Any], output_path: Path) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration dictionary
            output_path: Path to save configuration file
        """
        # Determine file format from extension
        file_ext = output_path.suffix.lower()
        
        try:
            if file_ext == '.yaml' or file_ext == '.yml':
                with open(output_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            elif file_ext == '.json':
                with open(output_path, 'w') as f:
                    json.dump(config, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {file_ext}")
                
            logger.info(f"Configuration saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise