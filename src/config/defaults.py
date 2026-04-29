from pathlib import Path
from typing import Dict, Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "exam_settings.yaml"


def _load_default_config() -> Dict[str, Any]:
    """Load the central human-editable default settings."""
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Default config at {DEFAULT_CONFIG_PATH} must be a mapping")

    return config

DEFAULT_CONFIG = _load_default_config()

# Schema for validating custom point mappings
POINT_MAPPING_SCHEMA = {
    "type": "object",
    "additionalProperties": {
        "type": "integer",
        "minimum": 1
    },
    "minProperties": 1
}

# Question types with defaults (expand as needed)
DEFAULT_QUESTION_TYPES = [
    "Multiple Choice",
    "True/False",
    "Short Answer",
    "Essay"
]

# Selection method descriptions for documentation
SELECTION_METHOD_DESCRIPTIONS = {
    "unified": "Point-aware selection with semantic diversity and block-question support"
}
