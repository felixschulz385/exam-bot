from typing import Dict, Any

# Default configuration values
DEFAULT_CONFIG = {
    # Input/output settings
    "io": {
        "question_bank_path": "questions.xlsx",
        "output_dir": "output",
        "template_path": "templates/default.docx",
        "timestamp_output": True
    },
    
    # Exam structure
    "exam": {
        "title": "Exam",
        "total_questions": 30,
        "randomize_order": True,
        "header": "",
        "footer": "",
        "instructions": "Select the best answer for each question."
    },
    
    # Selection settings
    "selection": {
        "method": "stratified",  # Options: random, stratified, semantic, clustered
        "seed": None,  # Set for reproducible selection
        
        # Only used if method is 'semantic' or 'clustered'
        "semantic": {
            "model": "paraphrase-multilingual-MiniLM-L12-v2",
            "cache_dir": "cache",
            "cluster_ratio": 0.3  # Used only by clustered method
        }
    },
    
    # Grading scheme
    "grading": {
        "topic_points": {
            # Example: "Topic A": 10
        },
        "type_points": {
            # Example: "Multiple Choice": 5
        }
    },
    
    # Document generation
    "document": {
        "student_version": True,
        "instructor_version": True,
        "include_topic_headers": True,
        "include_type_headers": False,
        "format": {
            "font": "Calibri",
            "font_size": 11,
            "line_spacing": 1.15
        }
    },
    
    # Logging
    "logging": {
        "level": "INFO",
        "file": "exam_generator.log"
    }
}

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
    "random": "Basic random selection of questions",
    "stratified": "Selection balanced across topics and question types",
    "semantic": "Selection that maximizes semantic diversity",
    "clustered": "Cluster-based selection that respects point distributions"
}