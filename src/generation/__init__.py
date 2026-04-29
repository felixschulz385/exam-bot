from .base import BaseExamGenerator
from .latex_generator import LatexGenerator
from .word_generator import WordGenerator
from .template import TemplateProcessor

def get_generator_class(backend: str):
    """Resolve the configured document backend to a generator class."""
    normalized = (backend or "latex").strip().lower()
    backends = {
        "latex": LatexGenerator,
        "word": WordGenerator,
    }
    if normalized not in backends:
        raise ValueError(f"Unsupported document backend '{backend}'. Supported backends: {sorted(backends)}")
    return backends[normalized]


__all__ = ['BaseExamGenerator', 'LatexGenerator', 'WordGenerator', 'TemplateProcessor', 'get_generator_class']
