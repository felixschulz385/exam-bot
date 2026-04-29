from datetime import datetime
from pathlib import Path
from typing import Optional


def resolve_path(path_value: Optional[str], base_dir: Path) -> Optional[Path]:
    """Resolve a possibly relative path against a base directory."""
    if not path_value:
        return None

    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate

    return (base_dir / candidate).resolve()


def ensure_directory(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_string() -> str:
    """Return a filesystem-friendly timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
