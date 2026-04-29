from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.utils import ensure_directory


class BaseExamGenerator(ABC):
    """Shared interface and helpers for exam document backends."""

    def __init__(self, output_dir: Path = Path("output")):
        self.output_dir = ensure_directory(Path(output_dir))
        self.doc_config: Dict[str, Any] = {}
        self.seed: Optional[int] = None

    @abstractmethod
    def generate_exam(
        self,
        questions: pd.DataFrame,
        config: Dict[str, Any],
    ) -> Tuple[Optional[Path], Optional[Path], Dict]:
        """Generate the student and instructor outputs plus answer mappings."""

    def _normalize_correct_answer(self, raw_value: object) -> Optional[int]:
        """Normalize numeric or letter answer keys to a 1-based answer index."""
        if raw_value is None or pd.isna(raw_value):
            return None

        if isinstance(raw_value, str):
            stripped = raw_value.strip().upper()
            if not stripped:
                return None
            if stripped in {"A", "B", "C", "D"}:
                return ord(stripped) - 64
            try:
                numeric_value = int(stripped)
            except ValueError:
                return None
            return numeric_value if 1 <= numeric_value <= 4 else None

        try:
            numeric_value = int(raw_value)
        except (TypeError, ValueError):
            return None

        return numeric_value if 1 <= numeric_value <= 4 else None

    def _should_randomize_answers(
        self,
        question_type: str,
        answer_choices: List[Tuple[int, str, str, bool]],
    ) -> bool:
        """Keep True/False order stable even when answer randomization is enabled."""
        if question_type == "True/False":
            return False

        normalized_answers = [answer_text.strip().lower() for _, _, answer_text, _ in answer_choices]
        if normalized_answers == ["true", "false"]:
            return False

        return True

    def _build_answer_choices(
        self,
        question_data: pd.Series,
        text_formatter: Optional[Callable[[str], str]] = None,
    ) -> List[Tuple[int, str, str, bool]]:
        """Build answer choice tuples with normalized correctness."""
        correct_answer_num = self._normalize_correct_answer(question_data.get("Correct"))
        answer_choices: List[Tuple[int, str, str, bool]] = []

        for answer_index in range(1, 5):
            answer_key = f"Answer {answer_index}"
            if answer_key not in question_data or pd.isna(question_data[answer_key]):
                continue

            answer_text = str(question_data[answer_key])
            if answer_text.endswith("."):
                answer_text = answer_text[:-1]
            if text_formatter is not None:
                answer_text = text_formatter(answer_text)

            answer_choices.append(
                (
                    answer_index,
                    chr(64 + answer_index),
                    answer_text,
                    correct_answer_num == answer_index,
                )
            )

        return answer_choices
