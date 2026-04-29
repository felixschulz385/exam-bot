import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.generation.word_generator import WordGenerator


class WordGeneratorTests(unittest.TestCase):
    def test_format_question_text_collapses_whitespace(self):
        generator = WordGenerator(output_dir=Path("/tmp"))
        self.assertEqual(generator._format_question_text("A   question\nwith\tspace"), "A question with space")

    def test_normalize_correct_answer_accepts_letters(self):
        generator = WordGenerator(output_dir=Path("/tmp"))
        self.assertEqual(generator._normalize_correct_answer("C"), 3)
        self.assertEqual(generator._normalize_correct_answer("2"), 2)
        self.assertIsNone(generator._normalize_correct_answer("Z"))

    def test_true_false_answers_are_never_randomized(self):
        generator = WordGenerator(output_dir=Path("/tmp"))
        answer_choices = [
            (1, "A", "True", True),
            (2, "B", "False", False),
        ]
        self.assertFalse(generator._should_randomize_answers("True/False", answer_choices))

    def test_format_question_text_renders_common_latex(self):
        generator = WordGenerator(output_dir=Path("/tmp"))
        text = r"In fuzzy RD, \( \frac{\text{jump in }Y}{\text{jump in }D} \) identifies \rho^2 with X_{it} and \gamma."
        formatted = generator._format_question_text(text)
        self.assertIn("(jump in Y)/(jump in D)", formatted)
        self.assertIn("rho²", formatted)
        self.assertIn("Xᵢₜ", formatted)
        self.assertIn("gamma", formatted)

    def test_convert_to_pdf_returns_none_for_missing_docx(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = WordGenerator(output_dir=Path(temp_dir))
            result = generator.convert_to_pdf(Path(temp_dir) / "missing.docx")

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
