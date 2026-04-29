import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.generation.marking_sheet import MarkingSheetGenerator


class MarkingSheetTests(unittest.TestCase):
    def test_generate_marking_sheet_with_randomization(self):
        questions = pd.DataFrame(
            [
                {
                    "Topic": "A",
                    "Type": "Single Choice",
                    "Question": "Question 1",
                    "Answer 1": "One",
                    "Answer 2": "Two",
                    "Answer 3": "Three",
                    "Answer 4": "Four",
                    "Correct": 2,
                }
            ]
        )
        config = {
            "exam": {"title": "Sample Exam"},
            "document": {"randomize_answers": True},
            "grading": {"points_per_type": {"Single Choice": 2}},
        }
        answer_mappings = {1: {1: "B", 2: "D", 3: "A", 4: "C"}}

        with tempfile.TemporaryDirectory() as temp_dir:
            generator = MarkingSheetGenerator(output_dir=Path(temp_dir))
            output_path = generator.generate_marking_sheet(
                questions,
                config,
                timestamp="20260423_120000",
                answer_mappings=answer_mappings,
            )

            compact = pd.read_excel(output_path, sheet_name="Marking Sheet")
            mapping = pd.read_excel(output_path, sheet_name="Answer Randomization")

        self.assertEqual(compact.loc[0, "Correct Answer Letter"], "D")
        self.assertEqual(compact.loc[0, "Points"], 2)
        self.assertIn("Orig A", mapping.columns)

    def test_generate_marking_sheet_accepts_letter_correct_answers(self):
        questions = pd.DataFrame(
            [
                {
                    "Topic": "A",
                    "Type": "Single Choice",
                    "Question": "Question 1",
                    "Answer 1": "One",
                    "Answer 2": "Two",
                    "Answer 3": "Three",
                    "Answer 4": "Four",
                    "Correct": "C",
                }
            ]
        )
        config = {
            "exam": {"title": "Sample Exam"},
            "document": {"randomize_answers": False},
            "grading": {"points_per_type": {"Single Choice": 2}},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            generator = MarkingSheetGenerator(output_dir=Path(temp_dir))
            output_path = generator.generate_marking_sheet(
                questions,
                config,
                timestamp="20260423_120000",
            )

            compact = pd.read_excel(output_path, sheet_name="Marking Sheet")
            detailed = pd.read_excel(output_path, sheet_name="Detailed")

        self.assertEqual(compact.loc[0, "Correct Answer Letter"], "C")
        self.assertEqual(detailed.loc[0, "Correct Answer #"], 3)
        self.assertEqual(detailed.loc[0, "Correct Answer Text"], "Three")


if __name__ == "__main__":
    unittest.main()
