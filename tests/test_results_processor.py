import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src.analysis.results_processor import ExamResultsProcessor


class ResultsProcessorTests(unittest.TestCase):
    def test_load_marking_sheet_normalizes_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "marking.xlsx"
            df = pd.DataFrame(
                {
                    "Question": ["Question 1", "Question 2", "Total"],
                    "Answer Key": ["A", "B", ""],
                    "Points": [2, 2, 4],
                }
            )
            df.to_excel(path, index=False)

            loaded = ExamResultsProcessor(output_dir=Path(temp_dir)).load_marking_sheet(path)

        self.assertEqual(list(loaded["Question #"]), [1, 2])
        self.assertEqual(list(loaded["Correct Answer Letter"]), ["A", "B"])

    def test_score_and_stats_pipeline(self):
        processor = ExamResultsProcessor(output_dir=Path("/tmp"))
        results_df = pd.DataFrame(
            {
                "Question": [0, 1],
                "Alice": ["A", "B"],
                "Bob": ["B", "B"],
            }
        )
        marking_df = pd.DataFrame(
            {
                "Question #": [0, 1],
                "Correct Answer Letter": ["A", "B"],
                "Points": [2, 3],
            }
        )

        student_scores = processor.calculate_student_scores(results_df, marking_df)
        question_stats = processor.generate_question_statistics(results_df, marking_df, student_scores)

        self.assertEqual(student_scores["Alice_total"].max(), 5)
        self.assertEqual(student_scores["Bob_total"].max(), 3)
        self.assertEqual(question_stats.loc[0, "# Correct"], 1)
        self.assertEqual(question_stats.loc[1, "# Correct"], 2)

    def test_generate_summary_report_writes_excel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = ExamResultsProcessor(output_dir=Path(temp_dir))
            student_scores = pd.DataFrame(
                {
                    "Alice_points": [2, 3],
                    "Alice_total": [5, 5],
                    "Bob_points": [0, 3],
                    "Bob_total": [3, 3],
                    "Question #": [0, 1],
                    "Correct Answer": ["A", "B"],
                    "Points Possible": [2, 3],
                }
            )
            question_stats = pd.DataFrame(
                {
                    "Question #": [0, 1],
                    "Correct Answer": ["A", "B"],
                    "Points": [2, 3],
                    "# Correct": [1, 2],
                    "% Correct": [50.0, 100.0],
                    "A": [1, 0],
                    "B": [1, 2],
                    "C": [0, 0],
                    "D": [0, 0],
                    "Blank": [0, 0],
                    "Discrimination": [0.0, 0.0],
                }
            )

            with mock.patch.object(processor, "_create_question_difficulty_chart", return_value=Path(temp_dir) / "q.png"), \
                mock.patch.object(processor, "_create_score_distribution_chart", return_value=Path(temp_dir) / "s.png"):
                output_path = processor.generate_summary_report(student_scores, question_stats, "report")

            self.assertTrue(output_path.exists())
            written = pd.read_excel(output_path, sheet_name="Student Summary", index_col=0)
            self.assertEqual(list(written.index), ["Alice", "Bob"])


if __name__ == "__main__":
    unittest.main()
