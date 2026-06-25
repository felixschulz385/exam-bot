import tempfile
import unittest
from pathlib import Path
import pandas as pd

from src.analysis.results_processor import ExamResultsProcessor


class ResultsProcessorTests(unittest.TestCase):
    def test_load_results_accepts_semicolon_csv_export(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "results.csv"
            path.write_text(
                '"StudentNumber";"Version";"A0";"A1";"RF60"\n'
                '"";"";"C";"D";""\n'
                '"2024000001";"";" A ";"b";""\n'
                '"2024000002";"";"D";"";" C "\n',
                encoding="utf-8",
            )

            loaded = ExamResultsProcessor(output_dir=Path(temp_dir)).load_results(path)

        self.assertEqual(list(loaded["Question"]), [0, 1, 60])
        self.assertEqual(list(loaded["2024000001"][:2]), ["A", "B"])
        self.assertTrue(pd.isna(loaded["2024000001"].iloc[2]))
        self.assertEqual(loaded["2024000002"].iloc[0], "D")
        self.assertTrue(pd.isna(loaded["2024000002"].iloc[1]))
        self.assertEqual(loaded["2024000002"].iloc[2], "C")

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
                "Question": [0, 1, 60],
                "Alice": ["A", "B", ""],
                "Bob": ["B", "B", "C"],
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
        self.assertEqual(list(student_scores["Question #"]), [0, 1])
        self.assertEqual(list(question_stats["Question #"]), [0, 1])
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
            results_df = pd.DataFrame(
                {
                    "Question": [0, 1],
                    "Alice": ["A", "B"],
                    "Bob": ["B", "B"],
                }
            )

            output_path = processor.generate_summary_report(
                student_scores,
                question_stats,
                "report",
                results_df=results_df,
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.name, "report_grading_report.xlsx")
            with pd.ExcelFile(output_path) as workbook:
                self.assertEqual(
                    workbook.sheet_names,
                    ["Overview", "Student Scores", "Question Analysis", "Answer Matrix", "Score Matrix"],
                )
            written = pd.read_excel(output_path, sheet_name="Student Scores")
            self.assertEqual(list(written["Student"]), ["Alice", "Bob"])
            answers = pd.read_excel(output_path, sheet_name="Answer Matrix")
            self.assertEqual(list(answers["Alice"]), ["A", "B"])


if __name__ == "__main__":
    unittest.main()
