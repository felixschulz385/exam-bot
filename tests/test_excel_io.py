import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.excel_reader import QuestionBankReader
from src.data.excel_writer import ExamSelectionWriter


def make_question_frame(question_text):
    return pd.DataFrame(
        [
            {
                "ID": "1",
                "Topic": "Topic A",
                "Type": "Single Choice",
                "Question": question_text,
                "Answer 1": "A",
                "Answer 2": "B",
                "Answer 3": "C",
                "Answer 4": "D",
                "Correct": 1,
                "Medium": "",
            }
        ]
    )


class ExcelIOTests(unittest.TestCase):
    def test_reader_defaults_to_v2_sheet(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workbook_path = Path(temp_dir) / "questions.xlsx"
            with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
                make_question_frame("from v1").to_excel(writer, sheet_name="v1", index=False)
                make_question_frame("from v2").to_excel(writer, sheet_name="v2", index=False)

            reader = QuestionBankReader(workbook_path)

        self.assertEqual(reader.df.iloc[0]["Question"], "from v2")

    def test_reader_drops_rows_missing_topic_type_or_question(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workbook_path = Path(temp_dir) / "questions.xlsx"
            rows = pd.DataFrame(
                [
                    {
                        "ID": "1",
                        "Topic": "Topic A",
                        "Type": "Single Choice",
                        "Question": "valid question",
                        "Answer 1": "A",
                        "Answer 2": "B",
                        "Answer 3": "C",
                        "Answer 4": "D",
                        "Correct": 1,
                        "Medium": "",
                    },
                    {
                        "ID": "2",
                        "Topic": "Topic A",
                        "Type": None,
                        "Question": "invalid type row",
                        "Answer 1": "A",
                        "Answer 2": "B",
                        "Answer 3": "C",
                        "Answer 4": "D",
                        "Correct": 1,
                        "Medium": "",
                    },
                    {
                        "ID": "3",
                        "Topic": "",
                        "Type": "Single Choice",
                        "Question": "invalid topic row",
                        "Answer 1": "A",
                        "Answer 2": "B",
                        "Answer 3": "C",
                        "Answer 4": "D",
                        "Correct": 1,
                        "Medium": "",
                    },
                ]
            )
            with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
                rows.to_excel(writer, sheet_name="v2", index=False)

            reader = QuestionBankReader(workbook_path)

        self.assertEqual(len(reader.df), 1)
        self.assertEqual(reader.df.iloc[0]["Question"], "valid question")

    def test_selection_writer_reads_configured_sheet(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workbook_path = Path(temp_dir) / "questions.xlsx"
            output_path = Path(temp_dir) / "selected.xlsx"

            with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
                make_question_frame("from v1").to_excel(writer, sheet_name="v1", index=False)
                make_question_frame("from v2").to_excel(writer, sheet_name="v2", index=False)

            writer = ExamSelectionWriter(
                workbook_path,
                sheet_name="v2",
            )
            writer.export_selected_questions([0], output_path=output_path)

            written = pd.read_excel(output_path)

        self.assertEqual(written.iloc[0]["Question"], "from v2")
        self.assertEqual(len(written), 1)


if __name__ == "__main__":
    unittest.main()
