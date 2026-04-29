import tempfile
import unittest
from pathlib import Path
from unittest import mock

import main as main_module


class MainOrchestrationTests(unittest.TestCase):
    def test_main_generation_uses_configured_sheet_for_selection_export(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "exam.yaml"
            question_bank = root / "questions.xlsx"

            config_path.write_text("exam: {}\n", encoding="utf-8")
            question_bank.touch()

            loaded_config = {
                "io": {
                    "question_bank_path": str(question_bank),
                    "question_bank_sheet": "v2",
                    "output_dir": str(root / "output"),
                    "timestamp_output": False,
                },
                "selection": {
                    "method": "unified",
                    "semantic": {},
                    "target_points": 2,
                    "topics": ["A"],
                    "type_ratios": {"Single Choice": 1},
                },
                "grading": {"points_per_type": {"Single Choice": 2}},
                "exam": {"title": "Test Exam"},
                "document": {"backend": "latex"},
                "logging": {"level": "INFO"},
            }

            fake_question_bank = mock.Mock()
            fake_question_bank.get_questions_by_indices.return_value = "selected-questions"

            fake_writer = mock.Mock()
            fake_writer.export_selected_questions.return_value = root / "selected.xlsx"

            fake_generator = mock.Mock()
            fake_generator.generate_exam.return_value = (root / "student.docx", root / "instructor.docx", {})

            fake_marking_generator = mock.Mock()
            fake_marking_generator.generate_marking_sheet.return_value = root / "marking.xlsx"

            with mock.patch.object(main_module.ConfigLoader, "load_config", return_value=loaded_config), \
                mock.patch.object(main_module, "QuestionBankReader", return_value=fake_question_bank) as reader_cls, \
                mock.patch.object(main_module, "select_questions", return_value=[0]), \
                mock.patch.object(main_module, "ExamSelectionWriter", return_value=fake_writer) as writer_cls, \
                mock.patch.object(main_module, "get_document_generator_class", return_value=lambda **kwargs: fake_generator), \
                mock.patch.object(main_module, "MarkingSheetGenerator", return_value=fake_marking_generator), \
                mock.patch.object(main_module.shutil, "copy"):
                main_module.main(config_path=config_path)

        reader_args, reader_kwargs = reader_cls.call_args
        self.assertEqual(Path(reader_args[0]).name, "questions.xlsx")
        self.assertEqual(reader_kwargs["sheet_name"], "v2")

        writer_args, writer_kwargs = writer_cls.call_args
        self.assertEqual(Path(writer_args[0]).name, "questions.xlsx")
        self.assertEqual(writer_kwargs["sheet_name"], "v2")

    def test_process_results_uses_processor_pipeline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results_path = root / "results.xlsx"
            marking_path = root / "marking.xlsx"
            results_path.touch()
            marking_path.touch()

            fake_processor = mock.Mock()
            fake_processor.load_results.return_value = "results-df"
            fake_processor.load_marking_sheet.return_value = "marking-df"
            fake_processor.calculate_student_scores.return_value = "scores-df"
            fake_processor.generate_question_statistics.return_value = "stats-df"
            fake_processor.generate_summary_report.return_value = root / "summary.xlsx"

            with mock.patch.object(main_module, "ExamResultsProcessor", return_value=fake_processor) as processor_cls:
                main_module.process_results(
                    results_path,
                    marking_path,
                    output_dir=root / "results-output",
                    output_name="summary-name",
                )

        processor_cls.assert_called_once()
        fake_processor.load_results.assert_called_once()
        fake_processor.load_marking_sheet.assert_called_once()
        fake_processor.calculate_student_scores.assert_called_once_with("results-df", "marking-df")
        fake_processor.generate_question_statistics.assert_called_once_with(
            "results-df",
            "marking-df",
            "scores-df",
        )
        fake_processor.generate_summary_report.assert_called_once_with("scores-df", "stats-df", "summary-name")


if __name__ == "__main__":
    unittest.main()
