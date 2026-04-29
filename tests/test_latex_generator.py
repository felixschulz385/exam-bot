import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src.generation import get_generator_class
from src.generation.base import BaseExamGenerator
from src.generation.latex_generator import LatexGenerator
from src.generation.word_generator import WordGenerator


class LatexGeneratorTests(unittest.TestCase):
    def test_backend_registry_resolves_latex_and_word_generators(self):
        self.assertIs(get_generator_class("latex"), LatexGenerator)
        self.assertIs(get_generator_class("word"), WordGenerator)
        self.assertTrue(issubclass(LatexGenerator, BaseExamGenerator))
        self.assertTrue(issubclass(WordGenerator, BaseExamGenerator))

    def test_latexify_text_normalizes_unicode_math_punctuation(self):
        generator = LatexGenerator(output_dir=Path("/tmp"))
        rendered = generator._latexify_text("Profit $\\pi=(f−c)m$ with physician’s choice")

        self.assertIn("$\\pi=(f-c)m$", rendered)
        self.assertIn("physician's choice", rendered)

    def test_latexify_text_escapes_literal_dollar_outside_math(self):
        generator = LatexGenerator(output_dir=Path("/tmp"))
        rendered = generator._latexify_text("Penalty is $5 and payoff is $x$ in equilibrium")

        self.assertIn(r"Penalty is \$5", rendered)
        self.assertIn("$x$", rendered)

    def make_question(self, qtype, question, correct, answers):
        row = {
            "Topic": "Topic A",
            "Type": qtype,
            "Question": question,
            "Correct": correct,
            "Leading Text": "",
            "Medium": "",
        }
        for index, answer in enumerate(answers, 1):
            row[f"Answer {index}"] = answer
        return row

    def test_generate_exam_writes_tex_and_invokes_pdflatex(self):
        questions = pd.DataFrame(
            [
                self.make_question("True/False", r"Is $\gamma > 0$?", "B", ["True", "False"]),
                self.make_question("Single Choice", r"What is $\frac{1}{2}$?", 2, ["One", "Half", "Three", "Four"]),
            ]
        )
        config = {
            "exam": {
                "title": "Sample Exam",
                "instructions": "### Instructions\nShow your work.",
            },
            "document": {
                "student_version": True,
                "instructor_version": True,
                "randomize_answers": True,
                "pdf_engine": "xelatex",
            },
            "selection": {"seed": 7},
            "grading": {"points_per_type": {"True/False": 2, "Single Choice": 2}},
            "_runtime": {},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            generator = LatexGenerator(output_dir=Path(temp_dir))
            def fake_run(command, check, capture_output, text, cwd):
                pdf_name = Path(command[-1]).with_suffix(".pdf").name
                Path(cwd, pdf_name).write_bytes(b"%PDF-1.4")
                return mock.Mock()

            with mock.patch("src.generation.latex_generator.subprocess.run", side_effect=fake_run) as run_mock:
                student_path, instructor_path, answer_mappings = generator.generate_exam(questions, config)

            tex_files = sorted(Path(temp_dir).glob("*.tex"))
            self.assertEqual(len(tex_files), 2)
            instructor_tex = [path for path in tex_files if "instructor" in path.name][0].read_text(encoding="utf-8")

        self.assertEqual(student_path.suffix, ".pdf")
        self.assertEqual(instructor_path.suffix, ".pdf")
        self.assertEqual(run_mock.call_count, 4)
        first_command = run_mock.call_args_list[0].args[0]
        self.assertEqual(first_command[0], "xelatex")
        self.assertIn("-interaction=nonstopmode", first_command)
        self.assertEqual(answer_mappings[1], {1: "A", 2: "B"})
        self.assertIn(r"\textcolor{answerblue}{\textbf{False}}", instructor_tex)
        self.assertIn(r"Is $\gamma > 0$?", instructor_tex)
        self.assertIn(r"What is $\frac{1}{2}$?", instructor_tex)
        self.assertIn(r"\section*{Instructions}", instructor_tex)
        self.assertNotIn(r"\textbf{Instructions}\par", instructor_tex)
        self.assertIn(r"\noindent\rule{\textwidth}{0.6pt}", instructor_tex)

    def test_build_tex_document_marks_blocks_and_adds_spacing(self):
        questions = pd.DataFrame(
            [
                self.make_question("Single Choice", "Standalone question before block", 1, ["S1", "S2", "S3", "S4"]),
                self.make_question("Single Choice Block", "Block question one", 1, ["A1", "A2", "A3", "A4"]),
                self.make_question("Single Choice Block", "Block question two", 2, ["B1", "B2", "B3", "B4"]),
                self.make_question("Single Choice", "Standalone question after block", 1, ["C1", "C2", "C3", "C4"]),
            ]
        )
        questions.loc[1, "Leading Text"] = "Shared case stem"

        fake_reader = mock.Mock()
        fake_reader.get_block_questions.return_value = {"block-1": [1, 2]}
        fake_reader.is_block_question.side_effect = lambda idx: idx in {1, 2}
        fake_reader.get_block_for_question.side_effect = lambda idx: "block-1" if idx in {1, 2} else None
        fake_reader.is_first_in_block.side_effect = lambda idx: idx == 1

        config = {
            "exam": {"title": "Sample Exam"},
            "document": {"randomize_answers": False},
            "selection": {"seed": 7},
            "grading": {"points_per_type": {"Single Choice Block": 3, "Single Choice": 2}},
            "_runtime": {"question_bank_reader": fake_reader},
        }

        generator = LatexGenerator(output_dir=Path("/tmp"))
        tex_source, _ = generator._build_tex_document(
            questions,
            config,
            include_answers=True,
            mark_correct=False,
        )

        self.assertIn(r"\textbf{\large Question Block 1}", tex_source)
        self.assertIn(r"\newpage" + "\n" + r"\vspace{1.25\baselineskip}", tex_source)
        self.assertIn(r"\rule{\textwidth}{0.8pt}", tex_source)
        self.assertIn(r"\rule{\textwidth}{0.6pt}", tex_source)
        self.assertIn(r"\vspace{0.9\baselineskip}", tex_source)
        self.assertIn(r"\usepackage{needspace}", tex_source)
        self.assertIn(r"\definecolor{answerblue}{RGB}{0,114,178}", tex_source)
        self.assertIn(r"\Needspace{10\baselineskip}", tex_source)
        self.assertIn(r"\begin{samepage}", tex_source)
        self.assertIn(r"\end{samepage}", tex_source)
        self.assertNotIn(r"\color{black!60}", tex_source)

    def test_build_tex_document_splits_instructions_from_question_body(self):
        questions = pd.DataFrame(
            [
                self.make_question("Single Choice", "Standalone question", 1, ["A1", "A2", "A3", "A4"]),
            ]
        )
        config = {
            "exam": {
                "title": "Sample Exam",
                "instructions": "### Instructions\nShow your work.",
            },
            "document": {"randomize_answers": False},
            "selection": {"seed": 7},
            "grading": {"points_per_type": {"Single Choice": 2}},
            "_runtime": {},
        }

        generator = LatexGenerator(output_dir=Path("/tmp"))
        tex_source, _ = generator._build_tex_document(
            questions,
            config,
            include_answers=True,
            mark_correct=False,
        )

        self.assertIn(r"\section*{Instructions}", tex_source)
        self.assertNotIn(r"\section*{Exam Instructions}", tex_source)
        self.assertNotIn(r"\textbf{Instructions}\par", tex_source)
        self.assertIn(
            "Show your work.\\par\n\\bigskip\n\\noindent\\rule{\\textwidth}{0.6pt}\n\\bigskip\n\\Needspace",
            tex_source,
        )


if __name__ == "__main__":
    unittest.main()
