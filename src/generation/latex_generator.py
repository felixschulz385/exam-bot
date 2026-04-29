import datetime
import logging
import random
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.utils import compute_last_in_block
from .base import BaseExamGenerator

logger = logging.getLogger(__name__)


class LatexGenerator(BaseExamGenerator):
    """Generate exam PDFs by writing LaTeX and compiling with a TeX engine."""

    _DISPLAY_MATH_PATTERN = re.compile(r"(\\\[.*?\\\]|\$\$.*?\$\$)", re.DOTALL)
    _INLINE_PAREN_MATH_PATTERN = re.compile(r"(\\\(.*?\\\))", re.DOTALL)
    _UNICODE_REPLACEMENTS = {
        "\u00a0": " ",
        "\u2009": " ",
        "\u202f": " ",
        "\u2212": "-",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "--",
        "\u2014": "---",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": "``",
        "\u201d": "''",
        "\u2026": r"\ldots{}",
    }
    _INSTRUCTION_HEADING_PATTERN = re.compile(
        r"^(?:#+\s*)?(?:exam\s+)?instructions\s*:?\s*$",
        re.IGNORECASE,
    )

    def _collapse_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text)).strip()

    def _normalize_unicode(self, text: str) -> str:
        normalized = str(text)
        for source, target in self._UNICODE_REPLACEMENTS.items():
            normalized = normalized.replace(source, target)
        return normalized

    def _normalize_math_delimiters(self, text: str) -> str:
        text = self._normalize_unicode(text)
        text = re.sub(r"\$<(.*?)>\$", r"$\1$", text, flags=re.DOTALL)
        if text.startswith(r"\(") and text.endswith(r"\)"):
            return f"${text[2:-2]}$"
        if text.startswith(r"\[") and text.endswith(r"\]"):
            return f"$${text[2:-2]}$$"
        return text

    def _escape_latex_text(self, text: str) -> str:
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
        }
        escaped = str(text)
        for source, target in replacements.items():
            escaped = escaped.replace(source, target)
        escaped = escaped.replace("^", r"\textasciicircum{}")
        escaped = escaped.replace("~", r"\textasciitilde{}")
        return escaped

    def _looks_like_math_segment(self, content: str) -> bool:
        stripped = content.strip()
        if not stripped:
            return False

        if "\\" in stripped:
            return True

        if re.search(r"[=<>_^{}\[\]()+\-*/]", stripped):
            return True

        if re.fullmatch(r"[A-Za-z0-9.,]+", stripped):
            return True

        words = re.findall(r"[A-Za-z]+", stripped)
        return bool(words) and len(words) <= 2 and all(len(word) <= 2 for word in words)

    def _split_inline_dollar_math(self, text: str) -> List[Tuple[bool, str]]:
        segments: List[Tuple[bool, str]] = []
        cursor = 0

        while cursor < len(text):
            start = text.find("$", cursor)
            if start == -1:
                if cursor < len(text):
                    segments.append((False, text[cursor:]))
                break

            if start > cursor:
                segments.append((False, text[cursor:start]))

            if start + 1 < len(text) and text[start + 1] == "$":
                end = text.find("$$", start + 2)
                if end != -1:
                    segments.append((True, text[start:end + 2]))
                    cursor = end + 2
                    continue
                segments.append((False, "$$"))
                cursor = start + 2
                continue

            end = text.find("$", start + 1)
            if end == -1:
                segments.append((False, "$"))
                cursor = start + 1
                continue

            candidate = text[start + 1:end]
            if self._looks_like_math_segment(candidate):
                segments.append((True, f"${candidate}$"))
                cursor = end + 1
            else:
                segments.append((False, "$"))
                cursor = start + 1

        return segments

    def _render_text_with_math(self, text: str) -> str:
        rendered: List[str] = []
        cursor = 0

        for pattern in (self._DISPLAY_MATH_PATTERN, self._INLINE_PAREN_MATH_PATTERN):
            parts: List[Tuple[int, int, str]] = []
            for match in pattern.finditer(text):
                parts.append((match.start(), match.end(), match.group(0)))
            if parts:
                break
        else:
            parts = []

        if not parts:
            segments = self._split_inline_dollar_math(text)
            return "".join(
                self._normalize_math_delimiters(segment) if is_math else self._escape_latex_text(segment)
                for is_math, segment in segments
            )

        for start, end, segment in parts:
            if start > cursor:
                rendered.append(self._render_text_with_math(text[cursor:start]))
            rendered.append(self._normalize_math_delimiters(segment))
            cursor = end

        if cursor < len(text):
            rendered.append(self._render_text_with_math(text[cursor:]))

        return "".join(rendered)

    def _latexify_text(self, text: str) -> str:
        """Escape prose while preserving embedded LaTeX math."""
        collapsed = self._collapse_text(self._normalize_unicode(text))
        return self._render_text_with_math(collapsed)

    def _render_instruction_section(self, instructions: str) -> List[str]:
        lines = [line.strip() for line in str(instructions).splitlines()]
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()

        if lines and self._INSTRUCTION_HEADING_PATTERN.match(lines[0]):
            lines.pop(0)

        rendered_lines = [r"\section*{Instructions}"]
        for line in lines:
            if not line:
                rendered_lines.append("")
                continue
            if line.startswith("###"):
                rendered_lines.append(
                    r"\textbf{" + self._latexify_text(line.replace("###", "", 1).strip()) + r"}\par"
                )
            else:
                rendered_lines.append(self._latexify_text(line) + r"\par")

        rendered_lines.extend(
            [
                r"\bigskip",
                r"\noindent\rule{\textwidth}{0.6pt}",
                r"\bigskip",
            ]
        )
        return rendered_lines

    def _render_answer_block(
        self,
        row: pd.Series,
        question_num: int,
        mark_correct: bool,
        randomize_answers: bool,
    ) -> Tuple[List[str], Dict[int, str]]:
        answer_choices = self._build_answer_choices(row, self._latexify_text)
        should_randomize = randomize_answers and self._should_randomize_answers(
            str(row.get("Type", "")),
            answer_choices,
        )

        answer_mapping: Dict[int, str] = {orig_idx: letter for orig_idx, letter, _, _ in answer_choices}
        if should_randomize:
            random_seed = question_num + (self.seed or 0)
            question_rng = random.Random(random_seed)
            answer_choices = answer_choices[:]
            question_rng.shuffle(answer_choices)
            answer_mapping = {
                orig_idx: chr(65 + position)
                for position, (orig_idx, _, _, _) in enumerate(answer_choices)
            }

        lines = [r"\begin{enumerate}[label=\Alph*),leftmargin=1.5cm]"]
        for position, (orig_idx, original_letter, answer_text, is_correct) in enumerate(answer_choices):
            display_letter = chr(65 + position) if should_randomize else original_letter
            rendered_text = answer_text
            if mark_correct and is_correct:
                rendered_text = r"\textcolor{answerblue}{\textbf{" + answer_text + "}}"
            lines.append(r"\item[" + display_letter + ")] " + rendered_text)
        lines.append(r"\end{enumerate}")

        return lines, answer_mapping

    def _render_question(
        self,
        row: pd.Series,
        question_num: int,
        points: int,
        include_answers: bool,
        mark_correct: bool,
        randomize_answers: bool,
    ) -> Tuple[List[str], Dict[int, str]]:
        question_text = self._latexify_text(row["Question"])
        displayed_number = f"{question_num - 1:02d}"

        lines = [
            r"\Needspace{10\baselineskip}",
            r"\begin{samepage}",
            r"\noindent\textbf{" + displayed_number + r".} "
            + question_text
            + r" \textit{("
            + str(points)
            + (" points" if points != 1 else " point")
            + r")}\par"
        ]

        answer_mapping: Dict[int, str] = {}
        if include_answers:
            answer_lines, answer_mapping = self._render_answer_block(
                row,
                question_num=question_num,
                mark_correct=mark_correct,
                randomize_answers=randomize_answers,
            )
            lines.extend(answer_lines)

        lines.append(r"\end{samepage}")
        lines.append(r"\vspace{0.8\baselineskip}")
        return lines, answer_mapping

    def _render_block_leading_content(
        self,
        row: pd.Series,
        figure_num: int,
        block_number: int,
    ) -> List[str]:
        lines: List[str] = [
            r"\vspace{1.25\baselineskip}",
            r"\noindent\rule{\textwidth}{0.8pt}",
            r"\par\smallskip",
            r"\noindent\textbf{\large Question Block " + str(block_number) + r"}\par",
            r"\smallskip",
        ]

        if "Leading Text" in row and pd.notna(row["Leading Text"]):
            lines.append(
                r"\begin{quote}\itshape "
                + self._latexify_text(str(row["Leading Text"]))
                + r"\end{quote}"
            )

        if "Medium" in row and pd.notna(row["Medium"]):
            try:
                match = re.search(r"(?:figure|fig\.?)\s+(\w+)", str(row["Medium"]), re.IGNORECASE)
                if match:
                    original_figure_num = match.group(1)
                    image_dir = self.doc_config.get("block_questions", {}).get("image_directory", "data/Figures")
                    image_path = Path(image_dir) / f"{original_figure_num}.png"
                    if image_path.exists():
                        lines.extend(
                            [
                                r"\begin{center}",
                                r"\includegraphics[width=0.75\textwidth]{\detokenize{" + str(image_path.resolve()) + r"}}",
                                r"\end{center}",
                            ]
                        )
            except Exception as exc:
                logger.warning("Failed to include block image: %s", exc)

        lines.append(r"\medskip")
        return lines

    def _build_tex_document(
        self,
        questions: pd.DataFrame,
        config: Dict[str, Any],
        include_answers: bool,
        mark_correct: bool,
    ) -> Tuple[str, Dict[int, Dict[int, str]]]:
        self.doc_config = config.get("document", {})
        randomize_answers = self.doc_config.get("randomize_answers", False)
        self.seed = config.get("selection", {}).get("seed")
        answer_mappings: Dict[int, Dict[int, str]] = {}

        grading_config = config.get("grading", {})
        type_points = grading_config.get("points_per_type", {})

        question_bank_reader = config.get("_runtime", {}).get("question_bank_reader")
        has_block_info = question_bank_reader is not None
        last_in_block = {}
        if has_block_info:
            last_in_block = compute_last_in_block(
                question_bank_reader.get_block_questions(),
                available_indices=list(questions.index),
            )

        body_lines: List[str] = [
            r"\documentclass[11pt]{article}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage{amsmath,amssymb}",
            r"\usepackage{graphicx}",
            r"\usepackage{needspace}",
            r"\usepackage{xcolor}",
            r"\definecolor{answerblue}{RGB}{0,114,178}",
            r"\usepackage{enumitem}",
            r"\usepackage{parskip}",
            r"\usepackage{fancyhdr}",
            r"\pagestyle{fancy}",
            r"\fancyhf{}",
            r"\cfoot{\thepage}",
            r"\begin{document}",
        ]

        header_text = config.get("exam", {}).get("header_text", "")
        if header_text:
            body_lines.append(r"\begin{center}\Large\textbf{" + self._latexify_text(header_text) + r"}\end{center}")

        instructions = config.get("exam", {}).get("instructions", "")
        if instructions:
            body_lines.extend(self._render_instruction_section(instructions))

        total_points = 0
        current_block = None
        question_num = 1
        figure_num = 1
        block_number = 0

        for idx, row in questions.iterrows():
            is_block_question = False
            is_first_in_block = False
            is_last_in_block = last_in_block.get(idx, False)

            if has_block_info:
                try:
                    is_block_question = question_bank_reader.is_block_question(idx)
                    if is_block_question:
                        block_id = question_bank_reader.get_block_for_question(idx)
                        is_first_in_block = question_bank_reader.is_first_in_block(idx)
                        if block_id != current_block and is_first_in_block:
                            if question_num > 1:
                                body_lines.append(r"\newpage")
                            current_block = block_id
                            block_number += 1
                            body_lines.extend(self._render_block_leading_content(row, figure_num, block_number))
                except Exception as exc:
                    logger.warning("Block question detection failed in LaTeX generator: %s", exc)

            points = type_points.get(row["Type"], 1)
            total_points += points

            question_lines, answer_mapping = self._render_question(
                row,
                question_num=question_num,
                points=points,
                include_answers=include_answers,
                mark_correct=mark_correct,
                randomize_answers=randomize_answers,
            )
            body_lines.extend(question_lines)
            if randomize_answers:
                answer_mappings[question_num] = answer_mapping

            if is_block_question and is_last_in_block:
                body_lines.extend(
                    [
                        r"\smallskip",
                        r"\noindent\rule{\textwidth}{0.6pt}",
                        r"\vspace{1.2\baselineskip}",
                    ]
                )
            elif not is_block_question:
                body_lines.append(r"\vspace{0.9\baselineskip}")

            if "Medium" in row and pd.notna(row["Medium"]):
                if re.search(r"(?:figure|fig\.?)\s+(\w+)", str(row["Medium"]), re.IGNORECASE):
                    figure_num += 1

            question_num += 1

        body_lines.extend(
            [
                r"\bigskip\hrule\bigskip",
                r"\begin{flushright}\textbf{Total possible points: " + str(total_points) + r"}\end{flushright}",
                r"\end{document}",
            ]
        )

        return "\n".join(body_lines), answer_mappings

    def _compile_pdf(self, tex_path: Path, pdf_path: Path, pdf_engine: str) -> None:
        """Compile a LaTeX source file to PDF using a direct TeX engine."""
        engine = (pdf_engine or "pdflatex").strip()
        command = [
            engine,
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={tex_path.parent}",
            str(tex_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, cwd=tex_path.parent)
            subprocess.run(command, check=True, capture_output=True, text=True, cwd=tex_path.parent)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"The configured LaTeX engine '{engine}' is required to generate PDF exams, but it was not found on PATH."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "The LaTeX engine failed to compile the exam to PDF.\n"
                f"Command: {' '.join(command)}\n"
                f"stdout:\n{exc.stdout}\n"
                f"stderr:\n{exc.stderr}"
            ) from exc

        if not pdf_path.exists():
            raise RuntimeError(
                f"LaTeX compilation finished without producing the expected PDF: {pdf_path}"
            )

    def _generate_version(
        self,
        questions: pd.DataFrame,
        config: Dict[str, Any],
        include_answers: bool,
        mark_correct: bool,
        version_suffix: str,
    ) -> Tuple[Path, Dict[int, Dict[int, str]]]:
        tex_source, answer_mappings = self._build_tex_document(
            questions,
            config,
            include_answers=include_answers,
            mark_correct=mark_correct,
        )

        exam_title = config.get("exam", {}).get("title", "Exam").replace(" ", "_")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{exam_title}_{version_suffix}_{timestamp}"
        tex_path = self.output_dir / f"{base_name}.tex"
        pdf_path = self.output_dir / f"{base_name}.pdf"
        tex_path.write_text(tex_source, encoding="utf-8")

        pdf_engine = config.get("document", {}).get("pdf_engine", "pdflatex")
        self._compile_pdf(tex_path, pdf_path, pdf_engine)
        logger.info("Generated LaTeX exam source: %s", tex_path)
        logger.info("Generated PDF exam: %s", pdf_path)

        return pdf_path, answer_mappings

    def generate_exam(
        self,
        questions: pd.DataFrame,
        config: Dict[str, Any],
    ) -> Tuple[Optional[Path], Optional[Path], Dict]:
        student_path = None
        instructor_path = None
        answer_mappings: Dict[int, Dict[int, str]] = {}

        if config.get("document", {}).get("student_version", True):
            student_path, answer_mappings = self._generate_version(
                questions,
                config,
                include_answers=True,
                mark_correct=False,
                version_suffix="student",
            )

        if config.get("document", {}).get("instructor_version", True):
            instructor_path, _ = self._generate_version(
                questions,
                config,
                include_answers=True,
                mark_correct=True,
                version_suffix="instructor",
            )

        return student_path, instructor_path, answer_mappings
