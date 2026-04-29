import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import main as main_module
from src.config import ConfigLoader, DEFAULT_CONFIG_PATH
from src.selection import UnifiedSampler


class ConfigAndSelectionTests(unittest.TestCase):
    def make_question(self, topic, qtype, question, correct=1, medium="", identifier=None, leading_text=None):
        row = {
            "Topic": topic,
            "Type": qtype,
            "Question": question,
            "Answer 1": "A",
            "Answer 2": "B",
            "Answer 3": "C",
            "Answer 4": "D",
            "Correct": correct,
            "Medium": medium,
        }
        if identifier is not None:
            row["ID"] = identifier
        if leading_text is not None:
            row["Leading Text"] = leading_text
        return row

    def test_loader_uses_central_yaml_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "override.yaml"
            config_path.write_text("exam:\n  title: Custom Exam\n", encoding="utf-8")

            loaded = ConfigLoader().load_config(config_path)

        self.assertEqual(loaded["exam"]["title"], "Custom Exam")
        self.assertEqual(loaded["io"]["question_bank_sheet"], "v2")
        self.assertIn("topics", loaded["selection"])
        self.assertIn("type_ratios", loaded["selection"])
        self.assertIn("points_per_type", loaded["grading"])
        self.assertEqual(DEFAULT_CONFIG_PATH.name, "exam_settings.yaml")

    def test_prepare_generation_config_resolves_nested_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            assets_dir = root / "assets"
            config_dir.mkdir()
            assets_dir.mkdir()

            question_bank = assets_dir / "questions.xlsx"
            figures = assets_dir / "figures"
            figures.mkdir()
            question_bank.touch()

            config_path = config_dir / "exam.yaml"
            config_path.write_text("exam: {}\n", encoding="utf-8")

            prepared = main_module.prepare_generation_config(
                {
                    "io": {
                        "question_bank_path": "../assets/questions.xlsx",
                        "output_dir": "../output",
                    },
                    "selection": {
                        "method": "unified",
                        "semantic": {"cache_dir": "../cache"},
                    },
                    "document": {
                        "block_questions": {"image_directory": "../assets/figures"}
                    },
                    "exam": {},
                },
                config_path,
            )

        self.assertTrue(prepared["io"]["question_bank_path"].endswith("assets/questions.xlsx"))
        self.assertTrue(prepared["selection"]["semantic"]["cache_dir"].endswith("cache"))
        self.assertTrue(prepared["document"]["block_questions"]["image_directory"].endswith("assets/figures"))

    def test_validate_grading_scheme_accepts_ratio_based_config(self):
        loader = ConfigLoader()
        self.assertTrue(
            loader.validate_grading_scheme(
                topics=["A", "B"],
                type_ratios={"Single Choice": 1},
                points_per_type={"Single Choice": 2},
            )
        )

    def test_select_questions_does_not_filter_checked_flag(self):
        questions = pd.DataFrame(
            [
                {
                    "Topic": "A",
                    "Type": "Single Choice",
                    "Question": "Q1",
                    "Answer 1": "A",
                    "Answer 2": "B",
                    "Answer 3": "C",
                    "Answer 4": "D",
                    "Correct": 1,
                    "Medium": "",
                    "Check: Felix": False,
                },
                {
                    "Topic": "A",
                    "Type": "Single Choice",
                    "Question": "Q2",
                    "Answer 1": "A",
                    "Answer 2": "B",
                    "Answer 3": "C",
                    "Answer 4": "D",
                    "Correct": 2,
                    "Medium": "",
                    "Check: Felix": True,
                },
            ]
        )

        class FakeBank:
            def get_all_questions(self):
                return questions

        captured = {}

        class FakeSampler:
            def __init__(self, *args, **kwargs):
                captured["init_kwargs"] = kwargs

            def select_unified(self, questions_df, *args, **kwargs):
                captured["row_count"] = len(questions_df)
                captured["select_kwargs"] = kwargs
                return list(questions_df.index)

        config = {
            "selection": {
                "method": "unified",
                "semantic": {},
                "target_points": 4,
                "topics": ["A"],
                "type_ratios": {"Single Choice": 1},
            },
            "grading": {"points_per_type": {"Single Choice": 2}},
            "io": {"question_bank_path": "questions.xlsx"},
        }

        with mock.patch.object(main_module, "UnifiedSampler", FakeSampler):
            selected = main_module.select_questions(config, FakeBank())

        self.assertEqual(captured["row_count"], 2)
        self.assertEqual(selected, [0, 1])
        self.assertEqual(captured["init_kwargs"]["model_name"], "paraphrase-multilingual-MiniLM-L12-v2")
        self.assertEqual(captured["select_kwargs"]["cluster_ratio"], 0.3)
        self.assertEqual(captured["select_kwargs"]["topics"], ["A"])

    def test_unified_sampler_selects_exact_ratio_distribution(self):
        questions = pd.DataFrame([
            self.make_question("Topic A", "Single Choice", "A1"),
            self.make_question("Topic B", "True/False", "B1"),
        ])

        sampler = UnifiedSampler(seed=1)
        selected = sampler.select_unified(
            questions,
            topic_ratios={"Topic A": 1, "Topic B": 1},
            type_ratios={"Single Choice": 1, "True/False": 1},
            points_per_type={"Single Choice": 2, "True/False": 2},
            target_points=4,
        )

        self.assertEqual(selected, [0, 1])

    def test_unified_sampler_selects_without_topic_ratios(self):
        questions = pd.DataFrame([
            self.make_question("Topic A", "Single Choice", "A1"),
            self.make_question("Topic B", "True/False", "B1"),
        ])

        sampler = UnifiedSampler(seed=1)
        selected = sampler.select_unified(
            questions,
            topic_ratios={},
            type_ratios={"Single Choice": 1, "True/False": 1},
            points_per_type={"Single Choice": 2, "True/False": 2},
            target_points=4,
            topics=["Topic A", "Topic B"],
        )

        self.assertEqual(selected, [0, 1])

    def test_unified_sampler_raises_when_exact_distribution_is_impossible(self):
        questions = pd.DataFrame([
            self.make_question("Topic A", "Single Choice", "A1"),
            self.make_question("Topic B", "Single Choice", "B1"),
        ])

        sampler = UnifiedSampler(seed=1)
        with self.assertRaises(ValueError) as ctx:
            sampler.select_unified(
                questions,
                topic_ratios={"Topic A": 1, "Topic B": 1},
                type_ratios={"Single Choice": 1, "True/False": 1},
                points_per_type={"Single Choice": 2, "True/False": 2},
                target_points=4,
            )

        self.assertIn("type distribution cannot be achieved exactly", str(ctx.exception).lower())
        self.assertIn("true/false", str(ctx.exception).lower())

    def test_unified_sampler_prefers_more_diverse_feasible_solution(self):
        questions = pd.DataFrame([
            self.make_question("Topic A", "Single Choice", "A-close"),
            self.make_question("Topic A", "Single Choice", "A-far"),
            self.make_question("Topic B", "Single Choice", "B-anchor"),
        ])

        sampler = UnifiedSampler(seed=1)
        sampler.embedder = mock.Mock()
        sampler.embedder.compute_pairwise_distances.return_value = np.array(
            [
                [0.0, 0.9, 0.1],
                [0.9, 0.0, 0.8],
                [0.1, 0.8, 0.0],
            ]
        )

        selected = sampler.select_unified(
            questions,
            topic_ratios={"Topic A": 1, "Topic B": 1},
            type_ratios={"Single Choice": 1},
            points_per_type={"Single Choice": 2},
            target_points=4,
            cluster_ratio=0.5,
        )

        self.assertEqual(selected, [1, 2])

    def test_unified_sampler_seed_changes_choice_with_multiple_top_candidates(self):
        questions = pd.DataFrame([
            self.make_question("Topic A", "Single Choice", "A1"),
            self.make_question("Topic A", "Single Choice", "A2"),
            self.make_question("Topic B", "Single Choice", "B1"),
            self.make_question("Topic B", "Single Choice", "B2"),
        ])

        distance_matrix = np.array(
            [
                [0.0, 0.2, 0.7, 0.7],
                [0.2, 0.0, 0.7, 0.7],
                [0.7, 0.7, 0.0, 0.2],
                [0.7, 0.7, 0.2, 0.0],
            ]
        )

        sampler_one = UnifiedSampler(seed=1)
        sampler_one.embedder = mock.Mock()
        sampler_one.embedder.compute_pairwise_distances.return_value = distance_matrix
        selected_one = sampler_one.select_unified(
            questions,
            topic_ratios={"Topic A": 1, "Topic B": 1},
            type_ratios={"Single Choice": 1},
            points_per_type={"Single Choice": 2},
            target_points=4,
            cluster_ratio=1.0,
        )

        sampler_two = UnifiedSampler(seed=5)
        sampler_two.embedder = mock.Mock()
        sampler_two.embedder.compute_pairwise_distances.return_value = distance_matrix
        selected_two = sampler_two.select_unified(
            questions,
            topic_ratios={"Topic A": 1, "Topic B": 1},
            type_ratios={"Single Choice": 1},
            points_per_type={"Single Choice": 2},
            target_points=4,
            cluster_ratio=1.0,
        )

        self.assertNotEqual(selected_one, selected_two)
        self.assertEqual(len(selected_one), 2)
        self.assertEqual(len(selected_two), 2)

    def test_unified_sampler_respects_block_atomicity_in_semantic_selection(self):
        questions = pd.DataFrame([
            self.make_question("Topic A", "Single Choice Block", "Block part 1", identifier="10-1", leading_text="Shared prompt"),
            self.make_question("Topic A", "Single Choice Block", "Block part 2", identifier="10-2"),
            self.make_question("Topic B", "Single Choice", "Standalone B"),
        ])

        sampler = UnifiedSampler(seed=1)
        sampler.embedder = mock.Mock()
        sampler.embedder.compute_pairwise_distances.return_value = np.array([[0.0, 0.6], [0.6, 0.0]])

        selected = sampler.select_unified(
            questions,
            topic_ratios={"Topic A": 3, "Topic B": 1},
            type_ratios={"Single Choice Block": 3, "Single Choice": 1},
            points_per_type={"Single Choice Block": 3, "Single Choice": 2},
            target_points=8,
            cluster_ratio=1.0,
        )

        self.assertEqual(selected, [0, 1, 2])

    def test_unified_sampler_calls_embedder_with_question_bank_path(self):
        questions = pd.DataFrame([
            self.make_question("Topic A", "Single Choice", "A1"),
            self.make_question("Topic A", "Single Choice", "A2"),
            self.make_question("Topic B", "Single Choice", "B1"),
        ])

        sampler = UnifiedSampler(seed=1)
        sampler.embedder = mock.Mock()
        sampler.embedder.compute_pairwise_distances.return_value = np.array(
            [
                [0.0, 0.3, 0.7],
                [0.3, 0.0, 0.8],
                [0.7, 0.8, 0.0],
            ]
        )

        sampler.select_unified(
            questions,
            topic_ratios={"Topic A": 1, "Topic B": 1},
            type_ratios={"Single Choice": 1},
            points_per_type={"Single Choice": 2},
            target_points=4,
            cluster_ratio=1.0,
            question_bank_path=Path("/tmp/question-bank.xlsx"),
        )

        sampler.embedder.compute_pairwise_distances.assert_called_once()
        _, kwargs = sampler.embedder.compute_pairwise_distances.call_args
        self.assertEqual(kwargs["question_bank_path"], Path("/tmp/question-bank.xlsx"))

    def test_unified_sampler_reports_non_integral_ratio_allocations_clearly(self):
        questions = pd.DataFrame([
            self.make_question("Topic A", "Single Choice", "A1"),
        ])

        sampler = UnifiedSampler(seed=1)
        with self.assertRaises(ValueError) as ctx:
            sampler.select_unified(
                questions,
                topic_ratios={"Topic A": 1, "Topic B": 1, "Topic C": 1},
                type_ratios={"Single Choice": 1},
                points_per_type={"Single Choice": 2},
                target_points=10,
            )

        message = str(ctx.exception)
        self.assertIn("selection.topic_ratios", message)
        self.assertIn("sum is 3", message)
        self.assertIn("10/3", message)

    def test_unified_sampler_reports_unknown_question_type_clearly(self):
        questions = pd.DataFrame([
            self.make_question("Topic A", "Essay", "Explain policy"),
        ])

        sampler = UnifiedSampler(seed=1)
        with self.assertRaises(ValueError) as ctx:
            sampler.select_unified(
                questions,
                topic_ratios={"Topic A": 1},
                type_ratios={"Essay": 1},
                points_per_type={"Single Choice": 2},
                target_points=2,
            )

        message = str(ctx.exception)
        self.assertIn("grading.points_per_type", message)
        self.assertIn("Essay", message)

    def test_unified_sampler_reports_unreachable_topic_totals_clearly(self):
        questions = pd.DataFrame([
            self.make_question("Topic A", "Single Choice", "A1"),
            self.make_question("Topic A", "Single Choice", "A2"),
            self.make_question("Topic B", "Single Choice", "B1"),
            self.make_question("Topic B", "Single Choice", "B2"),
        ])

        sampler = UnifiedSampler(seed=1)
        with self.assertRaises(ValueError) as ctx:
            sampler.select_unified(
                questions,
                topic_ratios={"Topic A": 1, "Topic B": 1},
                type_ratios={"Single Choice": 1},
                points_per_type={"Single Choice": 2},
                target_points=6,
            )

        message = str(ctx.exception).lower()
        self.assertIn("topic distribution cannot be achieved exactly", message)
        self.assertIn("topic a", message)
        self.assertIn("need 3 points", message)


if __name__ == "__main__":
    unittest.main()
