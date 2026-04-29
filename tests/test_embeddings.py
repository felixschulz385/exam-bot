import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.data.embeddings import QuestionEmbedder


class FakeModel:
    def get_sentence_embedding_dimension(self):
        return 2

    def encode(self, texts, convert_to_numpy=True):
        return np.array([[float(index + 1), float(index + 2)] for index, _ in enumerate(texts)])


class EmbeddingsTests(unittest.TestCase):
    def test_embed_questions_uses_cached_vectors_without_loading_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            bank_path = Path(temp_dir) / "questions.xlsx"
            bank_path.write_text("x", encoding="utf-8")

            embedder = QuestionEmbedder(cache_dir=cache_dir)
            embedder.embedding_cache = {
                "Q1": np.array([1.0, 2.0]),
                "Q2": np.array([3.0, 4.0]),
            }
            embedder._embedding_dimension = 2
            embedder._cached_bank_fingerprint = embedder._get_question_bank_fingerprint(bank_path)

            with mock.patch.object(embedder, "_load_cache", return_value=True), \
                 mock.patch.object(embedder, "_load_model") as load_model_mock:
                result = embedder.embed_questions(["Q1", "Q2"], bank_path)

        load_model_mock.assert_not_called()
        np.testing.assert_array_equal(result, np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_embed_questions_uses_cache_after_first_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            bank_path = Path(temp_dir) / "questions.xlsx"
            bank_path.write_text("x", encoding="utf-8")

            embedder = QuestionEmbedder(cache_dir=cache_dir)
            embedder.model = FakeModel()

            with mock.patch.object(embedder, "_load_model"):
                first = embedder.embed_questions(["Q1", "Q2"], bank_path)
                second = embedder.embed_questions(["Q1", "Q2"], bank_path)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(embedder.embedding_cache), 2)

    def test_legacy_cache_payload_is_supported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "legacy.pkl"
            embedder = QuestionEmbedder(cache_dir=Path(temp_dir))
            payload = {"Question": np.array([1.0, 2.0])}

            import pickle
            with open(cache_path, "wb") as handle:
                pickle.dump(payload, handle)

            loaded = embedder._load_cache(cache_path)

        self.assertTrue(loaded)
        self.assertIn("Question", embedder.embedding_cache)


if __name__ == "__main__":
    unittest.main()
