import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.utils import (
    build_block_index,
    compute_last_in_block,
    ensure_directory,
    get_block_for_question,
    is_first_in_block,
    parse_block_id,
    resolve_path,
)


class UtilsTests(unittest.TestCase):
    def test_parse_block_id(self):
        self.assertEqual(parse_block_id("22-3"), (22, 3))
        self.assertIsNone(parse_block_id("abc"))
        self.assertIsNone(parse_block_id("22-x"))

    def test_block_index_helpers(self):
        questions = pd.DataFrame(
            [
                {"ID": "10-2", "Topic": "A", "Type": "Single Choice"},
                {"ID": "10-1", "Topic": "A", "Type": "Single Choice"},
                {"ID": "11", "Topic": "B", "Type": "True/False"},
            ]
        )

        block_groups, flags = build_block_index(questions)

        self.assertEqual(block_groups[10], [1, 0])
        self.assertTrue(flags[0])
        self.assertFalse(flags[2])
        self.assertEqual(get_block_for_question(block_groups, 1), 10)
        self.assertTrue(is_first_in_block(block_groups, 1))
        self.assertEqual(compute_last_in_block(block_groups, [0, 1]), {0: True})

    def test_block_index_uses_id_for_grouping_and_bid_for_order(self):
        questions = pd.DataFrame(
            [
                {"ID": 5, "BID": 2, "Topic": "A", "Type": "Single Choice Block"},
                {"ID": 5, "BID": 1, "Topic": "A", "Type": "Single Choice Block"},
                {"ID": 6, "BID": 1, "Topic": "B", "Type": "Single Choice Block"},
                {"ID": 1, "BID": pd.NA, "Topic": "C", "Type": "Single Choice"},
            ]
        )

        block_groups, flags = build_block_index(questions)

        self.assertEqual(block_groups[5], [1, 0])
        self.assertEqual(block_groups[6], [2])
        self.assertTrue(flags[0])
        self.assertTrue(flags[2])
        self.assertFalse(flags[3])

    def test_runtime_helpers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "a" / "b"
            ensured = ensure_directory(nested)
            resolved = resolve_path("../file.txt", root / "a")

            self.assertEqual(ensured, nested)
            self.assertTrue(nested.exists())
            self.assertTrue(str(resolved).endswith("file.txt"))


if __name__ == "__main__":
    unittest.main()
