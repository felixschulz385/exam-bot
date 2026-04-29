from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def parse_block_id(question_id: object) -> Optional[Tuple[int, int]]:
    """Parse a block-style question identifier like '22-3'."""
    question_id = str(question_id or "")
    if "-" not in question_id:
        return None

    base_id, question_num = question_id.split("-", 1)
    try:
        return int(base_id), int(question_num)
    except ValueError:
        return None


def _normalize_numeric_identifier(value: object) -> Optional[int]:
    """Convert workbook identifiers like 3, 3.0, or '3' to an integer."""
    if pd.isna(value):
        return None

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        text_value = str(value).strip()
        if not text_value:
            return None
        try:
            return int(text_value)
        except ValueError:
            return None

    if numeric_value.is_integer():
        return int(numeric_value)
    return None


def build_block_index(questions: pd.DataFrame) -> Tuple[Dict[int, List[int]], Dict[int, bool]]:
    """Build block membership metadata from a question-bank dataframe."""
    block_groups: Dict[int, List[int]] = {}
    block_question_flags: Dict[int, bool] = {}
    order_keys: Dict[int, int] = {}

    for idx, row in questions.iterrows():
        item_order = _normalize_numeric_identifier(row.get("BID"))
        block_id = None

        if item_order is not None:
            block_id = _normalize_numeric_identifier(row.get("ID"))

        if block_id is None:
            parsed = parse_block_id(row.get("ID", ""))
            if parsed is None:
                block_question_flags[idx] = False
                continue
            block_id, item_order = parsed

        block_groups.setdefault(block_id, []).append(idx)
        block_question_flags[idx] = True
        if item_order is not None:
            order_keys[idx] = item_order

    for block_id, indices in block_groups.items():
        block_groups[block_id] = sorted(
            indices,
            key=lambda item_idx: (order_keys.get(item_idx, 10**9), item_idx),
        )

    return block_groups, block_question_flags


def get_block_for_question(block_groups: Dict[int, List[int]], idx: int) -> Optional[int]:
    """Return the block id for a question index if the question belongs to a block."""
    for block_id, indices in block_groups.items():
        if idx in indices:
            return block_id
    return None


def is_first_in_block(block_groups: Dict[int, List[int]], idx: int) -> bool:
    """Return whether a question index is the first member of its block."""
    block_id = get_block_for_question(block_groups, idx)
    if block_id is None:
        return False
    return block_groups[block_id][0] == idx


def compute_last_in_block(
    block_groups: Dict[int, List[int]],
    available_indices: Optional[List[int]] = None,
) -> Dict[int, bool]:
    """Return a mapping from question index to a flag for the last question in a block."""
    allowed = set(available_indices) if available_indices is not None else None
    last_in_block: Dict[int, bool] = {}

    for indices in block_groups.values():
        filtered = [idx for idx in indices if allowed is None or idx in allowed]
        if filtered:
            last_in_block[filtered[-1]] = True

    return last_in_block
