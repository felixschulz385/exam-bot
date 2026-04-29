from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SelectionItem:
    """Atomic selectable unit for exact selection: one question or one whole block."""

    indices: Tuple[int, ...]
    topic_points: Dict[str, int]
    type_points: Dict[str, int]
    total_points: int
    semantic_text: str


def _raise_missing_type_points(qtype: object, row_indices: Sequence[int]) -> None:
    """Raise a user-facing error for question types missing from grading config."""
    raise ValueError(
        "Question bank contains question type "
        f"{qtype!r} at row index/indices {list(row_indices)}, but grading.points_per_type does not define it. "
        "Either add that type under grading.points_per_type or remove/fix those rows in the question bank."
    )


def build_selection_items(
    questions: pd.DataFrame,
    block_groups: Dict[int, List[int]],
    points_per_type: Dict[str, int],
) -> List[SelectionItem]:
    """Collapse the question bank into semantic selection items."""
    block_question_indices = {idx for indices in block_groups.values() for idx in indices}
    items: List[SelectionItem] = []

    for block_id, indices in block_groups.items():
        topics = {questions.loc[idx, "Topic"] for idx in indices if idx in questions.index}
        if len(topics) != 1:
            raise ValueError(f"Block {block_id} spans multiple topics and cannot be allocated exactly")

        topic = next(iter(topics))
        type_point_totals: Dict[str, int] = {}
        total_points = 0
        text_parts: List[str] = []
        leading_text_added = False

        for idx in indices:
            row = questions.loc[idx]
            qtype = row["Type"]
            if qtype not in points_per_type:
                _raise_missing_type_points(qtype, [idx])

            point_value = points_per_type[qtype]
            type_point_totals[qtype] = type_point_totals.get(qtype, 0) + point_value
            total_points += point_value

            if not leading_text_added and pd.notna(row.get("Leading Text")):
                text_parts.append(str(row.get("Leading Text")).strip())
                leading_text_added = True
            text_parts.append(str(row.get("Question", "")).strip())

        items.append(
            SelectionItem(
                indices=tuple(indices),
                topic_points={topic: total_points},
                type_points=type_point_totals,
                total_points=total_points,
                semantic_text=" ".join(part for part in text_parts if part),
            )
        )

    for idx, row in questions.iterrows():
        if idx in block_question_indices:
            continue

        qtype = row["Type"]
        if qtype not in points_per_type:
            _raise_missing_type_points(qtype, [idx])

        point_value = points_per_type[qtype]
        text_parts = []
        if pd.notna(row.get("Leading Text")):
            text_parts.append(str(row.get("Leading Text")).strip())
        text_parts.append(str(row.get("Question", "")).strip())
        items.append(
            SelectionItem(
                indices=(idx,),
                topic_points={row["Topic"]: point_value},
                type_points={qtype: point_value},
                total_points=point_value,
                semantic_text=" ".join(part for part in text_parts if part),
            )
        )

    return items


def score_solution(item_indices: Sequence[int], distance_matrix: np.ndarray) -> float:
    """Score a solution by average pairwise semantic distance."""
    if len(item_indices) <= 1:
        return 0.0

    distances = []
    for left_pos, left_idx in enumerate(item_indices):
        for right_idx in item_indices[left_pos + 1 :]:
            distances.append(float(distance_matrix[left_idx, right_idx]))

    if not distances:
        return 0.0

    return float(sum(distances) / len(distances))


def top_window_size(candidate_count: int, cluster_ratio: float) -> int:
    """Map the existing cluster_ratio setting to semantic candidate breadth."""
    if candidate_count <= 1:
        return 1

    normalized_ratio = max(0.05, min(cluster_ratio, 1.0))
    return max(1, min(candidate_count, int(round(candidate_count * normalized_ratio))))
