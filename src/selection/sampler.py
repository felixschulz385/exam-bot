import pandas as pd
from typing import List, Dict, Optional, Tuple
import random
import logging
from functools import lru_cache
from src.utils import build_block_index
from src.data.embeddings import QuestionEmbedder
from .semantic_search import SelectionItem, build_selection_items, score_solution, top_window_size

logger = logging.getLogger(__name__)

class UnifiedSampler:
    """Exact topic/type-ratio sampler with block-question support."""
    
    def __init__(self, 
                model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                cache_dir: Optional[str] = None, 
                seed: Optional[int] = None):
        """Initialize the sampler.
        
        Args:
            model_name: Name of sentence transformer model to use
            cache_dir: Directory to cache embeddings
            seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.embedder = QuestionEmbedder(model_name=model_name, cache_dir=cache_dir)
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            logger.info(f"Unified sampler initialized with seed {seed}")
        else:
            logger.info("Unified sampler initialized without seed")
    
    def _identify_question_blocks(self, questions: pd.DataFrame) -> Tuple[Dict[str, List[int]], Dict[int, bool]]:
        """Identify block questions based on ID patterns.
        
        Args:
            questions: DataFrame containing questions
            
        Returns:
            Tuple of (block_groups, is_block_question)
        """
        block_groups, is_block_question = build_block_index(questions)
        
        logger.info(f"Identified {len(block_groups)} question blocks")
        return block_groups, is_block_question

    def _allocate_exact_points(self, total_points: int, ratios: Dict[str, int], label: str) -> Dict[str, int]:
        """Allocate exact integer point totals from a ratio mapping."""
        if not ratios:
            raise ValueError(f"selection.{label}_ratios must not be empty")

        total_ratio = sum(ratios.values())
        if total_ratio <= 0:
            raise ValueError(f"selection.{label}_ratios must sum to a positive number")

        allocations = {}
        for key, ratio in ratios.items():
            if ratio <= 0:
                raise ValueError(f"selection.{label}_ratios['{key}'] must be positive")

            numerator = total_points * ratio
            if numerator % total_ratio != 0:
                allocations_preview = ", ".join(
                    f"{name}={total_points * value}/{total_ratio}"
                    for name, value in ratios.items()
                )
                raise ValueError(
                    f"Cannot allocate {total_points} points exactly across selection.{label}_ratios because "
                    f"their sum is {total_ratio}. Entry '{key}: {ratio}' would receive {numerator}/{total_ratio} "
                    f"points, which is not an integer. Current implied allocations: {allocations_preview}. "
                    f"Change selection.target_points or adjust the {label} ratios so the total ratio sum divides "
                    f"{total_points} exactly."
                )

            allocations[key] = numerator // total_ratio

        logger.info("Exact %s point allocation: %s", label, allocations)
        return allocations

    def _filter_questions_by_topics(
        self,
        questions: pd.DataFrame,
        topics: Optional[List[str]],
    ) -> pd.DataFrame:
        """Restrict the selection pool to configured topics when provided."""
        if not topics:
            return questions

        filtered = questions[questions["Topic"].isin(topics)].copy()
        missing_topics = sorted(set(topics) - set(filtered["Topic"].dropna().astype(str)))
        if missing_topics:
            logger.warning("Configured topics not found in question bank: %s", missing_topics)
        if filtered.empty:
            raise ValueError(
                "selection.topics does not match any rows in the question bank. "
                f"Configured topics: {topics}"
            )

        logger.info(
            "Restricted question pool to %s configured topics, leaving %s questions",
            len(topics),
            len(filtered),
        )
        return filtered

    def _build_selection_items(
        self,
        questions: pd.DataFrame,
        points_per_type: Dict[str, int]
    ) -> List[SelectionItem]:
        """Collapse the question bank into selectable semantic items."""
        block_groups, _ = self._identify_question_blocks(questions)
        items = build_selection_items(questions, block_groups, points_per_type)

        item_rng = random.Random(self.seed)
        item_rng.shuffle(items)
        items.sort(key=lambda item: (len(item.indices), item.total_points), reverse=True)
        return items

    def _build_total_only_items(self, items: List[SelectionItem]) -> List[SelectionItem]:
        """Replace per-topic accounting with a single total-points bucket."""
        return [
            SelectionItem(
                indices=item.indices,
                topic_points={"__all__": item.total_points},
                type_points=item.type_points,
                total_points=item.total_points,
                semantic_text=item.semantic_text,
            )
            for item in items
        ]

    def _validate_required_type_points(
        self,
        type_point_allocations: Dict[str, int],
        points_per_type: Dict[str, int],
        max_per_type: Dict[str, int]
    ) -> Dict[str, int]:
        """Validate that type point allocations imply an exact integer question count."""
        required_type_counts = {}

        for qtype, allocated_points in type_point_allocations.items():
            point_value = points_per_type.get(qtype)
            if point_value is None:
                raise ValueError(
                    f"selection.type_ratios references question type '{qtype}', but grading.points_per_type does "
                    f"not define it. Add '{qtype}' under grading.points_per_type."
                )

            if allocated_points % point_value != 0:
                raise ValueError(
                    f"Type '{qtype}' receives {allocated_points} points, which is not divisible by its "
                    f"{point_value} points-per-question value. This would require a fractional number of questions. "
                    f"Change selection.type_ratios, selection.target_points, or grading.points_per_type['{qtype}']."
                )

            required_count = allocated_points // point_value
            if qtype in max_per_type and required_count > max_per_type[qtype]:
                raise ValueError(
                    f"Type '{qtype}' requires {required_count} questions to satisfy the exact ratio, "
                    f"but max_per_type only allows {max_per_type[qtype]}."
                )

            required_type_counts[qtype] = required_count

        return required_type_counts

    def _validate_item_feasibility(
        self,
        items: List[SelectionItem],
        topic_targets: Dict[str, int],
        type_targets: Dict[str, int],
        enforce_topic_targets: bool = True,
    ) -> None:
        """Raise targeted errors when topic or type targets are impossible before solving."""
        topic_item_points = {topic: [] for topic in topic_targets}
        type_available_points = {qtype: 0 for qtype in type_targets}

        for item in items:
            for topic, points in item.topic_points.items():
                if topic in topic_item_points:
                    topic_item_points[topic].append(points)
            for qtype, points in item.type_points.items():
                if qtype in type_available_points:
                    type_available_points[qtype] += points

        type_shortfalls = []
        for qtype, required_points in type_targets.items():
            available_points = type_available_points.get(qtype, 0)
            if available_points < required_points:
                type_shortfalls.append(
                    f"{qtype}: need {required_points} points, bank only provides {available_points}"
                )

        if type_shortfalls:
            raise ValueError(
                "The requested type distribution cannot be achieved exactly with the current question bank. "
                "Per-type shortfalls: " + "; ".join(type_shortfalls)
            )

        if not enforce_topic_targets:
            return

        unreachable_topics = []
        for topic, required_points in topic_targets.items():
            item_points = sorted(topic_item_points.get(topic, []))
            if not item_points:
                unreachable_topics.append(
                    f"{topic}: need {required_points} points, but the question bank has no selectable items for this topic"
                )
                continue

            reachable = {0}
            for point_value in item_points:
                reachable |= {
                    current_total + point_value
                    for current_total in list(reachable)
                    if current_total + point_value <= required_points
                }

            if required_points not in reachable:
                reachable_preview = sorted(reachable)
                preview_text = ", ".join(str(value) for value in reachable_preview[:12])
                if len(reachable_preview) > 12:
                    preview_text += ", ..."
                unreachable_topics.append(
                    f"{topic}: need {required_points} points, but atomic item sizes are {item_points} so reachable totals are {{{preview_text}}}"
                )

        if unreachable_topics:
            raise ValueError(
                "The requested topic distribution cannot be achieved exactly with the current question bank. "
                + "Unreachable topic targets: "
                + "; ".join(unreachable_topics)
            )

    def _solve_exact_selection(
        self,
        items: List[SelectionItem],
        topic_targets: Dict[str, int],
        type_targets: Dict[str, int]
    ) -> Optional[List[int]]:
        """Find an exact item subset satisfying topic and type point targets."""
        topics = tuple(topic_targets.keys())
        types = tuple(type_targets.keys())
        topic_index_map = {topic: offset for offset, topic in enumerate(topics)}
        type_index_map = {qtype: offset for offset, qtype in enumerate(types)}

        suffix_total_points = [0] * (len(items) + 1)
        suffix_topic_points = [dict() for _ in range(len(items) + 1)]
        suffix_type_points = [dict() for _ in range(len(items) + 1)]

        for idx in range(len(items) - 1, -1, -1):
            item = items[idx]
            suffix_total_points[idx] = suffix_total_points[idx + 1] + int(item.total_points)

            topic_map = suffix_topic_points[idx + 1].copy()
            for topic, points in item.topic_points.items():
                topic_map[topic] = topic_map.get(topic, 0) + points
            suffix_topic_points[idx] = topic_map

            type_map = suffix_type_points[idx + 1].copy()
            for qtype, points in item.type_points.items():
                type_map[qtype] = type_map.get(qtype, 0) + points
            suffix_type_points[idx] = type_map

        @lru_cache(maxsize=None)
        def search(position: int, topic_state: Tuple[int, ...], type_state: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
            remaining_topic_total = sum(topic_state)
            remaining_type_total = sum(type_state)

            if remaining_topic_total == 0 and remaining_type_total == 0:
                return ()

            if position >= len(items):
                return None

            if remaining_topic_total != remaining_type_total:
                return None

            if suffix_total_points[position] < remaining_topic_total:
                return None

            for offset, topic in enumerate(topics):
                if suffix_topic_points[position].get(topic, 0) < topic_state[offset]:
                    return None

            for offset, qtype in enumerate(types):
                if suffix_type_points[position].get(qtype, 0) < type_state[offset]:
                    return None

            item = items[position]
            new_topic_state = list(topic_state)
            new_type_state = list(type_state)
            fits = True

            for topic, points in item.topic_points.items():
                if topic not in topic_targets:
                    fits = False
                    break
                topic_index = topic_index_map[topic]
                if points > new_topic_state[topic_index]:
                    fits = False
                    break
                new_topic_state[topic_index] -= points

            if fits:
                for qtype, points in item.type_points.items():
                    if qtype not in type_targets:
                        fits = False
                        break
                    type_index = type_index_map[qtype]
                    if points > new_type_state[type_index]:
                        fits = False
                        break
                    new_type_state[type_index] -= points

            if fits:
                selected = search(position + 1, tuple(new_topic_state), tuple(new_type_state))
                if selected is not None:
                    return (position,) + selected

            return search(position + 1, topic_state, type_state)

        result = search(
            0,
            tuple(topic_targets[topic] for topic in topics),
            tuple(type_targets[qtype] for qtype in types),
        )

        if result is None:
            return None

        return list(result)

    def _collect_feasible_solutions(
        self,
        items: List[SelectionItem],
        topic_targets: Dict[str, int],
        type_targets: Dict[str, int],
        candidate_limit: int,
    ) -> List[Tuple[int, ...]]:
        """Collect a bounded set of feasible exact solutions for semantic reranking."""
        topics = tuple(topic_targets.keys())
        types = tuple(type_targets.keys())
        topic_index_map = {topic: offset for offset, topic in enumerate(topics)}
        type_index_map = {qtype: offset for offset, qtype in enumerate(types)}

        suffix_total_points = [0] * (len(items) + 1)
        suffix_topic_points = [dict() for _ in range(len(items) + 1)]
        suffix_type_points = [dict() for _ in range(len(items) + 1)]

        for idx in range(len(items) - 1, -1, -1):
            item = items[idx]
            suffix_total_points[idx] = suffix_total_points[idx + 1] + int(item.total_points)

            topic_map = suffix_topic_points[idx + 1].copy()
            for topic, points in item.topic_points.items():
                topic_map[topic] = topic_map.get(topic, 0) + points
            suffix_topic_points[idx] = topic_map

            type_map = suffix_type_points[idx + 1].copy()
            for qtype, points in item.type_points.items():
                type_map[qtype] = type_map.get(qtype, 0) + points
            suffix_type_points[idx] = type_map

        solutions: List[Tuple[int, ...]] = []

        def search(position: int, topic_state: Tuple[int, ...], type_state: Tuple[int, ...], chosen: Tuple[int, ...]) -> bool:
            remaining_topic_total = sum(topic_state)
            remaining_type_total = sum(type_state)

            if remaining_topic_total == 0 and remaining_type_total == 0:
                solutions.append(chosen)
                return len(solutions) >= candidate_limit

            if position >= len(items):
                return False

            if remaining_topic_total != remaining_type_total:
                return False

            if suffix_total_points[position] < remaining_topic_total:
                return False

            for offset, topic in enumerate(topics):
                if suffix_topic_points[position].get(topic, 0) < topic_state[offset]:
                    return False

            for offset, qtype in enumerate(types):
                if suffix_type_points[position].get(qtype, 0) < type_state[offset]:
                    return False

            item = items[position]
            include_next: List[Optional[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]] = []

            new_topic_state = list(topic_state)
            new_type_state = list(type_state)
            fits = True
            for topic, points in item.topic_points.items():
                topic_idx = topic_index_map.get(topic)
                if topic_idx is None or points > new_topic_state[topic_idx]:
                    fits = False
                    break
                new_topic_state[topic_idx] -= points

            if fits:
                for qtype, points in item.type_points.items():
                    type_idx = type_index_map.get(qtype)
                    if type_idx is None or points > new_type_state[type_idx]:
                        fits = False
                        break
                    new_type_state[type_idx] -= points

            if fits:
                include_next.append((tuple(new_topic_state), tuple(new_type_state), chosen + (position,)))
            include_next.append((topic_state, type_state, chosen))

            if self.seed is not None and len(include_next) > 1:
                branch_rng = random.Random(self.seed + position + len(chosen))
                branch_rng.shuffle(include_next)

            for next_state in include_next:
                if next_state is None:
                    continue
                if search(position + 1, next_state[0], next_state[1], next_state[2]):
                    return True
            return False

        search(
            0,
            tuple(topic_targets[topic] for topic in topics),
            tuple(type_targets[qtype] for qtype in types),
            (),
        )
        return solutions

    def _estimate_candidate_limit(self, item_count: int, cluster_ratio: float) -> int:
        """Interpret cluster_ratio as semantic search breadth."""
        normalized_ratio = max(0.05, min(cluster_ratio, 1.0))
        return max(12, min(250, int(round(item_count * 10 * normalized_ratio))))

    def _select_semantic_solution(
        self,
        items: List[SelectionItem],
        candidate_solutions: List[Tuple[int, ...]],
        question_bank_path: Optional[str],
        cluster_ratio: float,
    ) -> Tuple[int, ...]:
        """Score feasible solutions semantically and sample from the best window."""
        if not candidate_solutions:
            raise ValueError("No feasible semantic candidates were available for selection")
        if len(candidate_solutions) == 1:
            return candidate_solutions[0]

        semantic_texts = [item.semantic_text for item in items]
        distance_matrix = self.embedder.compute_pairwise_distances(
            semantic_texts,
            question_bank_path=question_bank_path,
        )

        scored = [
            (score_solution(solution, distance_matrix), solution)
            for solution in candidate_solutions
        ]
        scored.sort(key=lambda entry: entry[0], reverse=True)

        top_k = top_window_size(len(scored), cluster_ratio)
        top_candidates = scored[:top_k]
        top_score = top_candidates[0][0]
        logger.info(
            "Semantic reranking considered %s feasible solutions; best diversity %.4f, sampling from top %s",
            len(scored),
            top_score,
            top_k,
        )

        if top_k == 1:
            return top_candidates[0][1]

        selection_rng = random.Random(self.seed)
        selected_score, selected_solution = selection_rng.choice(top_candidates)
        logger.info("Selected semantic solution with diversity %.4f", selected_score)
        return selected_solution
    
    def select_unified(self,
                 questions: pd.DataFrame,
                 topic_ratios: Optional[Dict[str, int]],
                 type_ratios: Dict[str, int],
                 points_per_type: Dict[str, int],
                 target_points: Optional[int] = None,
                 max_per_type: Optional[Dict[str, int]] = None,
                 cluster_ratio: float = 0.3,
                 question_bank_path: Optional[str] = None,
                 topics: Optional[List[str]] = None) -> List[int]:
        """Select questions using exact type ratios, with optional exact topic ratios.
        
        Args:
            questions: DataFrame containing questions
            topic_ratios: Optional dictionary mapping topics to exact selection ratios
            type_ratios: Dictionary mapping question types to selection ratios
            points_per_type: Dictionary mapping question types to point values
            target_points: Target total points for the exam
            max_per_type: Dictionary mapping question types to maximum count
            cluster_ratio: Candidate breadth for the semantic reranking search
            question_bank_path: Path to question bank for embedding cache invalidation
            topics: Optional list of allowed topics. When provided without topic_ratios,
                topics are chosen freely from this pool while exactness is enforced only for types.
            
        Returns:
            List of selected question indices
        """
        if target_points is None:
            raise ValueError("selection.target_points is required for exact ratio-based selection")
        if max_per_type is None:
            max_per_type = {}
        if topic_ratios is None:
            topic_ratios = {}

        logger.info("Selecting questions for %s total points with exact type ratios", target_points)
        logger.info("Using max questions per type: %s", max_per_type)

        scoped_questions = self._filter_questions_by_topics(questions, topics)
        type_point_targets = self._allocate_exact_points(target_points, type_ratios, "type")
        required_type_counts = self._validate_required_type_points(
            type_point_targets,
            points_per_type,
            max_per_type,
        )
        logger.info("Required question counts by type: %s", required_type_counts)

        items = self._build_selection_items(scoped_questions, points_per_type)
        has_topic_targets = bool(topic_ratios)
        if has_topic_targets:
            topic_point_targets = self._allocate_exact_points(target_points, topic_ratios, "topic")
            solver_items = items
            solver_topic_targets = topic_point_targets
        else:
            topic_point_targets = {}
            solver_items = self._build_total_only_items(items)
            solver_topic_targets = {"__all__": target_points}
            logger.info(
                "No topic ratios configured; topics will be chosen freely from the configured pool while exact type ratios are enforced"
            )

        self._validate_item_feasibility(
            solver_items,
            solver_topic_targets,
            type_point_targets,
            enforce_topic_targets=has_topic_targets,
        )
        candidate_limit = self._estimate_candidate_limit(len(items), cluster_ratio)
        candidate_solutions = self._collect_feasible_solutions(
            solver_items,
            solver_topic_targets,
            type_point_targets,
            candidate_limit=candidate_limit,
        )

        if not candidate_solutions:
            raise ValueError(
                "The requested selection cannot be achieved precisely with the current question bank, "
                "selection.target_points, selection.type_ratios, and grading.points_per_type values. "
                "If you configured selection.topic_ratios, check those exact topic targets as well."
            )

        if len(candidate_solutions) >= candidate_limit:
            logger.info(
                "Semantic search reached the configured candidate limit (%s); returning the best valid solution from that set",
                candidate_limit,
            )

        selected_item_positions = self._select_semantic_solution(
            items,
            candidate_solutions,
            question_bank_path=question_bank_path,
            cluster_ratio=cluster_ratio,
        )

        selected_indices: List[int] = []
        for item_position in selected_item_positions:
            selected_indices.extend(items[item_position].indices)

        selected_indices = sorted(selected_indices)

        final_topic_points = {}
        final_type_points = {qtype: 0 for qtype in type_point_targets}
        final_type_counts = {qtype: 0 for qtype in type_point_targets}

        for idx in selected_indices:
            topic = scoped_questions.loc[idx, 'Topic']
            qtype = scoped_questions.loc[idx, 'Type']
            point_value = points_per_type[qtype]

            final_topic_points[topic] = final_topic_points.get(topic, 0) + point_value
            if qtype in final_type_points:
                final_type_points[qtype] += point_value
                final_type_counts[qtype] += 1

        if has_topic_targets and final_topic_points != topic_point_targets:
            raise ValueError("Internal error: selected questions do not match the requested exact topic distribution")
        if final_type_points != type_point_targets:
            raise ValueError("Internal error: selected questions do not match the requested exact type distribution")
        logger.info("Final topic point distribution: %s", final_topic_points)
        logger.info("Final type point distribution: %s", final_type_points)
        logger.info("Final question counts by type: %s", final_type_counts)

        return selected_indices
