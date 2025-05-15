import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Set
import random
import logging
from sklearn.cluster import KMeans
from src.data.embeddings import QuestionEmbedder

logger = logging.getLogger(__name__)

class UnifiedSampler:
    """Unified selection strategy combining point-based allocation with clustering and block support."""
    
    def __init__(self, 
                model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                cache_dir: Optional[str] = None, 
                seed: Optional[int] = None):
        """Initialize unified sampler.
        
        Args:
            model_name: Name of sentence transformer model to use
            cache_dir: Directory to cache embeddings
            seed: Random seed for reproducibility
        """
        self.embedder = QuestionEmbedder(model_name=model_name, cache_dir=cache_dir)
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
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
        block_groups = {}  # Maps block_id -> list of indices
        is_block_question = {}  # Maps index -> boolean
        
        # First pass: analyze question IDs to identify blocks
        for idx, row in questions.iterrows():
            question_id = str(row.get('ID', ''))
            
            # Check if ID matches block pattern (e.g., "22-3")
            if '-' in question_id:
                base_id, question_num = question_id.split('-', 1)
                
                # Try to convert to integers for validation
                try:
                    base_id = int(base_id)
                    question_num = int(question_num)
                    
                    # It's a block question
                    if base_id not in block_groups:
                        block_groups[base_id] = []
                    
                    block_groups[base_id].append(idx)
                    is_block_question[idx] = True
                    
                except ValueError:
                    # Not a valid block ID pattern
                    is_block_question[idx] = False
            else:
                is_block_question[idx] = False
        
        # Sort questions within each block by their question number
        for block_id, indices in block_groups.items():
            block_groups[block_id] = sorted(
                indices, 
                key=lambda idx: int(str(questions.loc[idx].get('ID', '')).split('-')[1])
            )
        
        logger.info(f"Identified {len(block_groups)} question blocks")
        return block_groups, is_block_question
    
    def _calculate_topic_type_points(self, 
                                    questions: pd.DataFrame, 
                                    topic_points: Dict[str, int], 
                                    type_points: Dict[str, int]) -> pd.DataFrame:
        """Calculate point values for each question based on its topic and type.
        
        Args:
            questions: DataFrame containing questions
            topic_points: Dictionary mapping topics to point values
            type_points: Dictionary mapping question types to point values
            
        Returns:
            DataFrame with added 'points' column
        """
        # Make a copy to avoid modifying the original
        df = questions.copy()
        
        # Add points column
        df['points'] = 0
        
        # Calculate points for each question
        for idx, row in df.iterrows():
            topic = row.get('Topic')
            qtype = row.get('Type')
            
            topic_value = topic_points.get(topic, 0)
            type_value = type_points.get(qtype, 0)
            
            # Total points is the sum of topic and type points
            df.at[idx, 'points'] = topic_value + type_value
        
        return df
    
    def _calculate_block_points(self, 
                              questions: pd.DataFrame, 
                              block_groups: Dict[str, List[int]]) -> Dict[str, int]:
        """Calculate total points for each block.
        
        Args:
            questions: DataFrame containing questions with 'points' column
            block_groups: Dictionary mapping block IDs to lists of question indices
            
        Returns:
            Dictionary mapping block IDs to total point values
        """
        block_points = {}
        
        for block_id, indices in block_groups.items():
            block_points[block_id] = questions.loc[indices, 'points'].sum()
            
        return block_points
    
    def _cluster_non_block_questions(self, 
                                   questions: pd.DataFrame, 
                                   topic: str,
                                   non_block_indices: List[int],
                                   cluster_ratio: float = 0.3,
                                   question_bank_path: Optional[str] = None) -> Dict[int, List[int]]:
        """Cluster non-block questions within a topic.
        
        Args:
            questions: DataFrame containing questions
            topic: Topic to cluster questions for
            non_block_indices: Indices of non-block questions
            cluster_ratio: Ratio to determine number of clusters
            question_bank_path: Path to question bank for caching embeddings
            
        Returns:
            Dictionary mapping cluster IDs to lists of question indices
        """
        if not non_block_indices:
            return {}
            
        # Filter to non-block questions in this topic
        topic_indices = [idx for idx in non_block_indices 
                         if idx in questions.index and questions.loc[idx, 'Topic'] == topic]
        
        if len(topic_indices) <= 1:
            return {0: topic_indices}
            
        # Get question texts
        question_texts = [questions.loc[idx, 'Question'] for idx in topic_indices]
        
        # Generate embeddings
        embeddings = self.embedder.embed_questions(question_texts, question_bank_path)
        
        # Determine number of clusters
        n_clusters = max(1, min(int(round(len(topic_indices) * cluster_ratio)), len(topic_indices) - 1))
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed)
        cluster_assignments = kmeans.fit_predict(embeddings)
        
        # Organize questions by cluster
        clusters = {i: [] for i in range(n_clusters)}
        for i, idx in enumerate(topic_indices):
            cluster_id = cluster_assignments[i]
            clusters[cluster_id].append(idx)
        
        logger.info(f"Clustered {len(topic_indices)} non-block questions in topic '{topic}' into {n_clusters} clusters")
        return clusters
    
    def select_unified(self,
                 questions: pd.DataFrame,
                 topic_points: Dict[str, int],
                 type_points: Dict[str, int],
                 total_questions: Optional[int] = None,
                 target_points: Optional[int] = None,
                 dynamic_sizing: bool = False,
                 max_per_type: Optional[Dict[str, int]] = None,
                 cluster_ratio: float = 0.3,
                 question_bank_path: Optional[str] = None) -> List[int]:
        """Select questions using unified approach.
        
        Args:
            questions: DataFrame containing questions
            topic_points: Dictionary mapping topics to point values
            type_points: Dictionary mapping question types to point values
            total_questions: Total number of questions to select
            target_points: Target total points for the exam
            dynamic_sizing: Whether to calculate exam size dynamically
            max_per_type: Dictionary mapping question types to maximum count
            cluster_ratio: Ratio to determine number of clusters
            question_bank_path: Path to question bank for caching embeddings
            
        Returns:
            List of selected question indices
        """
        # Check if we're using target points
        if target_points is not None:
            logger.info(f"Targeting {target_points} total exam points")
            # We'll handle this with point-based selection
        elif dynamic_sizing:
            total_questions = self.calculate_dynamic_exam_size(topic_points)
        elif total_questions is None:
            # Default if nothing specified
            total_questions = 30
            logger.info(f"Using default total questions: {total_questions}")

        if max_per_type is None:
            max_per_type = {}
            
        logger.info(f"Using max questions per type: {max_per_type}")

        # Identify block questions
        block_groups, is_block_question = self._identify_question_blocks(questions)
        
        # Calculate point values for each question
        questions_with_points = self._calculate_topic_type_points(questions, topic_points, type_points)
        
        # If using target points, use a different allocation approach
        if target_points is not None:
            # Allocate points to topics while respecting the target total
            topic_allocation = self._allocate_points_by_target(topic_points, target_points)
        else:
            # Use the regular allocation based on total questions
            topic_allocation = self._allocate_by_points(topic_points, total_questions)
        
        # When selecting questions within each topic, respect max_per_type limits
        selected_indices = []
        type_counts = {qtype: 0 for qtype in type_points.keys()}
        total_selected_points = 0
        
        # Process block questions first, but randomly
        block_selected_indices = self._select_block_questions(
            questions_with_points, 
            block_groups,
            topic_allocation,
            max_per_type,
            type_counts
        )
        
        # Calculate points from block questions
        block_points = sum(type_points.get(questions.loc[idx, 'Type'], 0) for idx in block_selected_indices)
        total_selected_points += block_points
        
        # Update the counts with what was selected in blocks
        for idx in block_selected_indices:
            qtype = questions.loc[idx, 'Type']
            if qtype in type_counts:
                type_counts[qtype] += 1

        selected_indices.extend(block_selected_indices)
        
        # Now process topics and select non-block questions
        # Randomize topic order
        topics = list(topic_allocation.keys())
        if self.seed is not None:
            random.Random(self.seed).shuffle(topics)
        else:
            random.shuffle(topics)
            
        logger.info(f"Processing topics in random order: {topics}")
        
        for topic in topics:
            # If using target points, calculate remaining points needed
            if target_points is not None:
                remaining_points = target_points - total_selected_points
                if remaining_points <= 0:
                    logger.info(f"Reached target points, stopping selection")
                    break
                    
                # Adjust topic allocation based on remaining points
                count = min(topic_allocation[topic], remaining_points)
            else:
                count = topic_allocation[topic]
                
            # Filter questions for this topic
            topic_questions = questions_with_points[questions_with_points['Topic'] == topic]
            
            # Filter out questions already selected
            topic_questions = topic_questions[~topic_questions.index.isin(selected_indices)]
            
            # Check if we still need questions from this topic
            topic_block_points = sum(
                type_points.get(questions.loc[idx, 'Type'], 0) 
                for idx in block_selected_indices 
                if questions.loc[idx, 'Topic'] == topic
            )
            
            remaining_topic_points = count - topic_block_points
            
            if remaining_topic_points <= 0:
                logger.info(f"Topic {topic} already has enough points from blocks, skipping")
                continue
                
            # Select non-block questions for this topic based on points
            topic_selected_indices = self._select_non_block_questions_by_points(
                topic_questions,
                remaining_topic_points,
                max_per_type,
                type_counts,
                type_points,
                cluster_ratio,
                question_bank_path
            )
            
            # Update type counts and total points
            for idx in topic_selected_indices:
                qtype = questions.loc[idx, 'Type']
                if qtype in type_counts:
                    type_counts[qtype] += 1
                total_selected_points += type_points.get(qtype, 0)
            
            selected_indices.extend(topic_selected_indices)
        
        # Check final point total
        final_points = sum(type_points.get(questions.loc[idx, 'Type'], 0) for idx in selected_indices)
        
        # Verify we hit target exactly
        if target_points is not None and final_points != target_points:
            logger.warning(f"Did not achieve exact point total: {final_points} vs target {target_points}")
            
            # Final adjustment if needed
            if len(selected_indices) > 0:
                # Try adding or removing individual questions to hit target exactly
                if final_points < target_points:
                    deficit = target_points - final_points
                    logger.info(f"Attempting to add exactly {deficit} more points")
                    
                    # Find questions not already selected
                    remaining_qs = questions[~questions.index.isin(selected_indices)]
                    
                    # Try to find a question worth exactly the deficit
                    for idx, row in remaining_qs.iterrows():
                        qtype = row['Type']
                        points = type_points.get(qtype, 0)
                        
                        if points == deficit:
                            selected_indices.append(idx)
                            final_points += points
                            logger.info(f"Added question {idx} of type {qtype} worth {points} to hit target exactly")
                            break
                    
                elif final_points > target_points:
                    excess = final_points - target_points
                    logger.info(f"Attempting to remove exactly {excess} points")
                    
                    # Try to find a question worth exactly the excess to remove
                    for idx in selected_indices[:]:  # Copy to allow modification during iteration
                        qtype = questions.loc[idx, 'Type']
                        points = type_points.get(qtype, 0)
                        
                        if points == excess:
                            selected_indices.remove(idx)
                            final_points -= points
                            logger.info(f"Removed question {idx} of type {qtype} worth {points} to hit target exactly")
                            break
        
        # Final log
        final_points = sum(type_points.get(questions.loc[idx, 'Type'], 0) for idx in selected_indices)
        logger.info(f"Final exam point total: {final_points}" + 
                   (f"/{target_points}" if target_points is not None else ""))
        
        # Count by type
        final_type_counts = {}
        for idx in selected_indices:
            qtype = questions.loc[idx, 'Type']
            final_type_counts[qtype] = final_type_counts.get(qtype, 0) + 1
        
        logger.info(f"Final question type distribution: {final_type_counts}")
        
        return selected_indices

    def _select_block_questions(self,
                               questions: pd.DataFrame,
                               block_groups: Dict[str, List[int]],
                               topic_allocation: Dict[str, int],
                               max_per_type: Dict[str, int],
                               type_counts: Dict[str, int]) -> List[int]:
        """Select block questions respecting type limits.
        
        Args:
            questions: DataFrame containing questions with points
            block_groups: Dictionary mapping block IDs to question indices
            topic_allocation: Dictionary mapping topics to allocated question counts
            max_per_type: Dictionary mapping question types to maximum count
            type_counts: Dictionary mapping question types to current counts
            
        Returns:
            List of selected question indices
        """
        selected_indices = []
        
        # Convert block_groups keys to a list and shuffle to randomize block selection
        block_ids = list(block_groups.keys())
        if self.seed is not None:
            random.Random(self.seed).shuffle(block_ids)
        else:
            random.shuffle(block_ids)
            
        logger.info(f"Processing blocks in random order: {block_ids}")
        
        for block_id in block_ids:
            indices = block_groups[block_id]
            # Check if all questions in the block are of the same topic
            topics = set(questions.loc[idx, 'Topic'] for idx in indices if idx in questions.index)
            
            if len(topics) != 1:
                logger.warning(f"Block {block_id} contains questions from multiple topics, skipping")
                continue
                
            topic = list(topics)[0]
            
            # Check if adding this block exceeds topic allocation
            if len(indices) > topic_allocation.get(topic, 0):
                logger.info(f"Block {block_id} would exceed topic allocation for {topic}, skipping")
                continue
                
            # Check if adding this block exceeds max_per_type for any question type
            block_type_counts = {}
            for idx in indices:
                if idx not in questions.index:
                    continue
                    
                qtype = questions.loc[idx, 'Type']
                block_type_counts[qtype] = block_type_counts.get(qtype, 0) + 1
                
            exceeds_type_limit = False
            for qtype, count in block_type_counts.items():
                if qtype in max_per_type and type_counts.get(qtype, 0) + count > max_per_type[qtype]:
                    logger.info(f"Block {block_id} would exceed max questions for type {qtype}, skipping")
                    exceeds_type_limit = True
                    break
                    
            if exceeds_type_limit:
                continue
                
            # Add the block questions to the selection
            selected_indices.extend([idx for idx in indices if idx in questions.index])
            
            # Update type counts
            for idx in indices:
                if idx not in questions.index:
                    continue
                    
                qtype = questions.loc[idx, 'Type']
                if qtype in type_counts:
                    type_counts[qtype] += 1
                    
            # Update topic allocation
            topic_allocation[topic] -= len(indices)
            
        return selected_indices

    def _select_non_block_questions(self,
                                  questions: pd.DataFrame,
                                  count: int,
                                  max_per_type: Dict[str, int],
                                  type_counts: Dict[str, int],
                                  cluster_ratio: float,
                                  question_bank_path: Optional[str]) -> List[int]:
        """Select non-block questions respecting type limits.
        
        Args:
            questions: DataFrame containing questions with points
            count: Number of questions to select
            max_per_type: Dictionary mapping question types to maximum count
            type_counts: Dictionary mapping question types to current counts
            cluster_ratio: Ratio to determine number of clusters
            question_bank_path: Path to question bank for caching embeddings
            
        Returns:
            List of selected question indices
        """
        if questions.empty or count <= 0:
            return []
            
        # Group questions by type
        questions_by_type = {}
        for qtype, group in questions.groupby('Type'):
            questions_by_type[qtype] = group
            
        selected_indices = []
        
        # Select questions by type, respecting max_per_type
        for qtype, type_questions in questions_by_type.items():
            # Calculate how many questions of this type we can still add
            remaining_capacity = max_per_type.get(qtype, float('inf')) - type_counts.get(qtype, 0)
            
            if remaining_capacity <= 0:
                logger.info(f"Type {qtype} has reached its maximum capacity, skipping")
                continue
                
            # Calculate how many questions of this type to select
            type_count = min(remaining_capacity, count - len(selected_indices))
            
            if type_count <= 0:
                continue
                
            # Use clustering to select diverse questions of this type
            type_selected_indices = self._select_clustered_questions(
                type_questions, 
                type_count,
                cluster_ratio,
                question_bank_path
            )
            
            selected_indices.extend(type_selected_indices)
            
            # Stop if we've selected enough questions
            if len(selected_indices) >= count:
                break
                
        return selected_indices

    def _select_clustered_questions(self,
                                 questions: pd.DataFrame,
                                 count: int,
                                 cluster_ratio: float,
                                 question_bank_path: Optional[str]) -> List[int]:
        """Select questions using clustering for diversity.
        
        Args:
            questions: DataFrame containing questions
            count: Number of questions to select
            cluster_ratio: Ratio to determine number of clusters
            question_bank_path: Path to question bank for caching embeddings
            
        Returns:
            List of selected question indices
        """
        if questions.empty or count <= 0:
            return []
            
        if len(questions) <= count:
            return list(questions.index)
            
        # Get question texts for embedding
        texts = questions['Question'].tolist()
        
        # Calculate embeddings - fix method name from embed_texts to embed_questions
        embeddings = self.embedder.embed_questions(texts, question_bank_path)
        
        # Determine number of clusters
        n_clusters = max(1, min(int(len(questions) * cluster_ratio), count))
        
        # Apply clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
        clusters = kmeans.fit_predict(embeddings)
        
        # Select questions from each cluster
        selected_indices = []
        
        # Calculate questions per cluster
        questions_per_cluster = self._allocate_evenly(n_clusters, count)
        
        for cluster_id in range(n_clusters):
            cluster_questions = questions.iloc[[i for i, c in enumerate(clusters) if c == cluster_id]]
            
            # Select the requested number of questions from this cluster
            cluster_count = questions_per_cluster[cluster_id]
            
            if len(cluster_questions) <= cluster_count:
                selected_indices.extend(list(cluster_questions.index))
            else:
                # Select randomly from this cluster
                cluster_selected = cluster_questions.sample(
                    n=cluster_count, 
                    random_state=self.seed + cluster_id if self.seed else None
                )
                selected_indices.extend(list(cluster_selected.index))
                
        return selected_indices
    
    def _allocate_by_points(self, point_dict: Dict[str, int], total: int) -> Dict[str, int]:
        """Allocate a total number proportionally based on points.
        
        Args:
            point_dict: Dictionary mapping keys to point values
            total: Total to allocate
            
        Returns:
            Dictionary with allocations
        """
        total_points = sum(point_dict.values())
        if total_points == 0:
            return {k: 0 for k in point_dict}
            
        allocations = {}
        remaining = total
        
        # First pass: allocate proportionally
        for key, points in point_dict.items():
            allocation = int((points / total_points) * total)
            allocations[key] = allocation
            remaining -= allocation
        
        # Distribute any remaining
        keys_by_points = sorted(point_dict.items(), key=lambda x: x[1], reverse=True)
        for key, _ in keys_by_points:
            if remaining <= 0:
                break
            allocations[key] += 1
            remaining -= 1
        
        return allocations
    
    def _allocate_evenly(self, num_buckets: int, total: int) -> List[int]:
        """Allocate a total evenly among buckets.
        
        Args:
            num_buckets: Number of buckets
            total: Total to allocate
            
        Returns:
            List of allocations
        """
        if num_buckets == 0:
            return []
            
        base = total // num_buckets
        remainder = total % num_buckets
        
        allocations = [base] * num_buckets
        
        # Distribute remainder
        for i in range(remainder):
            allocations[i] += 1
            
        return allocations
    
    def calculate_dynamic_exam_size(self, topic_points: Dict[str, int]) -> int:
        """Calculate the total number of questions dynamically based on topic points.
        
        Args:
            topic_points: Dictionary mapping topics to point values
            
        Returns:
            int: Calculated total number of questions (sum of all topic points)
        """
        # Sum of all topic points becomes the total number of questions
        total_questions = sum(topic_points.values())
        logger.info(f"Dynamically calculated exam size: {total_questions} questions based on sum of topic points")
        return total_questions
    
    def _allocate_points_by_target(self, point_dict: Dict[str, int], target_points: int) -> Dict[str, int]:
        """Allocate target points proportionally based on topic weights.
        
        Args:
            point_dict: Dictionary mapping topics to point values
            target_points: Target total points for the exam
            
        Returns:
            Dictionary mapping topics to point allocations
        """
        total_weights = sum(point_dict.values())
        if total_weights == 0:
            return {k: 0 for k in point_dict}
            
        allocations = {}
        remaining = target_points
        
        # First pass: allocate proportionally
        for key, weight in point_dict.items():
            allocation = int((weight / total_weights) * target_points)
            allocations[key] = allocation
            remaining -= allocation
        
        # Distribute any remaining points
        keys_by_points = sorted(point_dict.items(), key=lambda x: x[1], reverse=True)
        for key, _ in keys_by_points:
            if remaining <= 0:
                break
            allocations[key] += 1
            remaining -= 1
        
        logger.info(f"Allocated {target_points} points across topics: {allocations}")
        return allocations

    def _select_non_block_questions_by_points(self,
                                  questions: pd.DataFrame,
                                  target_points: int,
                                  max_per_type: Dict[str, int],
                                  type_counts: Dict[str, int],
                                  type_points: Dict[str, int],
                                  cluster_ratio: float,
                                  question_bank_path: Optional[str]) -> List[int]:
        """Select non-block questions to exactly meet target points.
        
        Args:
            questions: DataFrame containing questions with points
            target_points: Target points to select
            max_per_type: Dictionary mapping question types to maximum count
            type_counts: Dictionary mapping question types to current counts
            type_points: Dictionary mapping question types to point values
            cluster_ratio: Ratio to determine number of clusters
            question_bank_path: Path to question bank for caching embeddings
            
        Returns:
            List of selected question indices
        """
        if questions.empty or target_points <= 0:
            return []
            
        # Group questions by type
        questions_by_type = {}
        type_counts_available = {}
        
        for qtype, group in questions.groupby('Type'):
            questions_by_type[qtype] = group
            type_counts_available[qtype] = len(group)
        
        logger.info(f"Available questions by type: {type_counts_available}")
        
        selected_indices = []
        current_points = 0
        
        # Calculate target distribution percentages for each type
        available_types = [qtype for qtype, count in type_counts_available.items() if count > 0]
        
        # Initialize target distributions - aim for equal representation of each type
        if available_types:
            base_percentage = 1.0 / len(available_types)
            target_distribution = {qtype: base_percentage for qtype in available_types}
            
            # Adjust slightly to favor higher-point questions (but not as extremely as before)
            total_type_points = sum(type_points.get(qtype, 1) for qtype in available_types)
            
            for qtype in available_types:
                points = type_points.get(qtype, 1)
                # Mix of equal distribution (70%) and point-based distribution (30%)
                point_factor = 0.3 * (points / total_type_points) if total_type_points > 0 else 0
                equal_factor = 0.7 * base_percentage
                target_distribution[qtype] = equal_factor + point_factor
            
            # Normalize to ensure sum is 1.0
            total = sum(target_distribution.values())
            target_distribution = {k: v/total for k, v in target_distribution.items()}
            
            logger.info(f"Target question type distribution: {target_distribution}")
        
        # Calculate how many points to allocate to each type
        type_point_allocation = {}
        remaining_points = target_points
        
        for qtype in available_types:
            # Calculate target points for this type
            type_target = int(target_distribution[qtype] * target_points)
            # Ensure at least some points for each type if possible
            type_point_allocation[qtype] = max(1, type_target)
            remaining_points -= type_point_allocation[qtype]
        
        # Distribute any remaining points proportionally
        if remaining_points > 0:
            for qtype in sorted(available_types, 
                            key=lambda t: type_point_allocation[t]/target_distribution[t] 
                                        if target_distribution[t] > 0 else 0):
                type_point_allocation[qtype] += 1
                remaining_points -= 1
                if remaining_points <= 0:
                    break
        
        # First pass: select questions by type according to allocations
        candidate_indices = []
        potential_points = 0
        type_selected = {qtype: [] for qtype in available_types}
        
        # Randomize type order for balanced selection
        random_types = available_types.copy()
        if self.seed is not None:
            random.Random(self.seed).shuffle(random_types)
        else:
            random.shuffle(random_types)
        
        # First phase: Select some questions from each type, but don't exceed target
        for qtype in random_types:
            if qtype not in questions_by_type or qtype not in type_point_allocation:
                continue
                
            points_per_question = type_points.get(qtype, 1)
            if points_per_question <= 0:
                continue
                
            type_questions = questions_by_type[qtype]
            
            # Calculate how many questions of this type we can still add
            remaining_capacity = max_per_type.get(qtype, float('inf')) - type_counts.get(qtype, 0)
            
            if remaining_capacity <= 0:
                logger.info(f"Type {qtype} has reached its maximum capacity, skipping")
                continue
                
            # Calculate how many questions to select based on allocation
            allocated_points = type_point_allocation[qtype]
            type_count = min(remaining_capacity, allocated_points // points_per_question)
            
            # Ensure at least one question if we have points allocated
            if allocated_points > 0 and type_count == 0:
                type_count = 1
            
            if type_count <= 0:
                continue
            
            # Use clustering to select diverse questions of this type
            type_selected_indices = self._select_clustered_questions(
                type_questions, 
                type_count,
                cluster_ratio,
                question_bank_path
            )
            
            # Don't add them yet, just track them
            type_selected[qtype] = type_selected_indices
            potential_points += len(type_selected_indices) * points_per_question
        
        # Calculate exact point adjustment needed
        logger.info(f"Initial selection: {potential_points} points vs target {target_points}")
        
        # Second phase: Adjust selection to exactly match target points
        if potential_points < target_points:
            # We need to add more questions - prioritize types with few questions
            deficit = target_points - potential_points
            logger.info(f"Need {deficit} more points to reach target")
            
            # Get available questions that weren't selected in first pass
            available_questions = questions[~questions.index.isin([idx for idxs in type_selected.values() for idx in idxs])]
            
            # Sort types by representation (prioritizing types with fewer questions)
            types_by_representation = sorted(
                [(qtype, len(selected)) for qtype, selected in type_selected.items()],
                key=lambda x: (x[1], -type_points.get(x[0], 0))  # Fewest questions first, break ties with higher points
            )
            
            # Add questions until we reach target
            for qtype, _ in types_by_representation:
                if deficit <= 0:
                    break
                    
                points_per_question = type_points.get(qtype, 1)
                
                # Skip if this type can't contribute to reaching target exactly
                if points_per_question > deficit:
                    continue
                    
                # Get unselected questions of this type
                unselected = available_questions[available_questions['Type'] == qtype]
                
                if unselected.empty:
                    continue
                    
                # Check if we can add more of this type
                remaining_capacity = max_per_type.get(qtype, float('inf')) - (
                    type_counts.get(qtype, 0) + len(type_selected[qtype])
                )
                
                if remaining_capacity <= 0:
                    continue
                
                # How many more questions to add
                add_count = min(remaining_capacity, deficit // points_per_question)
                
                if add_count <= 0:
                    continue
                    
                # Select additional questions
                if len(unselected) <= add_count:
                    additional_indices = list(unselected.index)
                else:
                    # Sample randomly for simplicity, could enhance with clustering
                    additional_indices = unselected.sample(
                        n=add_count,
                        random_state=self.seed+99 if self.seed else None
                    ).index.tolist()
                    
                # Add to our selections
                type_selected[qtype].extend(additional_indices)
                deficit -= add_count * points_per_question
                
                logger.info(f"Added {add_count} additional {qtype} questions, remaining deficit: {deficit}")
                    
        elif potential_points > target_points:
            # We need to remove some questions - prioritize types with many questions
            excess = potential_points - target_points
            logger.info(f"Need to remove {excess} points to reach target")
            
            # Sort types by representation (prioritizing types with more questions)
            types_by_representation = sorted(
                [(qtype, len(selected)) for qtype, selected in type_selected.items() if len(selected) > 0],
                key=lambda x: (-x[1], type_points.get(x[0], 0))  # Most questions first, break ties with lower points
            )
            
            # Remove questions until we reach target
            for qtype, _ in types_by_representation:
                if excess <= 0:
                    break
                    
                points_per_question = type_points.get(qtype, 1)
                
                # Skip if this type can't help reduce excess exactly
                if points_per_question > excess:
                    continue
                    
                # How many questions to remove
                remove_count = min(len(type_selected[qtype]), excess // points_per_question)
                
                if remove_count <= 0:
                    continue
                    
                # Remove randomly
                if self.seed is not None:
                    random.Random(self.seed+999).shuffle(type_selected[qtype])
                else:
                    random.shuffle(type_selected[qtype])
                    
                type_selected[qtype] = type_selected[qtype][:-remove_count]
                excess -= remove_count * points_per_question
                
                logger.info(f"Removed {remove_count} {qtype} questions, remaining excess: {excess}")
        
        # Final verification
        for qtype, indices in type_selected.items():
            selected_indices.extend(indices)
            current_points += len(indices) * type_points.get(qtype, 1)
        
        # Check if we hit target exactly
        if current_points != target_points:
            logger.warning(f"Failed to achieve exact target points: got {current_points}, wanted {target_points}")
            
            # Last-ditch attempt: try more aggressive correction
            if current_points < target_points:
                deficit = target_points - current_points
                logger.info(f"Final deficit: {deficit} points")
                
                # Try to find questions that add up to exactly the deficit
                # Look for a question worth exactly the deficit
                for qtype, point_value in type_points.items():
                    if point_value == deficit:
                        # Find an unused question of this type
                        unused = questions[
                            (questions['Type'] == qtype) & 
                            (~questions.index.isin(selected_indices))
                        ]
                        
                        if not unused.empty:
                            # Add one question
                            add_idx = unused.index[0]
                            selected_indices.append(add_idx)
                            current_points += point_value
                            logger.info(f"Added one {qtype} question worth {point_value} points to hit target exactly")
                            break
            
            elif current_points > target_points:
                excess = current_points - target_points
                logger.info(f"Final excess: {excess} points")
                
                # First try: find a question worth exactly the excess
                for qtype, point_value in type_points.items():
                    if point_value == excess:
                        # Find used questions of this type
                        used_indices = [
                            idx for idx in selected_indices 
                            if questions.loc[idx, 'Type'] == qtype
                        ]
                        
                        if used_indices:
                            # Remove one question
                            selected_indices.remove(used_indices[0])
                            current_points -= point_value
                            logger.info(f"Removed one {qtype} question worth {point_value} points to hit target exactly")
                            break

                # If we still have excess points, try removing and replacing
                if current_points > target_points:
                    # Try removing a higher-point question and adding lower-point questions
                    # First, build a list of all used question types by point value (highest to lowest)
                    used_types = sorted(
                        [(qtype, point_value) for qtype, point_value in type_points.items() 
                         if any(questions.loc[idx, 'Type'] == qtype for idx in selected_indices)],
                        key=lambda x: -x[1]  # Sort by point value, highest first
                    )
                    
                    # Try removing one higher-point question
                    for remove_type, remove_points in used_types:
                        # Only consider if removing would help (not make deficit worse)
                        if remove_points <= excess:
                            continue
                            
                        # Find a question of this type to remove
                        remove_candidates = [
                            idx for idx in selected_indices 
                            if questions.loc[idx, 'Type'] == remove_type
                        ]
                        
                        if not remove_candidates:
                            continue
                        
                        # Try removing this question
                        test_selected = selected_indices.copy()
                        test_selected.remove(remove_candidates[0])
                        new_total = sum(type_points.get(questions.loc[idx, 'Type'], 0) for idx in test_selected)
                        new_deficit = target_points - new_total
                        
                        # If removing makes us too low, try adding back questions to hit target
                        if new_deficit > 0:
                            # Look for question types to add that can exactly fill the new deficit
                            for add_type, add_points in sorted(type_points.items(), key=lambda x: x[1]):
                                # Skip if this won't help
                                if add_points > new_deficit:
                                    continue
                                    
                                # How many to add?
                                add_count = new_deficit // add_points
                                if add_count <= 0 or add_count * add_points != new_deficit:
                                    continue
                                    
                                # Find questions to add
                                add_candidates = questions[
                                    (questions['Type'] == add_type) & 
                                    (~questions.index.isin(test_selected))
                                ]
                                
                                if len(add_candidates) < add_count:
                                    continue
                                    
                                # Perfect match! Remove and add
                                selected_indices.remove(remove_candidates[0])
                                for idx in add_candidates.index[:add_count]:
                                    selected_indices.append(idx)
                                    
                                logger.info(f"Fixed point total by removing one {remove_type} ({remove_points} pts) and adding {add_count} {add_type} ({add_count*add_points} pts)")
                                current_points = target_points
                                break
                    
                        # If we fixed it, stop the outer loop too
                        if current_points == target_points:
                            break
        
        # Final log of what we selected
        final_distribution = {}
        final_points = 0
        for idx in selected_indices:
            qtype = questions.loc[idx, 'Type']
            final_distribution[qtype] = final_distribution.get(qtype, 0) + 1
            final_points += type_points.get(qtype, 1)
            
        logger.info(f"Final point total: {final_points}/{target_points}")
        logger.info(f"Final non-block question distribution by type: {final_distribution}")
        
        return selected_indices