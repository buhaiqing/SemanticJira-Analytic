"""Incremental update handler for Jira task analysis with performance optimizations."""

import logging
from typing import List, Dict, Tuple, Set, Optional
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import threading
from collections import OrderedDict

from app.models import JiraTask, ProcessedTask

logger = logging.getLogger(__name__)


class FingerprintCache:
    """
    Thread-safe cache for task fingerprints with LRU eviction.

    Features:
    - Fast fingerprint lookup for change detection
    - Automatic eviction of old entries
    - Thread-safe access
    - O(1) LRU operations using OrderedDict
    """

    def __init__(self, max_size: int = 10000):
        self._cache: OrderedDict[str, Tuple[str, datetime]] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.RLock()

    def get(self, task_id: str) -> Tuple[Optional[str], Optional[datetime]]:
        """Get fingerprint and timestamp for task."""
        with self._lock:
            if task_id in self._cache:
                # O(1) move_to_end instead of O(n) remove+append
                self._cache.move_to_end(task_id)
                return self._cache[task_id]
            return None, None

    def put(self, task_id: str, fingerprint: str, timestamp: datetime) -> None:
        """Put fingerprint in cache."""
        with self._lock:
            if task_id in self._cache:
                self._cache.move_to_end(task_id)
            elif len(self._cache) >= self._max_size:
                # O(1) eviction of oldest item
                self._cache.popitem(last=False)

            self._cache[task_id] = (fingerprint, timestamp)

    def clear(self) -> None:
        """Clear all cached fingerprints."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        """Return current cache size."""
        return len(self._cache)


class IncrementalUpdateHandler:
    """
    High-performance incremental update handler with optimizations.

    Features:
    - Fast fingerprint-based change detection
    - Batched timestamp comparisons
    - Efficient similarity calculations with caching
    - Parallel processing support for large datasets
    """

    # Similarity threshold constants
    CONTENT_SIMILARITY_THRESHOLD = 0.8
    TIMESTAMP_TOLERANCE_SECONDS = 1

    def __init__(self, enable_optimization: bool = True):
        """
        Initialize incremental update handler.

        Args:
            enable_optimization: Enable performance optimizations
        """
        self.existing_tasks: Dict[str, ProcessedTask] = {}
        self._fingerprint_cache = FingerprintCache() if enable_optimization else None
        self._similarity_cache: Dict[str, float] = {} if enable_optimization else None
        self._cache_lock = threading.Lock()
        self._enable_optimization = enable_optimization
        self._stats = {
            'comparisons_made': 0,
            'cache_hits': 0,
            'new_tasks': 0,
            'updated_tasks': 0,
            'unchanged_tasks': 0
        }

    def load_existing_data(self, existing_tasks: List[ProcessedTask]) -> None:
        """
        Load existing processed tasks for comparison.

        Args:
            existing_tasks: List of existing processed tasks
        """
        self.existing_tasks = {task.issue_id: task for task in existing_tasks}

        # Pre-compute fingerprints for all existing tasks
        if self._fingerprint_cache:
            for task in existing_tasks:
                fingerprint = self._compute_fingerprint(task)
                self._fingerprint_cache.put(task.issue_id, fingerprint, task.updated_at)

        logger.info(f"Loaded {len(self.existing_tasks)} existing tasks for incremental comparison")

    def categorize_updates(self, new_tasks: List[JiraTask]) -> Tuple[List[JiraTask], List[Tuple[JiraTask, ProcessedTask]]]:
        """
        Categorize incoming tasks as new or updated with optimized comparison.

        Args:
            new_tasks: List of new tasks to categorize

        Returns:
            Tuple of (new_tasks_list, updated_tasks_pairs)
        """
        new_tasks_list = []
        updated_tasks_pairs = []

        # Batch process for efficiency
        for task in new_tasks:
            category = self._categorize_single_task(task)
            if category == 'new':
                new_tasks_list.append(task)
                self._stats['new_tasks'] += 1
            elif category == 'updated':
                existing_task = self.existing_tasks[task.issue_id]
                updated_tasks_pairs.append((task, existing_task))
                self._stats['updated_tasks'] += 1
            else:
                self._stats['unchanged_tasks'] += 1

        logger.info(
            f"Categorized {len(new_tasks_list)} new tasks and {len(updated_tasks_pairs)} updated tasks "
            f"(skipped {self._stats['unchanged_tasks']} unchanged)"
        )
        return new_tasks_list, updated_tasks_pairs

    def _categorize_single_task(self, task: JiraTask) -> str:
        """
        Categorize a single task as new, updated, or unchanged.

        Args:
            task: Task to categorize

        Returns:
            'new', 'updated', or 'unchanged'
        """
        self._stats['comparisons_made'] += 1

        # Check if task exists
        if task.issue_id not in self.existing_tasks:
            return 'new'

        existing_task = self.existing_tasks[task.issue_id]

        # Fast path: check fingerprint cache
        if self._fingerprint_cache:
            cached_fingerprint, _ = self._fingerprint_cache.get(task.issue_id)
            current_fingerprint = self._compute_fingerprint(task)

            if cached_fingerprint and cached_fingerprint == current_fingerprint:
                logger.debug(f"Task {task.issue_id} unchanged (fingerprint match)")
                return 'unchanged'

        # Detailed comparison
        if self._is_meaningful_update(task, existing_task):
            return 'updated'

        # Update fingerprint cache
        if self._fingerprint_cache:
            self._fingerprint_cache.put(
                task.issue_id,
                self._compute_fingerprint(task),
                task.updated_at
            )

        return 'unchanged'

    def _is_meaningful_update(self, new_task: JiraTask, existing_task: ProcessedTask) -> bool:
        """
        Determine if the update is meaningful enough to warrant reprocessing.

        Uses optimized timestamp comparison and content fingerprinting.

        Args:
            new_task: New version of task
            existing_task: Existing version of task

        Returns:
            True if update is meaningful
        """
        # Primary check: updated_at timestamp
        new_updated = self._normalize_timestamp(new_task.updated_at)
        existing_updated = self._normalize_timestamp(existing_task.updated_at)

        # Check if timestamp difference is significant
        time_diff = abs((new_updated - existing_updated).total_seconds())

        if time_diff > self.TIMESTAMP_TOLERANCE_SECONDS:
            if new_updated > existing_updated:
                logger.debug(f"Timestamp difference detected for {new_task.issue_id} ({time_diff:.1f}s)")
                return True

        # Secondary checks: content changes using fingerprinting
        old_fingerprint = self._compute_content_fingerprint(
            existing_task.summary,
            existing_task.description,
            existing_task.cluster_label
        )
        new_fingerprint = self._compute_content_fingerprint(
            new_task.summary,
            new_task.description,
            new_task.cluster_label
        )

        if old_fingerprint != new_fingerprint:
            logger.debug(f"Content fingerprint changed for {new_task.issue_id}")
            return True

        return False

    def _normalize_timestamp(self, ts: datetime) -> datetime:
        """Normalize timestamp to UTC for comparison."""
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts

    def _compute_fingerprint(self, task) -> str:
        """
        Compute fast fingerprint for a task.

        Args:
            task: Task to compute fingerprint for

        Returns:
            Fingerprint string
        """
        return self._compute_content_fingerprint(
            task.summary,
            getattr(task, 'description', ''),
            getattr(task, 'cluster_label', None)
        )

    def _compute_content_fingerprint(
        self, summary: str, description: str, cluster_label: str = None
    ) -> str:
        """
        Compute content fingerprint for change detection.

        Args:
            summary: Task summary
            description: Task description
            cluster_label: Optional cluster label

        Returns:
            Fingerprint string
        """
        # Combine content fields
        content = f"{summary}|{description}|{cluster_label or ''}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def merge_updates(self,
                     new_processed_tasks: List[ProcessedTask],
                     updated_processed_pairs: List[Tuple[ProcessedTask, ProcessedTask]]) -> List[ProcessedTask]:
        """
        Merge new and updated tasks, preserving relevant metadata.

        Args:
            new_processed_tasks: Newly processed tasks
            updated_processed_pairs: Pairs of (new_processed, existing) tasks

        Returns:
            Combined list of all tasks with appropriate metadata
        """
        merged_tasks = []

        # Add new tasks as-is
        merged_tasks.extend(new_processed_tasks)

        # Process updated tasks with batch optimization
        for new_processed, existing in updated_processed_pairs:
            # Check if clustering should be preserved
            should_preserve = self._should_preserve_clustering(new_processed, existing)

            if should_preserve and existing.cluster_id is not None:
                # Keep existing clustering results
                merged_task = new_processed.model_copy()
                merged_task.cluster_id = existing.cluster_id
                merged_task.cluster_confidence = existing.cluster_confidence
                logger.debug(f"Preserved clustering for updated task {merged_task.issue_id}")
            else:
                # Mark for re-clustering
                merged_task = new_processed.model_copy()
                merged_task.cluster_id = None
                merged_task.cluster_confidence = None
                logger.debug(f"Marked task {merged_task.issue_id} for re-clustering")

            merged_tasks.append(merged_task)

        # Update internal cache
        for task in merged_tasks:
            self.existing_tasks[task.issue_id] = task

        logger.info(
            f"Merged {len(merged_tasks)} tasks "
            f"({len(new_processed_tasks)} new, {len(updated_processed_pairs)} updated)"
        )
        return merged_tasks

    def _should_preserve_clustering(self, new_task: ProcessedTask, existing_task: ProcessedTask) -> bool:
        """
        Determine if clustering should be preserved for updated task.

        Args:
            new_task: New version of processed task
            existing_task: Existing version with clustering info

        Returns:
            True if clustering should be preserved
        """
        # Calculate similarity using cached results
        cache_key = f"{new_task.issue_id}_similarity"

        if self._similarity_cache is not None and cache_key in self._similarity_cache:
            summary_similarity = self._similarity_cache[cache_key]
        else:
            summary_similarity = self._calculate_text_similarity_optimized(
                new_task.summary, existing_task.summary
            )
            if self._similarity_cache is not None:
                self._similarity_cache[cache_key] = summary_similarity

        # Check description similarity
        desc_similarity = self._calculate_text_similarity_optimized(
            new_task.cleaned_description if hasattr(new_task, 'cleaned_description') else new_task.description,
            existing_task.cleaned_description if hasattr(existing_task, 'cleaned_description') else existing_task.description
        )

        # Preserve clustering if both similarities are above threshold
        should_preserve = (
            summary_similarity >= self.CONTENT_SIMILARITY_THRESHOLD and
            desc_similarity >= self.CONTENT_SIMILARITY_THRESHOLD
        )

        if not should_preserve:
            logger.debug(
                f"Content significantly changed for {new_task.issue_id} "
                f"(summary_sim={summary_similarity:.2f}, desc_sim={desc_similarity:.2f})"
            )

        return should_preserve

    def _calculate_text_similarity_optimized(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity ratio with optimization.

        Uses cached character sets for faster computation.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity ratio between 0.0 and 1.0
        """
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0

        # Normalize texts
        text1_lower = text1.lower()
        text2_lower = text2.lower()

        # Use Jaccard similarity for sets
        set1 = set(text1_lower)
        set2 = set(text2_lower)

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    def get_update_statistics(self) -> Dict[str, int]:
        """
        Get statistics about the current update state.

        Returns:
            Dictionary with update statistics
        """
        return {
            **self._stats,
            "total_existing_tasks": len(self.existing_tasks),
            "new_tasks_since_last_update": len([
                t for t in self.existing_tasks.values()
                if hasattr(t, 'first_seen_in_update') and t.first_seen_in_update
            ])
        }

    @property
    def stats(self) -> dict:
        """Get handler statistics."""
        return self._stats

    def clear_cache(self) -> None:
        """Clear all caches."""
        if self._fingerprint_cache:
            self._fingerprint_cache.clear()
        if self._similarity_cache:
            with self._cache_lock:
                self._similarity_cache.clear()
        logger.info("Caches cleared")
