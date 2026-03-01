"""Incremental update handler for Jira task analysis."""

import logging
from typing import List, Dict, Tuple
from datetime import datetime
from app.models import JiraTask, ProcessedTask

logger = logging.getLogger(__name__)


class IncrementalUpdateHandler:
    """Handles incremental updates and determines if records are new or modified."""
    
    def __init__(self):
        self.existing_tasks: Dict[str, ProcessedTask] = {}
        
    def load_existing_data(self, existing_tasks: List[ProcessedTask]) -> None:
        """Load existing processed tasks for comparison."""
        self.existing_tasks = {task.issue_id: task for task in existing_tasks}
        logger.info(f"Loaded {len(self.existing_tasks)} existing tasks for incremental comparison")
    
    def categorize_updates(self, new_tasks: List[JiraTask]) -> Tuple[List[JiraTask], List[Tuple[JiraTask, ProcessedTask]]]:
        """
        Categorize incoming tasks as new or updated.
        
        Returns:
            Tuple of (new_tasks, updated_tasks_pairs)
            where updated_tasks_pairs contains (new_task, existing_task) tuples
        """
        new_tasks_list = []
        updated_tasks_pairs = []
        
        for task in new_tasks:
            if task.issue_id in self.existing_tasks:
                existing_task = self.existing_tasks[task.issue_id]
                
                # Compare timestamps to determine if it's a meaningful update
                if self._is_meaningful_update(task, existing_task):
                    updated_tasks_pairs.append((task, existing_task))
                    logger.debug(f"Task {task.issue_id} identified as updated")
                else:
                    logger.debug(f"Task {task.issue_id} unchanged, skipping")
            else:
                new_tasks_list.append(task)
                logger.debug(f"Task {task.issue_id} identified as new")
        
        logger.info(f"Categorized {len(new_tasks_list)} new tasks and {len(updated_tasks_pairs)} updated tasks")
        return new_tasks_list, updated_tasks_pairs
    
    def _is_meaningful_update(self, new_task: JiraTask, existing_task: ProcessedTask) -> bool:
        """Determine if the update is meaningful enough to warrant reprocessing."""
        # Primary check: updated_at timestamp
        # Ensure both datetimes have timezone info for comparison
        new_updated = new_task.updated_at
        existing_updated = existing_task.updated_at
        
        # If one has timezone and other doesn't, normalize
        if new_updated.tzinfo is None and existing_updated.tzinfo is not None:
            from datetime import timezone
            new_updated = new_updated.replace(tzinfo=timezone.utc)
        elif new_updated.tzinfo is not None and existing_updated.tzinfo is None:
            from datetime import timezone
            existing_updated = existing_updated.replace(tzinfo=timezone.utc)
        
        if new_updated > existing_updated:
            logger.debug(f"Timestamp difference detected for {new_task.issue_id}")
            return True
        
        # Secondary checks: content changes
        content_changed = (
            new_task.summary != existing_task.summary or
            new_task.description != existing_task.description or
            new_task.status != existing_task.status
        )
        
        if content_changed:
            logger.debug(f"Content changes detected for {new_task.issue_id}")
            return True
        
        return False
    
    def merge_updates(self, 
                     new_processed_tasks: List[ProcessedTask],
                     updated_processed_pairs: List[Tuple[ProcessedTask, ProcessedTask]]) -> List[ProcessedTask]:
        """
        Merge new and updated tasks, preserving relevant metadata from existing tasks.
        
        Args:
            new_processed_tasks: Newly processed tasks
            updated_processed_pairs: Pairs of (new_processed, existing) tasks
            
        Returns:
            Combined list of all tasks with appropriate metadata
        """
        merged_tasks = []
        
        # Add new tasks as-is
        merged_tasks.extend(new_processed_tasks)
        
        # Process updated tasks
        for new_processed, existing in updated_processed_pairs:
            # Preserve clustering information if content hasn't changed significantly
            should_preserve_clustering = not self._content_significantly_changed(new_processed, existing)
            
            if should_preserve_clustering and existing.cluster_id is not None:
                # Keep existing clustering results
                merged_task = new_processed.model_copy()
                merged_task.cluster_id = existing.cluster_id
                merged_task.cluster_confidence = existing.cluster_confidence
                logger.debug(f"Preserved clustering for updated task {merged_task.issue_id}")
            else:
                # Remove clustering info for reprocessing
                merged_task = new_processed.model_copy()
                merged_task.cluster_id = None
                merged_task.cluster_confidence = None
                logger.debug(f"Marked task {merged_task.issue_id} for re-clustering")
            
            merged_tasks.append(merged_task)
        
        # Update internal cache
        for task in merged_tasks:
            self.existing_tasks[task.issue_id] = task
        
        logger.info(f"Merged {len(merged_tasks)} tasks ({len(new_processed_tasks)} new, {len(updated_processed_pairs)} updated)")
        return merged_tasks
    
    def _content_significantly_changed(self, new_task: ProcessedTask, existing_task: ProcessedTask) -> bool:
        """Determine if content changes are significant enough to warrant re-clustering."""
        # Calculate similarity threshold (simple approach)
        summary_similarity = self._calculate_text_similarity(new_task.summary, existing_task.summary)
        desc_similarity = self._calculate_text_similarity(new_task.cleaned_description, existing_task.cleaned_description)
        
        # If either summary or description changed significantly, re-cluster
        return summary_similarity < 0.8 or desc_similarity < 0.8
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity ratio."""
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
            
        # Simple character-based similarity (for demonstration)
        set1, set2 = set(text1.lower()), set(text2.lower())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_update_statistics(self) -> Dict[str, int]:
        """Get statistics about the current update state."""
        return {
            "total_existing_tasks": len(self.existing_tasks),
            "new_tasks_since_last_update": len([t for t in self.existing_tasks.values() 
                                              if hasattr(t, 'first_seen_in_update') and t.first_seen_in_update])
        }