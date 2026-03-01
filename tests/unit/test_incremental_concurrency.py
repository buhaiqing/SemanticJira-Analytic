"""Unit tests for incremental processing concurrency features."""

import pytest
import asyncio
import threading
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock
from typing import List
import sys

# Add project root to path
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])

from app.core.incremental import IncrementalUpdateHandler, FingerprintCache
from app.models import JiraTask, ProcessedTask


class TestFingerprintCache:
    """Test suite for FingerprintCache class."""

    @pytest.fixture
    def cache(self):
        """Create fingerprint cache instance."""
        return FingerprintCache(max_size=5)

    def test_cache_put_get(self, cache):
        """Test basic cache put and get."""
        timestamp = datetime.now()
        cache.put("task-001", "abc123", timestamp)

        fingerprint, ts = cache.get("task-001")
        assert fingerprint == "abc123"
        assert ts == timestamp

    def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        fingerprint, timestamp = cache.get("nonexistent")
        assert fingerprint is None
        assert timestamp is None

    def test_cache_lru_eviction(self, cache):
        """Test LRU eviction when cache is full."""
        timestamps = [datetime.now() + timedelta(seconds=i) for i in range(6)]

        # Fill cache
        for i in range(5):
            cache.put(f"task-{i}", f"fp-{i}", timestamps[i])

        # Add one more - should evict oldest
        cache.put("task-5", "fp-5", timestamps[5])

        # task-0 should be evicted
        fingerprint, _ = cache.get("task-0")
        assert fingerprint is None

        # Others should exist
        for i in range(1, 6):
            fingerprint, _ = cache.get(f"task-{i}")
            assert fingerprint == f"fp-{i}"

    def test_cache_lru_order_update(self, cache):
        """Test that accessing an item updates its LRU order."""
        timestamps = [datetime.now() + timedelta(seconds=i) for i in range(5)]

        # Fill cache
        for i in range(5):
            cache.put(f"task-{i}", f"fp-{i}", timestamps[i])

        # Access task-0 - should move to end of LRU order
        cache.get("task-0")

        # Add new task - should evict task-1 (now oldest)
        cache.put("task-5", "fp-5", datetime.now())

        # task-0 should still exist
        fingerprint, _ = cache.get("task-0")
        assert fingerprint == "fp-0"

        # task-1 should be evicted
        fingerprint, _ = cache.get("task-1")
        assert fingerprint is None

    def test_cache_clear(self, cache):
        """Test cache clear."""
        cache.put("task-001", "abc123", datetime.now())
        cache.put("task-002", "def456", datetime.now())

        cache.clear()

        fingerprint, _ = cache.get("task-001")
        assert fingerprint is None

    def test_cache_thread_safety(self, cache):
        """Test thread-safe cache access."""
        results = {'hits': 0, 'misses': 0, 'errors': []}
        lock = threading.Lock()

        def access_cache(task_id):
            try:
                fingerprint, _ = cache.get(task_id)
                with lock:
                    if fingerprint is not None:
                        results['hits'] += 1
                    else:
                        results['misses'] += 1
            except Exception as e:
                with lock:
                    results['errors'].append(e)

        # Pre-populate cache
        for i in range(10):
            cache.put(f"task-{i}", f"fp-{i}", datetime.now())

        # Concurrent access
        threads = [threading.Thread(target=access_cache, args=(f"task-{i % 10}",)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results['errors']) == 0
        assert results['hits'] + results['misses'] == 50


class TestIncrementalUpdateHandlerConcurrency:
    """Test suite for IncrementalUpdateHandler concurrency features."""

    @pytest.fixture
    def handler(self):
        """Create incremental update handler."""
        return IncrementalUpdateHandler(enable_optimization=True)

    @pytest.fixture
    def sample_tasks(self):
        """Create sample Jira tasks."""
        base_time = datetime.now()
        return [
            JiraTask(
                issue_id=f"TASK-{i:03d}",
                summary=f"Task summary {i}",
                description=f"Task description {i}",
                created_at=base_time - timedelta(days=i),
                updated_at=base_time,
                cluster_label="test"
            )
            for i in range(20)
        ]

    @pytest.fixture
    def sample_processed_tasks(self, sample_tasks):
        """Create sample processed tasks."""
        return [
            ProcessedTask(
                issue_id=task.issue_id,
                summary=task.summary,
                description=task.description,
                created_at=task.created_at,
                updated_at=task.updated_at,
                cluster_label=task.cluster_label,
                cleaned_description=f"Cleaned: {task.description}",
                processed_at=datetime.now()
            )
            for task in sample_tasks
        ]

    def test_categorize_updates_concurrent(self, handler, sample_tasks):
        """Test concurrent categorization of updates."""
        # Load existing data
        processed_tasks = [
            ProcessedTask(
                issue_id=task.issue_id,
                summary=task.summary,
                description=task.description,
                created_at=task.created_at,
                updated_at=task.updated_at,
                cluster_label=task.cluster_label,
                cleaned_description=f"Cleaned: {task.description}",
                processed_at=datetime.now()
            )
            for task in sample_tasks[:10]
        ]
        handler.load_existing_data(processed_tasks)

        results = {'new': 0, 'updated': 0, 'unchanged': 0}
        lock = threading.Lock()

        def categorize_batch(batch):
            new, updated = handler.categorize_updates(batch)
            with lock:
                results['new'] += len(new)
                results['updated'] += len(updated)
                results['unchanged'] += len(batch) - len(new) - len(updated)

        # Split tasks into batches for concurrent processing
        batch_size = 5
        batches = [sample_tasks[i:i+batch_size] for i in range(0, len(sample_tasks), batch_size)]

        threads = [threading.Thread(target=categorize_batch, args=(batch,)) for batch in batches]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # First 10 tasks should be unchanged, last 10 should be new
        assert results['new'] == 10
        assert results['unchanged'] == 10

    def test_load_existing_data_thread_safety(self, handler, sample_processed_tasks):
        """Test thread-safe loading of existing data."""
        results = {'success': 0, 'errors': []}
        lock = threading.Lock()

        def load_data():
            try:
                handler.load_existing_data(sample_processed_tasks)
                with lock:
                    results['success'] += 1
            except Exception as e:
                with lock:
                    results['errors'].append(e)

        threads = [threading.Thread(target=load_data) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results['errors']) == 0
        assert results['success'] == 10

    def test_fingerprint_computation_concurrent(self, handler, sample_tasks):
        """Test concurrent fingerprint computation."""
        results = {'fingerprints': [], 'errors': []}
        lock = threading.Lock()

        def compute_fingerprint(task):
            try:
                fp = handler._compute_fingerprint(task)
                with lock:
                    results['fingerprints'].append((task.issue_id, fp))
            except Exception as e:
                with lock:
                    results['errors'].append(e)

        threads = [threading.Thread(target=compute_fingerprint, args=(task,)) for task in sample_tasks]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results['errors']) == 0
        assert len(results['fingerprints']) == len(sample_tasks)

        # Verify all fingerprints are unique for different tasks
        fingerprints = [fp for _, fp in results['fingerprints']]
        assert len(set(fingerprints)) == len(sample_tasks)

    def test_merge_updates_concurrent(self, handler, sample_processed_tasks):
        """Test concurrent merge operations."""
        new_tasks = sample_processed_tasks[:5]
        updated_pairs = [
            (sample_processed_tasks[i], sample_processed_tasks[i])
            for i in range(5, 10)
        ]

        results = {'merged': [], 'errors': []}
        lock = threading.Lock()

        def merge_batch(new_list, updated_list):
            try:
                merged = handler.merge_updates(new_list, updated_list)
                with lock:
                    results['merged'].extend(merged)
            except Exception as e:
                with lock:
                    results['errors'].append(e)

        # Multiple concurrent merges
        threads = [
            threading.Thread(target=merge_batch, args=(new_tasks, updated_pairs)),
            threading.Thread(target=merge_batch, args=(new_tasks, [])),
            threading.Thread(target=merge_batch, args=([], updated_pairs)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results['errors']) == 0
        # Should have merged all batches
        assert len(results['merged']) == (10 + 5 + 5)

    def test_categorize_updates_with_modifications(self, handler, sample_tasks):
        """Test categorization with various modifications."""
        # Load existing data
        base_time = datetime.now()
        existing_tasks = [
            ProcessedTask(
                issue_id=f"TASK-{i:03d}",
                summary=f"Task summary {i}",
                description=f"Task description {i}",
                created_at=base_time - timedelta(days=i),
                updated_at=base_time,
                cluster_label="test",
                cleaned_description=f"Cleaned: Task description {i}",
                processed_at=base_time
            )
            for i in range(10)
        ]
        handler.load_existing_data(existing_tasks)

        # Create modified versions
        modified_tasks = []
        for i in range(10):
            if i < 3:
                # Unchanged
                modified_tasks.append(sample_tasks[i])
            elif i < 6:
                # Modified summary
                task = sample_tasks[i].model_copy()
                task.summary = f"Modified summary {i}"
                modified_tasks.append(task)
            else:
                # Modified description
                task = sample_tasks[i].model_copy()
                task.description = f"Modified description {i}"
                modified_tasks.append(task)

        new, updated = handler.categorize_updates(modified_tasks)

        # 0-2: unchanged, 3-5: updated (summary), 6-9: updated (description)
        assert len(new) == 0
        assert len(updated) == 7  # 3-9

    def test_stats_tracking(self, handler, sample_tasks):
        """Test statistics tracking."""
        # Load existing data
        existing_tasks = [
            ProcessedTask(
                issue_id=f"TASK-{i:03d}",
                summary=f"Task summary {i}",
                description=f"Task description {i}",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                cluster_label="test",
                cleaned_description=f"Cleaned: Task description {i}",
                processed_at=datetime.now()
            )
            for i in range(10)
        ]
        handler.load_existing_data(existing_tasks)

        # Categorize some updates
        handler.categorize_updates(sample_tasks[:15])

        stats = handler.stats
        assert stats['comparisons_made'] > 0
        assert stats['new_tasks'] + stats['updated_tasks'] + stats['unchanged_tasks'] == 15

    def test_cache_clear(self, handler):
        """Test cache clearing."""
        # Populate caches
        handler._fingerprint_cache.put("task-001", "abc123", datetime.now())
        if handler._similarity_cache is not None:
            handler._similarity_cache["test_key"] = 0.95

        handler.clear_cache()

        # Verify caches are cleared
        fingerprint, _ = handler._fingerprint_cache.get("task-001")
        assert fingerprint is None

        if handler._similarity_cache is not None:
            assert "test_key" not in handler._similarity_cache

    def test_incremental_update_handler_disabled_optimization(self):
        """Test handler with optimizations disabled."""
        handler = IncrementalUpdateHandler(enable_optimization=False)

        assert handler._fingerprint_cache is None
        assert handler._similarity_cache is None

        # Should still work without optimizations
        existing_tasks = [
            ProcessedTask(
                issue_id="TASK-001",
                summary="Test",
                description="Test description",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                cluster_label="test",
                cleaned_description="Cleaned",
                processed_at=datetime.now()
            )
        ]
        handler.load_existing_data(existing_tasks)
        assert len(handler.existing_tasks) == 1


class TestTimestampNormalization:
    """Test timestamp normalization edge cases."""

    @pytest.fixture
    def handler(self):
        """Create handler for timestamp tests."""
        return IncrementalUpdateHandler()

    def test_normalize_timestamp_with_tz(self, handler):
        """Test normalizing timestamp with timezone."""
        ts = datetime.now(timezone.utc)
        result = handler._normalize_timestamp(ts)
        assert result.tzinfo is not None
        assert result == ts

    def test_normalize_timestamp_without_tz(self, handler):
        """Test normalizing timestamp without timezone."""
        ts = datetime.now()
        assert ts.tzinfo is None

        result = handler._normalize_timestamp(ts)
        assert result.tzinfo is timezone.utc

    def test_timestamp_tolerance(self, handler):
        """Test timestamp tolerance for small differences."""
        base_time = datetime.now(timezone.utc)

        # Create tasks with small time difference (within tolerance)
        task1 = JiraTask(
            issue_id="TASK-001",
            summary="Test",
            description="Test",
            created_at=base_time,
            updated_at=base_time,
            cluster_label="test"
        )

        existing1 = ProcessedTask(
            issue_id="TASK-001",
            summary="Test",
            description="Test",
            created_at=base_time,
            updated_at=base_time + timedelta(seconds=0.5),  # Within tolerance
            cluster_label="test",
            cleaned_description="Cleaned",
            processed_at=base_time
        )

        # Should be considered unchanged due to tolerance
        handler.load_existing_data([existing1])
        new, updated = handler.categorize_updates([task1])

        # Should be unchanged (within tolerance and content same)
        assert len(new) == 0
        assert len(updated) == 0


class TestContentFingerprint:
    """Test content fingerprint computation."""

    @pytest.fixture
    def handler(self):
        """Create handler for fingerprint tests."""
        return IncrementalUpdateHandler()

    def test_fingerprint_same_content(self, handler):
        """Test fingerprint for same content."""
        fp1 = handler._compute_content_fingerprint("Summary", "Description", "label")
        fp2 = handler._compute_content_fingerprint("Summary", "Description", "label")

        assert fp1 == fp2

    def test_fingerprint_different_content(self, handler):
        """Test fingerprint for different content."""
        fp1 = handler._compute_content_fingerprint("Summary A", "Description A", "label")
        fp2 = handler._compute_content_fingerprint("Summary B", "Description B", "label")

        assert fp1 != fp2

    def test_fingerprint_empty_fields(self, handler):
        """Test fingerprint with empty fields."""
        fp1 = handler._compute_content_fingerprint("", "", None)
        fp2 = handler._compute_content_fingerprint("", "", "")

        assert fp1 == fp2

    def test_fingerprint_consistency(self, handler):
        """Test fingerprint consistency across multiple calls."""
        fingerprints = [
            handler._compute_content_fingerprint(f"Summary {i}", f"Description {i}", "label")
            for i in range(100)
        ]

        # All fingerprints should be unique
        assert len(set(fingerprints)) == 100


class TestSimilarityCalculation:
    """Test similarity calculation."""

    @pytest.fixture
    def handler(self):
        """Create handler for similarity tests."""
        return IncrementalUpdateHandler()

    def test_similarity_identical_texts(self, handler):
        """Test similarity for identical texts."""
        similarity = handler._calculate_text_similarity_optimized("Hello World", "Hello World")
        assert similarity == 1.0

    def test_similarity_different_texts(self, handler):
        """Test similarity for completely different texts."""
        similarity = handler._calculate_text_similarity_optimized("ABC", "XYZ")
        assert similarity < 0.5

    def test_similarity_empty_texts(self, handler):
        """Test similarity for empty texts."""
        assert handler._calculate_text_similarity_optimized("", "") == 1.0
        assert handler._calculate_text_similarity_optimized("Hello", "") == 0.0
        assert handler._calculate_text_similarity_optimized("", "World") == 0.0

    def test_similarity_partial_match(self, handler):
        """Test similarity for partially matching texts."""
        similarity = handler._calculate_text_similarity_optimized("Hello World", "Hello Universe")
        assert 0 < similarity < 1.0


def teardown_module():
    """Clean up after tests."""
    pass
