"""Unit tests for embedding and clustering concurrency features."""

import pytest
import asyncio
import threading
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import List
import sys

# Add project root to path
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])

from app.core.embedding import VectorEmbedder, ModelCache, get_model_cache_stats
from app.core.clustering import TaskClusterer, ClusteringOptimizer, _clustering_optimizer


class TestModelCache:
    """Test suite for ModelCache class."""

    @pytest.fixture
    def cache(self):
        """Create model cache instance."""
        return ModelCache(max_size=3)

    def test_cache_put_get(self, cache):
        """Test basic cache put and get."""
        mock_model = Mock()
        cache.put("test-model", mock_model)

        result = cache.get("test-model")
        assert result is mock_model

    def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        result = cache.get("nonexistent")
        assert result is None

    def test_cache_lru_eviction(self, cache):
        """Test LRU eviction when cache is full."""
        models = [Mock() for _ in range(4)]

        # Fill cache to max size
        cache.put("model1", models[0])
        cache.put("model2", models[1])
        cache.put("model3", models[2])

        # Add one more - should evict oldest
        cache.put("model4", models[3])

        # model1 should be evicted
        assert cache.get("model1") is None
        assert cache.get("model2") is models[1]
        assert cache.get("model3") is models[2]
        assert cache.get("model4") is models[3]

    def test_cache_lru_order_update(self, cache):
        """Test that accessing an item updates its LRU order."""
        models = [Mock() for _ in range(3)]

        cache.put("model1", models[0])
        cache.put("model2", models[1])
        cache.put("model3", models[2])

        # Access model1 - should move to end of LRU order
        cache.get("model1")

        # Add new model - should evict model2 (now oldest)
        cache.put("model4", Mock())

        assert cache.get("model1") is models[0]  # Should still exist
        assert cache.get("model2") is None  # Should be evicted

    def test_cache_clear(self, cache):
        """Test cache clear."""
        cache.put("model1", Mock())
        cache.put("model2", Mock())

        cache.clear()

        assert cache.get("model1") is None
        assert cache.get("model2") is None

    def test_cache_stats(self, cache):
        """Test cache statistics."""
        cache.put("model1", Mock())
        cache.get("model1")  # Hit
        cache.get("model1")  # Hit
        cache.get("nonexistent")  # Miss

        stats = cache.stats
        assert stats['size'] == 1
        assert stats['max_size'] == 3
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert float(stats['hit_rate'].rstrip('%')) > 0

    def test_cache_thread_safety(self, cache):
        """Test thread-safe cache access."""
        results = {'hits': 0, 'misses': 0, 'errors': 0}
        lock = threading.Lock()

        def access_cache(model_id):
            try:
                result = cache.get(model_id)
                with lock:
                    if result is not None:
                        results['hits'] += 1
                    else:
                        results['misses'] += 1
            except Exception as e:
                with lock:
                    results['errors'] += 1

        # Put some models
        for i in range(5):
            cache.put(f"model{i}", Mock())

        # Concurrent access
        threads = [threading.Thread(target=access_cache, args=(f"model{i % 5}",)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results['errors'] == 0
        assert results['hits'] + results['misses'] == 50


class TestVectorEmbedderConcurrency:
    """Test suite for VectorEmbedder concurrency features."""

    @pytest.fixture
    def embedder(self):
        """Create embedder instance."""
        return VectorEmbedder(model_name="text-embedding-3-small")

    @pytest.mark.asyncio
    async def test_concurrent_model_initialization(self, embedder):
        """Test concurrent model initialization doesn't cause race conditions."""
        async def initialize():
            await embedder.initialize_model()
            return embedder.model

        # Run multiple initializations concurrently
        results = await asyncio.gather(*[initialize() for _ in range(10)])

        # All should get the same model instance
        assert all(r is results[0] for r in results)
        assert embedder.model is not None

    @pytest.mark.asyncio
    async def test_concurrent_embedding_generation(self, embedder):
        """Test concurrent embedding generation."""
        await embedder.initialize_model()

        texts = [f"Test text {i}" for i in range(10)]

        # Generate embeddings concurrently
        tasks = [embedder.generate_embedding(text) for text in texts]
        results = await asyncio.gather(*tasks)

        assert len(results) == len(texts)
        assert all(isinstance(r, list) and len(r) > 0 for r in results)

    @pytest.mark.asyncio
    async def test_batch_vs_single_consistency(self, embedder):
        """Test that batch and single embedding generation produce consistent results."""
        await embedder.initialize_model()

        texts = ["Test text 1", "Test text 2", "Test text 3"]

        # Generate embeddings individually
        single_results = [await embedder.generate_embedding(text) for text in texts]

        # Generate embeddings in batch
        batch_results = await embedder.generate_embeddings_batch(texts)

        # Results should be identical
        assert len(single_results) == len(batch_results)
        for single, batch in zip(single_results, batch_results):
            assert single == batch

    @pytest.mark.asyncio
    async def test_concurrent_batch_processing(self, embedder):
        """Test concurrent batch processing."""
        await embedder.initialize_model()

        batches = [[f"Batch {b} text {i}" for i in range(5)] for b in range(5)]

        # Process batches concurrently
        tasks = [embedder.generate_embeddings_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks)

        assert len(results) == len(batches)
        for result, batch in zip(results, batches):
            assert len(result) == len(batch)

    @pytest.mark.asyncio
    async def test_embedding_cache_thread_safety(self, embedder):
        """Test embedding cache thread safety."""
        embedder._cache_embeddings = True
        await embedder.initialize_model()

        results = {'success': 0, 'errors': 0}
        lock = threading.Lock()

        def generate_and_cache(text):
            try:
                # Run async function in event loop
                loop = asyncio.new_event_loop()
                embedding = loop.run_until_complete(embedder.generate_embedding(text))
                loop.close()

                with lock:
                    results['success'] += 1
            except Exception as e:
                with lock:
                    results['errors'] += 1

        # Generate embeddings concurrently
        threads = [threading.Thread(target=generate_and_cache, args=(f"Text {i}",)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results['errors'] == 0
        assert results['success'] == 20

    @pytest.mark.asyncio
    async def test_dynamic_batch_size_calculation(self, embedder):
        """Test dynamic batch size calculation."""
        short_texts = ["Short"] * 100
        long_texts = ["This is a very long text " * 50] * 100

        short_batch_size = embedder._calculate_optimal_batch_size(short_texts)
        long_batch_size = embedder._calculate_optimal_batch_size(long_texts)

        # Longer texts should get smaller batch sizes
        assert short_batch_size >= long_batch_size

    @pytest.mark.asyncio
    async def test_embed_tasks_concurrent(self, embedder):
        """Test concurrent task embedding."""
        from app.models import ProcessedTask
        from datetime import datetime

        await embedder.initialize_model()

        tasks = [
            ProcessedTask(
                issue_id=f"TEST-{i}",
                summary=f"Task {i}",
                description=f"Description {i}",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                cluster_label="test",
                cleaned_description=f"Cleaned {i}",
                processed_at=datetime.now()
            )
            for i in range(10)
        ]

        # Embed tasks concurrently
        tasks1, tasks2 = tasks[:5], tasks[5:]
        results = await asyncio.gather(
            embedder.embed_tasks(tasks1),
            embedder.embed_tasks(tasks2)
        )

        assert len(results[0]) == 5
        assert len(results[1]) == 5
        assert all(t.embedding is not None for batch in results for t in batch)

    @pytest.mark.asyncio
    async def test_stats_tracking(self, embedder):
        """Test statistics tracking."""
        await embedder.initialize_model()

        # Generate some embeddings
        await embedder.generate_embedding("Test 1")
        await embedder.generate_embedding("Test 2")
        await embedder.generate_embeddings_batch(["Test 3", "Test 4"])

        stats = embedder.stats
        assert stats['embeddings_generated'] >= 4
        assert stats['model_loaded'] is True

    @pytest.mark.asyncio
    async def test_memory_efficient_mode(self, embedder):
        """Test memory efficient mode."""
        embedder._memory_efficient = True
        await embedder.initialize_model()

        # Generate many embeddings
        texts = [f"Memory test {i}" for i in range(50)]
        results = await embedder.generate_embeddings_batch(texts)

        assert len(results) == 50

    @pytest.mark.asyncio
    async def test_cache_disabled(self):
        """Test with caching disabled."""
        embedder = VectorEmbedder(cache_embeddings=False)
        await embedder.initialize_model()

        # Generate same embedding twice
        result1 = await embedder.generate_embedding("Test")
        result2 = await embedder.generate_embedding("Test")

        # Both should work but cache shouldn't be used
        assert result1 == result2
        assert embedder._embedding_cache is None


class TestClusteringConcurrency:
    """Test suite for clustering concurrency features."""

    @pytest.fixture
    def config(self):
        """Create clustering config."""
        from app.models import ClusteringConfig
        return ClusteringConfig(algorithm="kmeans", min_cluster_size=2)

    @pytest.fixture
    def clusterer(self, config):
        """Create clusterer instance."""
        return TaskClusterer(config, max_workers=2)

    @pytest.fixture
    def sample_tasks(self):
        """Create sample embedded tasks."""
        from app.models import ProcessedTask
        from datetime import datetime

        tasks = []
        for i in range(20):
            # Create embeddings with some structure
            cluster_id = i // 5
            base_value = cluster_id * 0.3
            embedding = [base_value + 0.01 * j for j in range(20)]

            task = ProcessedTask(
                issue_id=f"TEST-{i}",
                summary=f"Task {i}",
                description=f"Description {i}",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                cluster_label="test",
                cleaned_description=f"Cleaned {i}",
                processed_at=datetime.now(),
                embedding=embedding
            )
            tasks.append(task)
        return tasks

    @pytest.mark.asyncio
    async def test_concurrent_clustering(self, clusterer, sample_tasks):
        """Test concurrent clustering operations."""
        # Split tasks
        tasks1, tasks2 = sample_tasks[:10], sample_tasks[10:]

        # Run clustering concurrently
        results = await asyncio.gather(
            clusterer.cluster_tasks(tasks1),
            clusterer.cluster_tasks(tasks2),
            return_exceptions=True
        )

        # Both should complete successfully
        assert len(results) == 2
        for result in results:
            assert not isinstance(result, Exception)
            assert hasattr(result, 'clusters_found')

    @pytest.mark.asyncio
    async def test_parallel_silhouette_calculation(self, clusterer, sample_tasks):
        """Test parallel silhouette score calculation."""
        import numpy as np

        embeddings = np.array([task.embedding for task in sample_tasks], dtype=np.float32)
        k_values = list(range(2, 5))

        loop = asyncio.get_event_loop()
        scores = await clusterer._calculate_silhouette_scores_parallel(embeddings, k_values, loop)

        assert len(scores) == len(k_values)
        for k, score in scores:
            assert -1 <= score <= 1

    @pytest.mark.asyncio
    async def test_optimizer_caching(self, sample_tasks):
        """Test clustering optimizer caching."""
        from app.models import ClusteringConfig
        import numpy as np

        config = ClusteringConfig(algorithm="kmeans", min_cluster_size=2)
        clusterer = TaskClusterer(config, enable_optimization=True)

        embeddings = np.array([task.embedding for task in sample_tasks], dtype=np.float32)

        # First call - should calculate
        loop = asyncio.get_event_loop()
        k1 = await clusterer._find_optimal_clusters_kmeans_cached(embeddings)

        # Second call - should use cache
        k2 = await clusterer._find_optimal_clusters_kmeans_cached(embeddings)

        assert k1 == k2

    @pytest.mark.asyncio
    async def test_clusterer_stats(self, clusterer, sample_tasks):
        """Test clusterer statistics tracking."""
        await clusterer.cluster_tasks(sample_tasks[:10])
        await clusterer.cluster_tasks(sample_tasks[10:])

        stats = clusterer.stats
        assert stats['clustering_runs'] == 2
        assert stats['total_time'] > 0

    @pytest.mark.asyncio
    async def test_hdbscan_clustering(self, sample_tasks):
        """Test HDBSCAN clustering."""
        from app.models import ClusteringConfig

        config = ClusteringConfig(algorithm="hdbscan", min_cluster_size=3)
        clusterer = TaskClusterer(config)

        result = await clusterer.cluster_tasks(sample_tasks)

        assert result.total_tasks == len(sample_tasks)
        assert result.processing_time > 0

    @pytest.mark.asyncio
    async def test_kmeans_clustering(self, sample_tasks):
        """Test K-Means clustering."""
        from app.models import ClusteringConfig

        config = ClusteringConfig(algorithm="kmeans", min_cluster_size=2)
        clusterer = TaskClusterer(config)

        result = await clusterer.cluster_tasks(sample_tasks)

        assert result.total_tasks == len(sample_tasks)
        assert result.clusters_found >= 1


class TestClusteringOptimizer:
    """Test suite for ClusteringOptimizer class."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance."""
        return ClusteringOptimizer(max_cache_size=5)

    def test_silhouette_cache(self, optimizer):
        """Test silhouette score caching."""
        embeddings = __import__('numpy').np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        # Cache a score
        optimizer.cache_silhouette(embeddings, 3, 0.75)

        # Retrieve cached score
        score = optimizer.get_cached_silhouette(embeddings, 3)
        assert score == 0.75

        # Non-existent score returns None
        assert optimizer.get_cached_silhouette(embeddings, 5) is None

    def test_optimal_k_cache(self, optimizer):
        """Test optimal K caching."""
        embeddings = __import__('numpy').np.array([[0.1, 0.2], [0.3, 0.4]])

        # Cache optimal K
        optimizer.cache_optimal_k(embeddings, 4)

        # Retrieve cached K
        k = optimizer.get_cached_optimal_k(embeddings)
        assert k == 4

    def test_cache_eviction(self, optimizer):
        """Test cache eviction when full."""
        import numpy as np

        # Fill cache
        for i in range(6):
            embeddings = np.array([[float(i), float(i+1)]])
            optimizer.cache_silhouette(embeddings, 3, 0.5)

        # Cache should have max 5 entries
        assert len(optimizer._silhouette_cache) <= 5


class TestIntegrationConcurrency:
    """Integration tests for concurrency across modules."""

    @pytest.mark.asyncio
    async def test_full_pipeline_concurrent(self):
        """Test full embedding + clustering pipeline concurrently."""
        from app.models import ProcessedTask, ClusteringConfig
        from datetime import datetime

        # Create tasks
        tasks = [
            ProcessedTask(
                issue_id=f"INTEGRATION-{i}",
                summary=f"Integration task {i}",
                description=f"Description {i}",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                cluster_label="test",
                cleaned_description=f"Cleaned {i}",
                processed_at=datetime.now()
            )
            for i in range(20)
        ]

        embedder = VectorEmbedder(model_name="text-embedding-3-small")
        config = ClusteringConfig(algorithm="kmeans", min_cluster_size=3)
        clusterer = TaskClusterer(config)

        try:
            # Initialize
            await embedder.initialize_model()

            # Embed tasks
            embedded_tasks = await embedder.embed_tasks(tasks)
            assert all(t.embedding is not None for t in embedded_tasks)

            # Cluster tasks
            result = await clusterer.cluster_tasks(embedded_tasks)
            assert result.total_tasks == len(embedded_tasks)

        finally:
            await embedder.close()
            await clusterer.close()

    @pytest.mark.asyncio
    async def test_model_cache_across_instances(self):
        """Test model cache is shared across embedder instances."""
        embedder1 = VectorEmbedder(model_name="text-embedding-3-small")
        embedder2 = VectorEmbedder(model_name="text-embedding-3-small")

        try:
            # Initialize both
            await embedder1.initialize_model()
            await embedder2.initialize_model()

            # Both should have the same model (from cache)
            assert embedder1.model is embedder2.model

            # Check cache stats
            stats = get_model_cache_stats()
            assert stats['hits'] >= 1

        finally:
            await embedder1.close()
            await embedder2.close()


def teardown_module():
    """Clean up after tests."""
    # Clear global caches
    from app.core.embedding import _model_cache
    _model_cache.clear()
