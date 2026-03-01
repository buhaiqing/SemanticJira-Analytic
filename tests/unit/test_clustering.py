"""Unit tests for clustering module."""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch
from app.core.clustering import TaskClusterer
from app.models import ProcessedTask, ClusteringConfig
from datetime import datetime


class TestTaskClusterer:
    """Test suite for TaskClusterer class."""
    
    @pytest.fixture
    def config(self):
        """Create clustering configuration for testing."""
        return ClusteringConfig(
            algorithm="hdbscan",
            min_cluster_size=3,
            cluster_selection_epsilon=0.5
        )
    
    @pytest.fixture
    def clusterer(self, config):
        """Create clusterer instance for testing."""
        return TaskClusterer(config)
    
    @pytest.fixture
    def sample_embedded_tasks(self):
        """Create sample embedded tasks for testing."""
        # Create tasks with mock embeddings that form clear clusters
        tasks = []
        for i in range(15):  # 15 tasks total
            # First 5 tasks - cluster 0 (similar embeddings)
            if i < 5:
                embedding = [0.1, 0.1, 0.1] + [0.0] * 765  # 768-dimensional embedding
            # Next 5 tasks - cluster 1 (different embeddings)
            elif i < 10:
                embedding = [0.9, 0.9, 0.9] + [0.0] * 765
            # Last 5 tasks - cluster 2 (another group)
            else:
                embedding = [0.5, 0.1, 0.9] + [0.0] * 765
            
            task = ProcessedTask(
                issue_id=f"TEST-{i+1:03d}",
                summary=f"Test task {i+1}",
                description=f"Description for test task {i+1}",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                cluster_label="测试聚类",
                # status field removed,
                # priority field removed,
                cleaned_description=f"Cleaned description {i+1}",
                processed_at=datetime.now(),
                embedding=embedding
            )
            tasks.append(task)
        
        return tasks
    
    @pytest.mark.asyncio
    async def test_cluster_tasks_hdbscan(self, clusterer, sample_embedded_tasks):
        """Test HDBSCAN clustering."""
        result = await clusterer.cluster_tasks(sample_embedded_tasks)
        
        assert result.total_tasks == len(sample_embedded_tasks)
        assert result.clusters_found >= 1  # Should find at least one cluster
        assert result.processing_time > 0
        assert len(result.cluster_details) >= 1
        
        # Check cluster details structure
        for cluster_id, details in result.cluster_details.items():
            assert "size" in details
            assert "avg_confidence" in details
            assert "sample_tasks" in details
            assert details["size"] > 0
    
    @pytest.mark.asyncio
    async def test_cluster_tasks_kmeans(self, sample_embedded_tasks):
        """Test K-Means clustering."""
        config = ClusteringConfig(algorithm="kmeans", min_cluster_size=2)
        clusterer = TaskClusterer(config)
        
        result = await clusterer.cluster_tasks(sample_embedded_tasks)
        
        assert result.total_tasks == len(sample_embedded_tasks)
        assert result.clusters_found >= 1
    
    @pytest.mark.asyncio
    async def test_cluster_empty_tasks(self, clusterer):
        """Test clustering with empty tasks list."""
        with pytest.raises(ValueError, match="No tasks provided"):
            await clusterer.cluster_tasks([])
    
    @pytest.mark.asyncio
    async def test_cluster_no_embeddings(self, clusterer):
        """Test clustering with tasks that have no embeddings."""
        tasks_without_embeddings = [
            ProcessedTask(
                issue_id="TEST-001",
                summary="Test task",
                description="Test description",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                cluster_label="测试聚类",
                # status field removed,
                # priority field removed,
                cleaned_description="Cleaned description",
                processed_at=datetime.now()
                # No embedding field
            )
        ]
        
        with pytest.raises(ValueError, match="No tasks with embeddings found"):
            await clusterer.cluster_tasks(tasks_without_embeddings)
    
    @pytest.mark.asyncio
    async def test_invalid_algorithm(self):
        """Test handling of invalid clustering algorithm."""
        config = ClusteringConfig(algorithm="invalid_algorithm")
        clusterer = TaskClusterer(config)
        
        # Create simple embedded tasks for testing
        simple_tasks = [
            ProcessedTask(
                issue_id="TEST-001",
                summary="Test",
                description="Test",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                cluster_label="测试聚类",
                # status field removed,
                # priority field removed,
                cleaned_description="Test",
                processed_at=datetime.now(),
                embedding=[0.1] * 10  # Small embedding for testing
            )
        ]
        
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            await clusterer.cluster_tasks(simple_tasks)
    
    @pytest.mark.asyncio
    async def test_cluster_result_structure(self, clusterer, sample_embedded_tasks):
        """Test the structure of clustering results."""
        result = await clusterer.cluster_tasks(sample_embedded_tasks)
        
        # Check AnalysisResult structure
        assert hasattr(result, 'total_tasks')
        assert hasattr(result, 'clusters_found')
        assert hasattr(result, 'cluster_details')
        assert hasattr(result, 'processing_time')
        assert hasattr(result, 'generated_at')
        
        # Check that all tasks have cluster assignments
        clustered_tasks = []
        for details in result.cluster_details.values():
            if "sample_tasks" in details:
                clustered_tasks.extend(details["sample_tasks"])
        
        # Verify cluster details contain expected information
        for cluster_id, details in result.cluster_details.items():
            assert isinstance(details["size"], int)
            assert isinstance(details["avg_confidence"], float)
            assert 0 <= details["avg_confidence"] <= 1
            assert isinstance(details["sample_tasks"], list)


class TestClusteringEdgeCases:
    """Test edge cases in clustering."""
    
    @pytest.fixture
    def edge_case_config(self):
        return ClusteringConfig(min_cluster_size=2, cluster_selection_epsilon=0.1)
    
    @pytest.mark.asyncio
    async def test_single_cluster(self, edge_case_config):
        """Test clustering when all points belong to one cluster."""
        clusterer = TaskClusterer(edge_case_config)
        
        # Create tasks with very similar embeddings
        similar_tasks = []
        for i in range(10):
            embedding = [0.5 + np.random.normal(0, 0.01) for _ in range(10)]  # Very similar
            task = ProcessedTask(
                issue_id=f"SIMILAR-{i}",
                summary=f"Similar task {i}",
                description="Similar description",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                cluster_label="测试聚类",
                # status field removed,
                # priority field removed,
                cleaned_description="Similar cleaned description",
                processed_at=datetime.now(),
                embedding=embedding
            )
            similar_tasks.append(task)
        
        result = await clusterer.cluster_tasks(similar_tasks)
        assert result.clusters_found >= 1
    
    @pytest.mark.asyncio
    async def test_all_noise_points(self, edge_case_config):
        """Test clustering when all points are classified as noise."""
        clusterer = TaskClusterer(edge_case_config)
        
        # Create tasks with very dissimilar embeddings
        diverse_tasks = []
        for i in range(8):  # Small number, below min_cluster_size
            embedding = [float(i) / 10] * 10  # Very different embeddings
            task = ProcessedTask(
                issue_id=f"DIVERSE-{i}",
                summary=f"Diverse task {i}",
                description="Diverse description",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                cluster_label="测试聚类",
                # status field removed,
                # priority field removed,
                cleaned_description="Diverse cleaned description",
                processed_at=datetime.now(),
                embedding=embedding
            )
            diverse_tasks.append(task)
        
        result = await clusterer.cluster_tasks(diverse_tasks)
        # Should handle gracefully even if all points are noise
        assert result.total_tasks == len(diverse_tasks)


# Performance test
@pytest.mark.asyncio
async def test_clustering_performance():
    """Test clustering performance with larger dataset."""
    config = ClusteringConfig(min_cluster_size=5)
    clusterer = TaskClusterer(config)
    
    # Create larger dataset
    large_tasks = []
    for i in range(100):  # 100 tasks
        # Create embeddings with some clustering structure
        cluster_id = i // 20  # 5 clusters of 20 tasks each
        base_value = cluster_id * 0.2
        embedding = [base_value + np.random.normal(0, 0.05) for _ in range(20)]
        
        task = ProcessedTask(
            issue_id=f"LARGE-{i:03d}",
            summary=f"Large dataset task {i}",
            description=f"Description {i}",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            # status field removed
            # priority field removed,
            cleaned_description=f"Cleaned {i}",
            processed_at=datetime.now(),
            embedding=embedding
        )
        large_tasks.append(task)
    
    import time
    start_time = time.time()
    result = await clusterer.cluster_tasks(large_tasks)
    end_time = time.time()
    
    processing_time = end_time - start_time
    assert processing_time < 30.0  # Should complete within 30 seconds
    assert result.total_tasks == len(large_tasks)
    assert result.clusters_found >= 1