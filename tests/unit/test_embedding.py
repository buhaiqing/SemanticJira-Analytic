"""Unit tests for vector embedding module."""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch
from app.core.embedding import VectorEmbedder
from app.models import ProcessedTask
from datetime import datetime


class TestVectorEmbedder:
    """Test suite for VectorEmbedder class."""
    
    @pytest.fixture
    def embedder(self):
        """Create embedder instance for testing."""
        return VectorEmbedder(model_name="text-embedding-3-small")  # Use smaller model for testing
    
    @pytest.fixture
    def sample_tasks(self):
        """Create sample processed tasks for testing."""
        return [
            ProcessedTask(
                issue_id="TEST-001",
                summary="Fix login authentication bug",
                description="User authentication failing after password reset",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                cluster_label="test",
                cleaned_description="User authentication failing after password reset",
                processed_at=datetime.now()
            ),
            ProcessedTask(
                issue_id="TEST-002",
                summary="Add user profile picture upload",
                description="Implement profile picture upload feature",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                cluster_label="test",
                cleaned_description="Implement profile picture upload feature",
                processed_at=datetime.now()
            )
        ]
    
    @pytest.mark.asyncio
    async def test_initialize_model(self, embedder):
        """Test model initialization."""
        await embedder.initialize_model()
        assert embedder.model is not None
    
    @pytest.mark.asyncio
    async def test_generate_embedding(self, embedder):
        """Test single embedding generation."""
        await embedder.initialize_model()
        text = "This is a test sentence for embedding"
        embedding = await embedder.generate_embedding(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_batch(self, embedder):
        """Test batch embedding generation."""
        await embedder.initialize_model()
        texts = [
            "First test sentence",
            "Second test sentence",
            "Third test sentence"
        ]
        
        embeddings = await embedder.generate_embeddings_batch(texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert all(isinstance(x, float) for x in embedding)
    
    def test_prepare_text_for_embedding(self, embedder, sample_tasks):
        """Test text preparation for embedding."""
        task = sample_tasks[0]
        prepared_text = embedder.prepare_text_for_embedding(task)
        
        assert isinstance(prepared_text, str)
        assert task.summary in prepared_text
        assert task.cleaned_description in prepared_text
        assert " | " in prepared_text  # Separator should be present
    
    @pytest.mark.asyncio
    async def test_embed_tasks(self, embedder, sample_tasks):
        """Test embedding of multiple tasks."""
        await embedder.initialize_model()
        embedded_tasks = await embedder.embed_tasks(sample_tasks)
        
        assert len(embedded_tasks) == len(sample_tasks)
        for task in embedded_tasks:
            assert isinstance(task, ProcessedTask)
            assert task.embedding is not None
            assert isinstance(task.embedding, list)
            assert len(task.embedding) > 0
    
    @pytest.mark.asyncio
    async def test_embed_empty_tasks_list(self, embedder):
        """Test handling of empty tasks list."""
        result = await embedder.embed_tasks([])
        assert result == []
    
    @pytest.mark.asyncio
    async def test_get_embedding_dimension(self, embedder):
        """Test getting embedding dimension."""
        await embedder.initialize_model()
        dimension = embedder.get_embedding_dimension()
        
        assert isinstance(dimension, int)
        assert dimension > 0
    
    @pytest.mark.asyncio
    async def test_close_method(self, embedder):
        """Test resource cleanup."""
        await embedder.initialize_model()
        assert embedder.model is not None
        
        await embedder.close()
        # Model should be cleared after close
        assert embedder.model is None
    
    @pytest.mark.asyncio
    async def test_concurrent_embedding_generation(self, embedder, sample_tasks):
        """Test concurrent embedding generation."""
        await embedder.initialize_model()
        
        # Generate embeddings for the same tasks concurrently
        tasks = [embedder.embed_tasks(sample_tasks) for _ in range(3)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for result in results:
            assert len(result) == len(sample_tasks)
            for task in result:
                assert task.embedding is not None


class TestVectorEmbedderErrorHandling:
    """Test error handling in VectorEmbedder."""
    
    @pytest.fixture
    def embedder(self):
        return VectorEmbedder(model_name="invalid-model")
    
    @pytest.mark.asyncio
    async def test_initialize_model_failure(self, embedder):
        """Test handling of model initialization failure."""
        with pytest.raises(Exception):
            await embedder.initialize_model()
    
    @pytest.mark.asyncio
    async def test_generate_embedding_without_initialization(self, embedder):
        """Test generating embedding without model initialization."""
        with pytest.raises(RuntimeError, match="Model not initialized"):
            await embedder.generate_embedding("test text")
    
    @pytest.mark.asyncio
    async def test_get_dimension_without_initialization(self, embedder):
        """Test getting dimension without model initialization."""
        with pytest.raises(RuntimeError, match="Model not initialized"):
            embedder.get_embedding_dimension()


# Integration test
@pytest.mark.asyncio
async def test_full_embedding_pipeline():
    """Test complete embedding pipeline from text to vectors."""
    # This test uses a real (small) model
    embedder = VectorEmbedder(model_name="text-embedding-3-small")
    
    try:
        # Initialize
        await embedder.initialize_model()
        
        # Test single embedding
        single_embedding = await embedder.generate_embedding("Machine learning is fascinating")
        assert len(single_embedding) > 0
        
        # Test batch embedding
        batch_texts = [
            "Natural language processing",
            "Computer vision applications",
            "Deep learning architectures"
        ]
        batch_embeddings = await embedder.generate_embeddings_batch(batch_texts)
        assert len(batch_embeddings) == len(batch_texts)
        
        # Verify embeddings are different for different texts
        assert batch_embeddings[0] != batch_embeddings[1]
        assert batch_embeddings[1] != batch_embeddings[2]
        
    finally:
        await embedder.close()