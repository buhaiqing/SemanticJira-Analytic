"""Vector embedding generation for Jira task analysis."""

from typing import List, Optional
import logging
from sentence_transformers import SentenceTransformer
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.models import ProcessedTask

logger = logging.getLogger(__name__)


class VectorEmbedder:
    """Handles vector embedding generation for text data."""
    
    # 类级别模型缓存
    _model_cache = {}
    _cache_lock = asyncio.Lock()
    
    def __init__(self, model_name: str = "BGE-M3"):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self._executor = ThreadPoolExecutor(max_workers=4)
        
    async def initialize_model(self) -> None:
        """Initialize the embedding model asynchronously with caching."""
        if self.model is not None:
            return
            
        # 检查缓存
        cache_key = f"{self.model_name}_model"
        async with self._cache_lock:
            if cache_key in self._model_cache:
                self.model = self._model_cache[cache_key]
                logger.info(f"Using cached model: {self.model_name}")
                return
            
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Map model names to actual model identifiers
            model_mapping = {
                "BGE-M3": "BAAI/bge-m3",
                "text-embedding-3-small": "sentence-transformers/all-MiniLM-L6-v2"
            }
            
            model_id = model_mapping.get(self.model_name, "BAAI/bge-m3")
            
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                self._executor,
                lambda: SentenceTransformer(model_id)
            )
            
            # 存储到缓存
            async with self._cache_lock:
                self._model_cache[cache_key] = self.model
            
            logger.info(f"Successfully loaded model: {model_id}")
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    def _generate_embedding_sync(self, text: str) -> List[float]:
        """Generate embedding synchronously."""
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text asynchronously."""
        if self.model is None:
            await self.initialize_model()
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._generate_embedding_sync,
            text
        )
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch."""
        if self.model is None:
            await self.initialize_model()
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            
            # 动态batch_size调整
            avg_length = sum(len(text) for text in texts) / len(texts)
            dynamic_batch_size = min(64, max(8, int(1000 / max(avg_length, 1))))
            logger.info(f"Using dynamic batch size: {dynamic_batch_size} (avg text length: {avg_length:.1f})")
            
            # Use model's built-in batch processing
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                self._executor,
                lambda: self.model.encode(texts, convert_to_numpy=True, batch_size=dynamic_batch_size)
            )
            
            result = [emb.tolist() for emb in embeddings]
            logger.info(f"Generated {len(result)} embeddings")
            return result
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    def prepare_text_for_embedding(self, task: ProcessedTask) -> str:
        """Prepare text content for embedding generation."""
        # Combine summary and cleaned description
        content_parts = [task.summary]
        
        if task.cleaned_description:
            content_parts.append(task.cleaned_description)
        
        # Join with separator
        text = " | ".join(content_parts)
        return text.strip()
    
    async def embed_tasks(self, tasks: List[ProcessedTask]) -> List[ProcessedTask]:
        """Generate embeddings for a list of processed tasks."""
        if not tasks:
            return tasks
        
        try:
            logger.info(f"Embedding {len(tasks)} tasks")
            
            # Prepare texts for embedding
            texts = [self.prepare_text_for_embedding(task) for task in tasks]
            
            # Generate embeddings in batch
            embeddings = await self.generate_embeddings_batch(texts)
            
            # Attach embeddings to tasks
            embedded_tasks = []
            for task, embedding in zip(tasks, embeddings):
                embedded_task = task.model_copy()
                embedded_task.embedding = embedding
                embedded_tasks.append(embedded_task)
            
            logger.info(f"Successfully embedded {len(embedded_tasks)} tasks")
            return embedded_tasks
            
        except Exception as e:
            logger.error(f"Error embedding tasks: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of generated embeddings."""
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")
        
        # Test with a sample text to determine dimension
        sample_embedding = self._generate_embedding_sync("test")
        return len(sample_embedding)
    
    async def close(self) -> None:
        """Clean up resources."""
        self._executor.shutdown(wait=True)
        if self.model:
            # Clear model from memory
            del self.model
            self.model = None
        logger.info("Vector embedder closed")