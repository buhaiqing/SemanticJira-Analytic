"""Vector embedding generation for Jira task analysis with performance optimizations."""

from typing import List, Optional, Dict
from collections import OrderedDict
import logging
from sentence_transformers import SentenceTransformer
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from functools import lru_cache
import numpy as np

from app.models import ProcessedTask

logger = logging.getLogger(__name__)


class ModelCache:
    """
    Thread-safe model cache with LRU eviction and metrics tracking.

    Features:
    - Thread-safe access with read-write locks
    - LRU eviction with configurable max size
    - Cache hit/miss metrics
    - Automatic memory management
    """

    def __init__(self, max_size: int = 3):
        self._cache: Dict[str, SentenceTransformer] = {}
        self._access_order: List[str] = []
        self._max_size = max_size
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[SentenceTransformer]:
        """Get model from cache with LRU update."""
        with self._lock:
            if key in self._cache:
                self._hits += 1
                # Update access order for LRU
                self._access_order.remove(key)
                self._access_order.append(key)
                logger.debug(f"Cache hit for model: {key} (hits={self._hits}, misses={self._misses})")
                return self._cache[key]
            self._misses += 1
            logger.debug(f"Cache miss for model: {key} (hits={self._hits}, misses={self._misses})")
            return None

    def put(self, key: str, model: SentenceTransformer) -> None:
        """Put model in cache with LRU eviction."""
        with self._lock:
            if key in self._cache:
                # Update existing entry
                self._access_order.remove(key)
            elif len(self._cache) >= self._max_size:
                # Evict least recently used
                oldest_key = self._access_order.pop(0)
                evicted_model = self._cache.pop(oldest_key)
                del evicted_model
                logger.info(f"Evicted model from cache: {oldest_key}")

            self._cache[key] = model
            self._access_order.append(key)
            logger.info(f"Cached model: {key} (cache_size={len(self._cache)}/{self._max_size})")

    def clear(self) -> None:
        """Clear all cached models."""
        with self._lock:
            for model in self._cache.values():
                del model
            self._cache.clear()
            self._access_order.clear()
            logger.info("Model cache cleared")

    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            return {
                'size': len(self._cache),
                'max_size': self._max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': f"{hit_rate:.1f}%"
            }


# Global model cache instance
_model_cache = ModelCache(max_size=3)


class VectorEmbedder:
    """
    High-performance vector embedder with advanced optimizations.

    Features:
    - Global model caching across instances
    - Dynamic batch size adjustment based on text length
    - Parallel embedding generation with memory efficiency
    - Lazy model loading
    - Automatic memory management
    """

    def __init__(self, model_name: str = "BGE-M3", cache_embeddings: bool = True,
                 max_batch_size: int = 64, memory_efficient: bool = True):
        """
        Initialize vector embedder with performance optimizations.

        Args:
            model_name: Name of the embedding model to use
            cache_embeddings: Whether to cache generated embeddings
            max_batch_size: Maximum batch size for parallel processing
            memory_efficient: Enable memory-efficient processing mode
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="Embedder")
        self._cache_embeddings = cache_embeddings
        self._max_batch_size = max_batch_size
        self._memory_efficient = memory_efficient
        # Use OrderedDict for LRU caching with numpy arrays for memory efficiency
        self._embedding_cache: Optional[Dict[str, np.ndarray]] = OrderedDict() if cache_embeddings else None
        self._max_cache_size = 1000
        self._cache_lock = threading.Lock()
        self._init_lock = asyncio.Lock()
        self._stats = {
            'embeddings_generated': 0,
            'cache_hits': 0,
            'batch_count': 0,
            'total_time': 0.0
        }

    async def initialize_model(self) -> None:
        """Initialize the embedding model asynchronously with caching."""
        if self.model is not None:
            return

        # Check global cache first
        cache_key = f"{self.model_name}_model"
        cached_model = _model_cache.get(cache_key)
        if cached_model is not None:
            self.model = cached_model
            logger.info(f"Using cached model: {self.model_name}")
            return

        # Acquire init lock to prevent duplicate loading
        async with self._init_lock:
            # Double-check after acquiring lock
            if self.model is not None:
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
                    self._load_model_sync,
                    model_id
                )

                # Store in global cache
                _model_cache.put(cache_key, self.model)
                logger.info(f"Successfully loaded and cached model: {model_id}")

            except Exception as e:
                logger.error(f"Error loading embedding model: {e}")
                raise

    def _load_model_sync(self, model_id: str) -> SentenceTransformer:
        """Load model synchronously (for executor)."""
        return SentenceTransformer(model_id)

    def _compute_text_hash(self, text: str) -> str:
        """Compute fast hash for text caching."""
        # Use faster hashing for short texts
        if len(text) < 1000:
            return hash(text)
        # For longer texts, use md5
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()[:16]

    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from local cache if available."""
        if not self._cache_embeddings or self._embedding_cache is None:
            return None

        text_hash = self._compute_text_hash(text)
        with self._cache_lock:
            if text_hash in self._embedding_cache:
                # Move to end for LRU tracking
                self._embedding_cache.move_to_end(text_hash)
                self._stats['cache_hits'] += 1
                # Convert numpy array to list for return
                return self._embedding_cache[text_hash].tolist()
        return None

    def _cache_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache generated embedding with LRU eviction."""
        if not self._cache_embeddings or self._embedding_cache is None:
            return

        text_hash = self._compute_text_hash(text)
        with self._cache_lock:
            if text_hash in self._embedding_cache:
                # Move to end for LRU tracking
                self._embedding_cache.move_to_end(text_hash)
            elif len(self._embedding_cache) >= self._max_cache_size:
                # Evict oldest (first) item - O(1) with OrderedDict
                self._embedding_cache.popitem(last=False)

            # Store as numpy array for memory efficiency (float32 = 4 bytes vs 28 bytes for Python float)
            self._embedding_cache[text_hash] = np.array(embedding, dtype=np.float32)

    def _calculate_optimal_batch_size(self, texts: List[str]) -> int:
        """
        Calculate optimal batch size based on text lengths and memory constraints.

        Args:
            texts: List of texts to process

        Returns:
            Optimal batch size
        """
        if not texts:
            return self._max_batch_size

        # Calculate average text length
        avg_length = sum(len(text) for text in texts) / len(texts)

        # Estimate memory usage per token (rough estimate: 4 bytes per float16)
        # BGE-M3 produces 1024-dimensional embeddings = ~4KB per embedding
        embedding_size_kb = 4  # Approximate

        # Available memory budget (conservative estimate: 256MB for embeddings)
        memory_budget_kb = 256 * 1024

        # Calculate batch size based on memory budget
        memory_based_batch_size = max(1, int(memory_budget_kb / (embedding_size_kb * 2)))

        # Adjust based on text length (longer texts = smaller batches)
        if avg_length > 500:
            length_factor = 0.5
        elif avg_length > 200:
            length_factor = 0.75
        else:
            length_factor = 1.0

        optimal_batch_size = int(memory_based_batch_size * length_factor)

        # Apply bounds
        optimal_batch_size = max(8, min(self._max_batch_size, optimal_batch_size))

        logger.debug(f"Calculated optimal batch size: {optimal_batch_size} "
                    f"(avg_length={avg_length:.0f}, memory_based={memory_based_batch_size})")

        return optimal_batch_size

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

        # Check cache first
        cached = self._get_cached_embedding(text)
        if cached is not None:
            return cached

        # Optimization: For short texts, execute synchronously to avoid executor overhead
        # Threshold: 500 characters (adjustable based on profiling)
        if len(text) < 500:
            start_time = time.time()
            embedding = self._generate_embedding_sync(text)
            self._cache_embedding(text, embedding)
            self._stats['embeddings_generated'] += 1
            self._stats['total_time'] += time.time() - start_time
            return embedding

        # For longer texts, use executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        start_time = time.time()

        embedding = await loop.run_in_executor(
            self._executor,
            self._generate_embedding_sync,
            text
        )

        # Cache the result
        self._cache_embedding(text, embedding)

        # Update stats
        self._stats['embeddings_generated'] += 1
        self._stats['total_time'] += time.time() - start_time

        return embedding

    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch with optimizations.

        Features:
        - Dynamic batch size adjustment
        - Progress logging for large batches
        - Memory-efficient processing
        - Parallel execution

        Args:
            texts: List of texts to process

        Returns:
            List of embeddings
        """
        if not texts:
            return []

        if self.model is None:
            await self.initialize_model()

        start_time = time.time()
        logger.info(f"Generating embeddings for {len(texts)} texts")

        # Calculate optimal batch size
        optimal_batch_size = self._calculate_optimal_batch_size(texts)
        logger.info(f"Using dynamic batch size: {optimal_batch_size}")

        try:
            # Use model's built-in batch processing with optimized batch size
            loop = asyncio.get_event_loop()

            # Process in chunks for very large inputs
            if len(texts) > optimal_batch_size * 10:
                all_embeddings = []
                for i in range(0, len(texts), optimal_batch_size * 10):
                    chunk = texts[i:i + optimal_batch_size * 10]
                    chunk_embeddings = await loop.run_in_executor(
                        self._executor,
                        lambda c=chunk: self.model.encode(
                            c,
                            convert_to_numpy=True,
                            batch_size=optimal_batch_size,
                            show_progress_bar=True
                        )
                    )
                    all_embeddings.extend([emb.tolist() for emb in chunk_embeddings])
                    logger.info(f"Processed chunk {i//optimal_batch_size + 1}/{(len(texts)-1)//(optimal_batch_size*10) + 1}")
                result = all_embeddings
            else:
                embeddings = await loop.run_in_executor(
                    self._executor,
                    lambda: self.model.encode(
                        texts,
                        convert_to_numpy=True,
                        batch_size=optimal_batch_size
                    )
                )
                result = [emb.tolist() for emb in embeddings]

            self._stats['batch_count'] += 1
            self._stats['embeddings_generated'] += len(result)
            self._stats['total_time'] += time.time() - start_time

            logger.info(f"Generated {len(result)} embeddings in {time.time() - start_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise

    def prepare_text_for_embedding(self, task: ProcessedTask) -> str:
        """
        Prepare text content for embedding generation with optimization.

        Args:
            task: Processed task to prepare

        Returns:
            Prepared text string
        """
        # Combine summary and cleaned description
        content_parts = [task.summary]

        if task.cleaned_description:
            content_parts.append(task.cleaned_description)

        # Join with separator
        text = " | ".join(content_parts)
        return text.strip()

    async def embed_tasks(self, tasks: List[ProcessedTask]) -> List[ProcessedTask]:
        """
        Generate embeddings for a list of processed tasks with batch optimization.

        Args:
            tasks: List of processed tasks

        Returns:
            List of tasks with embeddings attached
        """
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

    @property
    def stats(self) -> dict:
        """Get embedder statistics."""
        return {
            **self._stats,
            'model_loaded': self.model is not None,
            'cache_enabled': self._cache_embeddings,
            'cache_size': len(self._embedding_cache) if self._embedding_cache else 0
        }

    async def close(self) -> None:
        """Clean up resources."""
        self._executor.shutdown(wait=True)
        if self._embedding_cache:
            with self._cache_lock:
                self._embedding_cache.clear()
        logger.info(f"Vector embedder closed. Stats: {self.stats}")


# Convenience function for getting cache statistics
def get_model_cache_stats() -> dict:
    """Get global model cache statistics."""
    return _model_cache.stats
