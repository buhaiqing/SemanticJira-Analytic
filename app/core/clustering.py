"""Clustering algorithms for Jira task analysis with performance optimizations."""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import hdbscan
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache

from app.models import ProcessedTask, AnalysisResult, ClusteringConfig

logger = logging.getLogger(__name__)


class ClusteringOptimizer:
    """
    Optimizer for clustering algorithms with caching and performance tuning.

    Features:
    - Silhouette score caching
    - Optimal K caching for repeated datasets
    - Parallel silhouette calculation
    """

    def __init__(self, max_cache_size: int = 100):
        self._silhouette_cache: Dict[str, float] = {}
        self._optimal_k_cache: Dict[str, int] = {}
        self._cache_lock = threading.Lock()
        self._max_cache_size = max_cache_size

    def _compute_data_hash(self, embeddings: np.ndarray) -> str:
        """Compute hash for embedding data."""
        return hash(embeddings.shape) ^ hash(embeddings.tobytes()[:1000])

    def get_cached_optimal_k(self, embeddings: np.ndarray) -> int:
        """Get cached optimal K value if available."""
        data_hash = self._compute_data_hash(embeddings)
        with self._cache_lock:
            return self._optimal_k_cache.get(str(data_hash))

    def cache_optimal_k(self, embeddings: np.ndarray, optimal_k: int) -> None:
        """Cache optimal K value for future use."""
        data_hash = self._compute_data_hash(embeddings)
        with self._cache_lock:
            if len(self._optimal_k_cache) >= self._max_cache_size:
                # Remove oldest entry
                if self._optimal_k_cache:
                    self._optimal_k_cache.pop(next(iter(self._optimal_k_cache)))
            self._optimal_k_cache[str(data_hash)] = optimal_k

    def get_cached_silhouette(self, embeddings: np.ndarray, k: int) -> float:
        """Get cached silhouette score if available."""
        cache_key = f"{self._compute_data_hash(embeddings)}_{k}"
        with self._cache_lock:
            return self._silhouette_cache.get(cache_key)

    def cache_silhouette(self, embeddings: np.ndarray, k: int, score: float) -> None:
        """Cache silhouette score for future use."""
        cache_key = f"{self._compute_data_hash(embeddings)}_{k}"
        with self._cache_lock:
            if len(self._silhouette_cache) >= self._max_cache_size:
                self._silhouette_cache.pop(next(iter(self._silhouette_cache)))
            self._silhouette_cache[cache_key] = score


# Global optimizer instance
_clustering_optimizer = ClusteringOptimizer()


class TaskClusterer:
    """
    High-performance clustering of Jira tasks based on embeddings.

    Features:
    - Optimized async execution with configurable workers
    - Parallel silhouette score calculation
    - Memory-efficient embedding processing
    - Cached optimal cluster detection
    - Batch processing for large datasets
    """

    def __init__(self, config: ClusteringConfig, max_workers: int = None,
                 enable_optimization: bool = True):
        """
        Initialize task clusterer with performance optimizations.

        Args:
            config: Clustering configuration
            max_workers: Maximum workers for parallel processing
            enable_optimization: Enable performance optimizations
        """
        self.config = config
        self._max_workers = max_workers or 2
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="ClusterWorker"
        )
        self._enable_optimization = enable_optimization
        self._optimizer = _clustering_optimizer if enable_optimization else None
        self._stats = {
            'clustering_runs': 0,
            'total_time': 0.0,
            'avg_time': 0.0
        }

    async def cluster_tasks(self, tasks: List[ProcessedTask]) -> AnalysisResult:
        """
        Perform clustering analysis on embedded tasks with optimizations.

        Args:
            tasks: List of processed tasks with embeddings

        Returns:
            AnalysisResult with clustering details
        """
        if not tasks:
            raise ValueError("No tasks provided for clustering")

        # Filter tasks with embeddings
        embedded_tasks = [task for task in tasks if task.embedding is not None]
        if not embedded_tasks:
            raise ValueError("No tasks with embeddings found")

        logger.info(f"Clustering {len(embedded_tasks)} embedded tasks")

        start_time = datetime.now()

        try:
            if self.config.algorithm.lower() == "hdbscan":
                result = await self._cluster_with_hdbscan(embedded_tasks)
            elif self.config.algorithm.lower() == "kmeans":
                result = await self._cluster_with_kmeans(embedded_tasks)
            else:
                raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")

            processing_time = (datetime.now() - start_time).total_seconds()

            # Update stats
            self._stats['clustering_runs'] += 1
            self._stats['total_time'] += processing_time
            self._stats['avg_time'] = self._stats['total_time'] / self._stats['clustering_runs']

            result.processing_time = processing_time

            logger.info(f"Clustering completed in {processing_time:.2f} seconds")
            logger.info(f"Found {result.clusters_found} clusters")

            return result

        except Exception as e:
            logger.error(f"Error during clustering: {e}")
            raise

    async def _cluster_with_hdbscan(self, tasks: List[ProcessedTask]) -> AnalysisResult:
        """Perform HDBSCAN clustering with optimizations."""
        try:
            # Extract embeddings as numpy array
            embeddings = np.array([task.embedding for task in tasks], dtype=np.float32)

            # Perform clustering in thread pool
            loop = asyncio.get_event_loop()
            clusterer = await loop.run_in_executor(
                self._executor,
                self._fit_hdbscan,
                embeddings
            )

            # Batch assign cluster IDs to tasks (more efficient)
            labels = clusterer.labels_
            probabilities = clusterer.probabilities_ if hasattr(clusterer, 'probabilities_') else None

            clustered_tasks = []
            for i, (task, cluster_id) in enumerate(zip(tasks, labels)):
                clustered_task = task.model_copy()
                clustered_task.cluster_id = int(cluster_id) if cluster_id != -1 else None
                clustered_task.cluster_confidence = (
                    float(probabilities[i]) if probabilities is not None and cluster_id != -1 else None
                )
                clustered_tasks.append(clustered_task)

            # Generate analysis result
            result = self._generate_analysis_result_optimized(clustered_tasks, clusterer, embeddings)
            return result

        except Exception as e:
            logger.error(f"Error in HDBSCAN clustering: {e}")
            raise

    def _fit_hdbscan(self, embeddings: np.ndarray):
        """Fit HDBSCAN model (sync for executor) with optimized parameters."""
        n_samples = len(embeddings)
        return hdbscan.HDBSCAN(
            min_cluster_size=self.config.min_cluster_size,
            cluster_selection_epsilon=self.config.cluster_selection_epsilon,
            metric='euclidean',  # Use euclidean (cosine requires precomputed distance matrix)
            core_dist_n_neighbors=min(15, max(5, n_samples // 20)),
            approx_min_span_tree=True,  # Enable approximation for faster execution
            n_jobs=-1  # Use all CPU cores
        ).fit(embeddings)

    async def _cluster_with_kmeans(self, tasks: List[ProcessedTask]) -> AnalysisResult:
        """Perform K-Means clustering with automatic optimal cluster detection."""
        try:
            embeddings = np.array([task.embedding for task in tasks], dtype=np.float32)

            # Find optimal number of clusters with caching
            optimal_k = await self._find_optimal_clusters_kmeans_cached(embeddings)

            # Perform clustering
            loop = asyncio.get_event_loop()
            kmeans = await loop.run_in_executor(
                self._executor,
                self._fit_kmeans,
                embeddings,
                optimal_k
            )

            # Batch process cluster assignments
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_

            clustered_tasks = []
            for i, (task, cluster_id) in enumerate(zip(tasks, labels)):
                clustered_task = task.model_copy()
                clustered_task.cluster_id = int(cluster_id)
                clustered_task.cluster_confidence = self._calculate_kmeans_confidence_fast(
                    embeddings[i],
                    centers[cluster_id]
                )
                clustered_tasks.append(clustered_task)

            # Generate analysis result
            result = self._generate_analysis_result_kmeans(clustered_tasks, kmeans, optimal_k)
            return result

        except Exception as e:
            logger.error(f"Error in K-Means clustering: {e}")
            raise

    def _fit_kmeans(self, embeddings: np.ndarray, optimal_k: int):
        """Fit KMeans model (sync for executor)."""
        return KMeans(n_clusters=optimal_k, random_state=42, n_init=10).fit(embeddings)

    async def _find_optimal_clusters_kmeans_cached(self, embeddings: np.ndarray) -> int:
        """Find optimal K with caching support."""
        # Check cache first
        if self._optimizer:
            cached_k = self._optimizer.get_cached_optimal_k(embeddings)
            if cached_k is not None:
                logger.info(f"Using cached optimal K: {cached_k}")
                return cached_k

        # Calculate optimal K
        optimal_k = await self._find_optimal_clusters_kmeans(embeddings)

        # Cache result
        if self._optimizer:
            self._optimizer.cache_optimal_k(embeddings, optimal_k)

        return optimal_k

    async def _find_optimal_clusters_kmeans(self, embeddings: np.ndarray) -> int:
        """Find optimal number of clusters using silhouette analysis with parallel computation."""
        try:
            min_k = 2
            max_k = min(20, len(embeddings) // 2)

            if max_k <= min_k:
                return min_k

            loop = asyncio.get_event_loop()

            # Calculate silhouette scores in parallel batches
            k_values = list(range(min_k, max_k + 1))

            # Parallel silhouette calculation
            silhouette_scores = await self._calculate_silhouette_scores_parallel(
                embeddings, k_values, loop
            )

            # Return k with highest silhouette score
            if not silhouette_scores:
                return min_k

            optimal_k, best_score = max(silhouette_scores, key=lambda x: x[1])
            logger.info(f"Optimal number of clusters: {optimal_k} (silhouette_score={best_score:.3f})")

            return optimal_k

        except Exception as e:
            logger.warning(f"Error finding optimal clusters, using default: {e}")
            return 5

    async def _calculate_silhouette_scores_parallel(
        self, embeddings: np.ndarray, k_values: List[int], loop: asyncio.AbstractEventLoop
    ) -> List[Tuple[int, float]]:
        """
        Calculate silhouette scores in parallel for multiple k values.

        Args:
            embeddings: Input embeddings
            k_values: List of k values to test
            loop: Event loop for async execution

        Returns:
            List of (k, score) tuples
        """
        # Check cache for all k values first
        cached_scores = []
        uncached_scores = []
        uncached_k = []

        if self._optimizer:
            for k in k_values:
                cached_score = self._optimizer.get_cached_silhouette(embeddings, k)
                if cached_score is not None:
                    cached_scores.append((k, cached_score))
                else:
                    uncached_k.append(k)
        else:
            uncached_k = k_values

        # Calculate uncached scores in parallel
        if uncached_k:
            # Create futures for parallel execution
            futures = []
            for k in uncached_k:
                future = loop.run_in_executor(
                    self._executor,
                    self._calculate_single_silhouette_score,
                    embeddings,
                    k
                )
                futures.append((k, future))

            # Gather results
            for k, future in futures:
                score = await future
                if score >= 0:  # Valid score
                    uncached_scores.append((k, score))
                    # Cache the result
                    if self._optimizer:
                        self._optimizer.cache_silhouette(embeddings, k, score)

        return cached_scores + uncached_scores

    def _calculate_single_silhouette_score(self, embeddings: np.ndarray, k: int) -> float:
        """Calculate silhouette score for a single k value."""
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embeddings)
            score = silhouette_score(embeddings, kmeans.labels_)
            return float(score)
        except Exception:
            return -1.0  # Invalid score

    def _calculate_kmeans_confidence_fast(self, point: np.ndarray, centroid: np.ndarray) -> float:
        """Calculate confidence based on distance to cluster center (optimized)."""
        try:
            # Use numpy operations for better performance
            distance = np.linalg.norm(point - centroid)
            # Exponential decay for smoother confidence values
            std_val = np.std(point) or 1.0  # Avoid division by zero
            confidence = np.exp(-distance / std_val)
            return float(np.clip(confidence, 0.0, 1.0))
        except Exception:
            return 0.5

    def _generate_analysis_result_optimized(
        self, tasks: List[ProcessedTask], clusterer, embeddings: np.ndarray
    ) -> AnalysisResult:
        """Generate analysis result from HDBSCAN clustering (optimized)."""
        cluster_details = {}
        noise_points = 0

        # Single pass through tasks for better performance
        cluster_counts: Dict[int, int] = {}
        cluster_tasks_map: Dict[int, List[ProcessedTask]] = {}

        for task in tasks:
            if task.cluster_id is not None:
                cluster_counts[task.cluster_id] = cluster_counts.get(task.cluster_id, 0) + 1
                if task.cluster_id not in cluster_tasks_map:
                    cluster_tasks_map[task.cluster_id] = []
                cluster_tasks_map[task.cluster_id].append(task)
            else:
                noise_points += 1

        # Generate cluster details
        for cluster_id, count in cluster_counts.items():
            cluster_task_list = cluster_tasks_map[cluster_id]
            confidences = [t.cluster_confidence for t in cluster_task_list if t.cluster_confidence is not None]

            cluster_details[cluster_id] = {
                "size": count,
                "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
                "sample_tasks": [
                    {
                        "issue_id": task.issue_id,
                        "summary": task.summary[:100] + "..." if len(task.summary) > 100 else task.summary
                    }
                    for task in cluster_task_list[:3]
                ]
            }

        if noise_points > 0:
            cluster_details[-1] = {
                "size": noise_points,
                "avg_confidence": 0.0,
                "is_noise": True,
                "description": "Unclustered/Noise points"
            }

        return AnalysisResult(
            total_tasks=len(tasks),
            clusters_found=len(cluster_counts),
            cluster_details=cluster_details,
            processing_time=0.0
        )

    def _generate_analysis_result_kmeans(
        self, tasks: List[ProcessedTask], kmeans, optimal_k: int
    ) -> AnalysisResult:
        """Generate analysis result from K-Means clustering."""
        cluster_details = {}

        # Single pass counting
        cluster_counts: Dict[int, int] = {}
        cluster_tasks_map: Dict[int, List[ProcessedTask]] = {}

        for task in tasks:
            cluster_counts[task.cluster_id] = cluster_counts.get(task.cluster_id, 0) + 1
            if task.cluster_id not in cluster_tasks_map:
                cluster_tasks_map[task.cluster_id] = []
            cluster_tasks_map[task.cluster_id].append(task)

        # Generate cluster details
        for cluster_id, count in cluster_counts.items():
            cluster_task_list = cluster_tasks_map[cluster_id]
            confidences = [t.cluster_confidence for t in cluster_task_list if t.cluster_confidence is not None]

            cluster_details[cluster_id] = {
                "size": count,
                "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
                "centroid_distance": float(np.linalg.norm(kmeans.cluster_centers_[cluster_id])),
                "sample_tasks": [
                    {
                        "issue_id": task.issue_id,
                        "summary": task.summary[:100] + "..." if len(task.summary) > 100 else task.summary
                    }
                    for task in cluster_task_list[:3]
                ]
            }

        return AnalysisResult(
            total_tasks=len(tasks),
            clusters_found=optimal_k,
            cluster_details=cluster_details,
            processing_time=0.0
        )

    @property
    def stats(self) -> dict:
        """Get clusterer statistics."""
        return self._stats

    async def close(self) -> None:
        """Clean up resources."""
        self._executor.shutdown(wait=True)
        logger.info(f"Task clusterer closed. Stats: {self.stats}")
