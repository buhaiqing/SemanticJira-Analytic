"""Clustering algorithms for Jira task analysis."""

import numpy as np
from typing import List
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import hdbscan
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.models import ProcessedTask, AnalysisResult, ClusteringConfig

logger = logging.getLogger(__name__)


class TaskClusterer:
    """Handles clustering of Jira tasks based on their embeddings."""
    
    def __init__(self, config: ClusteringConfig):
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=2)
        
    async def cluster_tasks(self, tasks: List[ProcessedTask]) -> AnalysisResult:
        """Perform clustering analysis on embedded tasks."""
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
            result.processing_time = processing_time
            
            logger.info(f"Clustering completed in {processing_time:.2f} seconds")
            logger.info(f"Found {result.clusters_found} clusters")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during clustering: {e}")
            raise
    
    async def _cluster_with_hdbscan(self, tasks: List[ProcessedTask]) -> AnalysisResult:
        """Perform HDBSCAN clustering."""
        try:
            # Extract embeddings
            embeddings = np.array([task.embedding for task in tasks])
            
            # Perform clustering in thread pool
            loop = asyncio.get_event_loop()
            clusterer = await loop.run_in_executor(
                self._executor,
                lambda: hdbscan.HDBSCAN(
                    min_cluster_size=self.config.min_cluster_size,
                    cluster_selection_epsilon=self.config.cluster_selection_epsilon,
                    metric='euclidean'
                ).fit(embeddings)
            )
            
            # Assign cluster IDs to tasks
            clustered_tasks = []
            for task, cluster_id in zip(tasks, clusterer.labels_):
                clustered_task = task.model_copy()
                clustered_task.cluster_id = int(cluster_id) if cluster_id != -1 else None
                clustered_task.cluster_confidence = (
                    float(clusterer.probabilities_[tasks.index(task)]) 
                    if cluster_id != -1 else None
                )
                clustered_tasks.append(clustered_task)
            
            # Generate analysis result
            result = self._generate_analysis_result(clustered_tasks, clusterer)
            return result
            
        except Exception as e:
            logger.error(f"Error in HDBSCAN clustering: {e}")
            raise
    
    async def _cluster_with_kmeans(self, tasks: List[ProcessedTask]) -> AnalysisResult:
        """Perform K-Means clustering with automatic optimal cluster detection."""
        try:
            embeddings = np.array([task.embedding for task in tasks])
            
            # Find optimal number of clusters using silhouette analysis
            optimal_k = await self._find_optimal_clusters_kmeans(embeddings)
            
            # Perform clustering
            loop = asyncio.get_event_loop()
            kmeans = await loop.run_in_executor(
                self._executor,
                lambda: KMeans(n_clusters=optimal_k, random_state=42, n_init=10).fit(embeddings)
            )
            
            # Assign cluster IDs to tasks
            clustered_tasks = []
            for task, cluster_id in zip(tasks, kmeans.labels_):
                clustered_task = task.model_copy()
                clustered_task.cluster_id = int(cluster_id)
                clustered_task.cluster_confidence = self._calculate_kmeans_confidence(
                    embeddings[tasks.index(task)], 
                    kmeans.cluster_centers_[cluster_id]
                )
                clustered_tasks.append(clustered_task)
            
            # Generate analysis result
            result = self._generate_analysis_result_kmeans(clustered_tasks, kmeans, optimal_k)
            return result
            
        except Exception as e:
            logger.error(f"Error in K-Means clustering: {e}")
            raise
    
    async def _find_optimal_clusters_kmeans(self, embeddings: np.ndarray) -> int:
        """Find optimal number of clusters using silhouette analysis."""
        try:
            min_k = 2
            max_k = min(20, len(embeddings) // 2)
            
            if max_k <= min_k:
                return min_k
            
            loop = asyncio.get_event_loop()
            
            # Calculate silhouette scores for different k values
            silhouette_scores = []
            for k in range(min_k, max_k + 1):
                score = await loop.run_in_executor(
                    self._executor,
                    self._calculate_silhouette_score,
                    embeddings,
                    k
                )
                silhouette_scores.append((k, score))
            
            # Return k with highest silhouette score
            optimal_k, _ = max(silhouette_scores, key=lambda x: x[1])
            logger.info(f"Optimal number of clusters determined: {optimal_k}")
            
            return optimal_k
            
        except Exception as e:
            logger.warning(f"Error finding optimal clusters, using default: {e}")
            return 5
    
    def _calculate_silhouette_score(self, embeddings: np.ndarray, k: int) -> float:
        """Calculate silhouette score for given k (sync version)."""
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embeddings)
            score = silhouette_score(embeddings, kmeans.labels_)
            return float(score)
        except Exception:
            return -1.0  # Invalid score
    
    def _calculate_kmeans_confidence(self, point: np.ndarray, centroid: np.ndarray) -> float:
        """Calculate confidence based on distance to cluster center."""
        try:
            distance = np.linalg.norm(point - centroid)
            # Convert distance to confidence (inverse relationship)
            # Using exponential decay for smoother confidence values
            confidence = np.exp(-distance / np.std(point))
            return float(np.clip(confidence, 0.0, 1.0))
        except Exception:
            return 0.5
    
    def _generate_analysis_result(self, tasks: List[ProcessedTask], clusterer) -> AnalysisResult:
        """Generate analysis result from HDBSCAN clustering."""
        cluster_details = {}
        noise_points = 0
        
        # Count clusters and noise points
        cluster_counts = {}
        for task in tasks:
            if task.cluster_id is not None:
                cluster_counts[task.cluster_id] = cluster_counts.get(task.cluster_id, 0) + 1
            else:
                noise_points += 1
        
        # Generate cluster details
        for cluster_id, count in cluster_counts.items():
            cluster_tasks = [t for t in tasks if t.cluster_id == cluster_id]
            confidences = [t.cluster_confidence for t in cluster_tasks if t.cluster_confidence is not None]
            
            cluster_details[cluster_id] = {
                "size": count,
                "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
                "sample_tasks": [
                    {
                        "issue_id": task.issue_id,
                        "summary": task.summary[:100] + "..." if len(task.summary) > 100 else task.summary
                    }
                    for task in cluster_tasks[:3]  # Top 3 sample tasks
                ]
            }
        
        if noise_points > 0:
            cluster_details[-1] = {
                "size": noise_points,
                "avg_confidence": 0.0,
                "is_noise": True,
                "description": "Unclustered/Noise points"
            }
        
        import time
        import time
        processing_time = time.time() - getattr(self, '_start_time', time.time())
        return AnalysisResult(
            total_tasks=len(tasks),
            clusters_found=len(cluster_counts),
            cluster_details=cluster_details,
            processing_time=max(0.0, processing_time)  # Ensure non-negative
        )
    
    def _generate_analysis_result_kmeans(self, tasks: List[ProcessedTask], kmeans, optimal_k: int) -> AnalysisResult:
        """Generate analysis result from K-Means clustering."""
        cluster_details = {}
        
        # Count clusters
        cluster_counts = {}
        for task in tasks:
            cluster_counts[task.cluster_id] = cluster_counts.get(task.cluster_id, 0) + 1
        
        # Generate cluster details
        for cluster_id, count in cluster_counts.items():
            cluster_tasks = [t for t in tasks if t.cluster_id == cluster_id]
            confidences = [t.cluster_confidence for t in cluster_tasks if t.cluster_confidence is not None]
            
            cluster_details[cluster_id] = {
                "size": count,
                "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
                "centroid_distance": float(np.linalg.norm(kmeans.cluster_centers_[cluster_id])),
                "sample_tasks": [
                    {
                        "issue_id": task.issue_id,
                        "summary": task.summary[:100] + "..." if len(task.summary) > 100 else task.summary
                    }
                    for task in cluster_tasks[:3]  # Top 3 sample tasks
                ]
            }
        
        return AnalysisResult(
            total_tasks=len(tasks),
            clusters_found=optimal_k,
            cluster_details=cluster_details
        )
    
    async def close(self) -> None:
        """Clean up resources."""
        self._executor.shutdown(wait=True)
        logger.info("Task clusterer closed")