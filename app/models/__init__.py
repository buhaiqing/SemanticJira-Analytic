"""Data models for Jira task analysis system."""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


# Status field has been removed from the data structure


# Priority field has been removed from the data structure


class JiraTask(BaseModel):
    """Represents a Jira task."""
    issue_id: str = Field(..., description="Unique task identifier")
    summary: str = Field(..., description="Task summary/title")
    description: str = Field(..., description="Task detailed description")
    created_at: datetime = Field(..., description="Task creation timestamp")
    updated_at: datetime = Field(..., description="Task last update timestamp")
    cluster_label: Optional[str] = Field(None, description="Semantic cluster label for tagging tasks")


class ProcessedTask(JiraTask):
    """Processed task with additional metadata."""
    processed_at: datetime = Field(default_factory=datetime.now)
    cleaned_description: str = Field(..., description="Cleaned task description")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    cluster_id: Optional[int] = Field(None, description="Assigned cluster ID")
    cluster_confidence: Optional[float] = Field(None, description="Clustering confidence")


class ClusteringConfig(BaseModel):
    """Configuration for clustering algorithm."""
    min_cluster_size: int = Field(default=10, ge=2, le=100)
    cluster_selection_epsilon: float = Field(default=0.5, ge=0.1, le=2.0)
    algorithm: str = Field(default="hdbscan", pattern="^(hdbscan|kmeans)$")


class AnalysisResult(BaseModel):
    """Result of task analysis."""
    total_tasks: int = Field(..., description="Total number of tasks analyzed")
    clusters_found: int = Field(..., description="Number of clusters identified")
    cluster_details: Dict[int, Dict[str, Any]] = Field(..., description="Detailed cluster information")
    processing_time: float = Field(..., description="Analysis processing time in seconds")
    generated_at: datetime = Field(default_factory=datetime.now)


class ChatMessage(BaseModel):
    """Chat message model for Agent Skill."""
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatSession(BaseModel):
    """Chat session model."""
    session_id: str = Field(..., description="Unique session identifier")
    messages: List[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    context: Dict[str, Any] = Field(default_factory=dict)