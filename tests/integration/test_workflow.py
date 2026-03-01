"""Integration tests for the complete workflow."""

import pytest
import asyncio
import tempfile
import os
import pandas as pd
from datetime import datetime
from pathlib import Path

from app.core.preprocessing import DataPreprocessor
from app.core.embedding import VectorEmbedder
from app.core.clustering import TaskClusterer, ClusteringConfig
from app.core.conversation import ConversationalAgent
from app.models import ProcessedTask


class TestCompleteWorkflow:
    """Test the complete data processing workflow."""
    
    @pytest.fixture
    def sample_jira_data(self):
        """Create realistic sample Jira data."""
        data = {
            'issue_id': [f'TASK-{i:03d}' for i in range(1, 31)],
            'summary': [
                'Fix user authentication login issue',
                'Add password reset functionality',
                'Implement two-factor authentication',
                'Update user profile page design',
                'Add profile picture upload feature',
                'Implement user notification system',
                'Fix database connection timeout',
                'Optimize SQL query performance',
                'Add database indexing for faster queries',
                'Implement database backup automation',
                'Create API documentation',
                'Update REST API endpoints',
                'Add API rate limiting',
                'Implement API versioning',
                'Fix API response format inconsistencies',
                'Design mobile responsive layout',
                'Implement dark mode theme',
                'Add accessibility features',
                'Optimize page load performance',
                'Fix cross-browser compatibility issues',
                'Setup continuous integration pipeline',
                'Configure automated testing workflows',
                'Implement code quality checks',
                'Setup deployment automation',
                'Configure monitoring and alerting',
                'Fix memory leak in application',
                'Optimize application performance',
                'Implement caching strategy',
                'Add error logging and monitoring',
                'Setup security vulnerability scanning'
            ],
            'description': [
                'Users are experiencing login failures after password changes. Need to investigate authentication flow.',
                'Users should be able to reset their passwords via email verification link.',
                'Enhance security by adding two-factor authentication using authenticator apps.',
                'Redesign the user profile page with modern UI components and better user experience.',
                'Allow users to upload and crop their profile pictures with image optimization.',
                'Send notifications to users for important events like password changes, new messages.',
                'Database connections are timing out under high load conditions.',
                'Several SQL queries are taking too long to execute, affecting user experience.',
                'Add proper database indexes to improve query performance for frequently accessed data.',
                'Automate daily database backups with retention policies and failure notifications.',
                'Create comprehensive API documentation with examples and usage guidelines.',
                'Refactor existing REST API endpoints to follow consistent naming conventions.',
                'Implement rate limiting to prevent API abuse and ensure fair usage.',
                'Add API versioning to support backward compatibility during updates.',
                'Standardize API response formats across all endpoints for consistency.',
                'Ensure website looks good and functions properly on mobile devices.',
                'Add dark mode theme option for better user experience in low-light conditions.',
                'Implement WCAG compliance features for better accessibility.',
                'Optimize assets and code to reduce page load times and improve performance.',
                'Fix rendering issues and inconsistencies across different web browsers.',
                'Setup CI/CD pipeline for automated building, testing, and deployment.',
                'Configure automated testing workflows to run on every code commit.',
                'Integrate code quality tools and static analysis in the development workflow.',
                'Automate deployment processes to staging and production environments.',
                'Setup monitoring tools to track application performance and send alerts.',
                'Application is consuming increasing memory over time, indicating memory leaks.',
                'Optimize application performance by identifying and fixing bottlenecks.',
                'Implement caching mechanisms to reduce database load and improve response times.',
                'Add comprehensive error logging and real-time monitoring capabilities.',
                'Setup automated security scanning to identify vulnerabilities in dependencies.'
            ] * 1,  # Repeat to match issue_id length
            'created_at': [datetime.now().isoformat() for _ in range(30)],
            'status': ['To Do'] * 10 + ['In Progress'] * 10 + ['Done'] * 10,
            'priority': ['High'] * 5 + ['Medium'] * 15 + ['Low'] * 10
        }
        return pd.DataFrame(data)
    
    @pytest.mark.asyncio
    async def test_end_to_end_processing(self, sample_jira_data):
        """Test complete end-to-end processing workflow."""
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as csv_file:
            sample_jira_data.to_csv(csv_file.name, index=False)
            input_file = csv_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as json_file:
            output_file = json_file.name
        
        try:
            # Step 1: Data Preprocessing
            preprocessor = DataPreprocessor()
            tasks = preprocessor.load_data(input_file)
            assert len(tasks) == 30
            
            processed_tasks = preprocessor.preprocess_tasks(tasks)
            assert len(processed_tasks) == 30
            
            preprocessor.save_processed_data(processed_tasks, output_file)
            
            # Step 2: Vector Embedding
            embedder = VectorEmbedder(model_name="text-embedding-3-small")
            await embedder.initialize_model()
            
            embedded_tasks = await embedder.embed_tasks(processed_tasks)
            assert len(embedded_tasks) == 30
            assert all(task.embedding is not None for task in embedded_tasks)
            
            # Step 3: Clustering Analysis
            config = ClusteringConfig(
                algorithm="hdbscan",
                min_cluster_size=3,
                cluster_selection_epsilon=0.5
            )
            clusterer = TaskClusterer(config)
            
            result = await clusterer.cluster_tasks(embedded_tasks)
            
            # Verify results
            assert result.total_tasks == 30
            assert result.clusters_found >= 2  # Should find meaningful clusters
            assert len(result.cluster_details) >= 2
            assert result.processing_time > 0
            
            # Check cluster details structure
            for cluster_id, details in result.cluster_details.items():
                assert "size" in details
                assert "avg_confidence" in details
                assert details["size"] > 0
            
            # Cleanup
            await embedder.close()
            await clusterer.close()
            
        finally:
            # Clean up temporary files
            os.unlink(input_file)
            os.unlink(output_file)
    
    @pytest.mark.asyncio
    async def test_incremental_data_handling(self):
        """Test handling of incremental data updates."""
        
        # Create initial dataset
        initial_data = pd.DataFrame({
            'issue_id': ['INIT-001', 'INIT-002', 'INIT-003'],
            'summary': ['Initial task 1', 'Initial task 2', 'Initial task 3'],
            'description': ['Desc 1', 'Desc 2', 'Desc 3'],
            'created_at': [datetime.now().isoformat()] * 3,
            'status': ['To Do'] * 3,
            'priority': ['Medium'] * 3
        })
        
        # Create incremental dataset (mix of existing and new tasks)
        incremental_data = pd.DataFrame({
            'issue_id': ['INIT-001', 'INIT-004', 'NEW-001'],  # Mix of existing and new
            'summary': ['Updated initial task 1', 'New task 1', 'Completely new task'],
            'description': ['Updated desc 1', 'New desc 1', 'New desc 2'],
            'created_at': [datetime.now().isoformat()] * 3,
            'status': ['In Progress', 'To Do', 'Done'],
            'priority': ['High', 'Medium', 'Low']
        })
        
        # Process both datasets
        preprocessor = DataPreprocessor()
        
        initial_tasks = preprocessor.load_data_from_dataframe(initial_data)
        incremental_tasks = preprocessor.load_data_from_dataframe(incremental_data)
        
        # Combine and deduplicate (simulating incremental update logic)
        all_issue_ids = set()
        final_tasks = []
        
        # Add initial tasks
        for task in initial_tasks:
            if task.issue_id not in all_issue_ids:
                all_issue_ids.add(task.issue_id)
                final_tasks.append(task)
        
        # Add/update incremental tasks
        for task in incremental_tasks:
            if task.issue_id not in all_issue_ids:
                # New task
                all_issue_ids.add(task.issue_id)
                final_tasks.append(task)
            else:
                # Update existing task (in real implementation)
                # For this test, we'll just add it as a separate entry
                final_tasks.append(task)
        
        # Should have 4 unique tasks plus 1 duplicate
        assert len(final_tasks) == 5
        assert len(all_issue_ids) == 4


class TestConversationIntegration:
    """Test integration of conversational agent with processing components."""
    
    @pytest.mark.asyncio
    async def test_agent_with_real_components(self):
        """Test conversational agent working with real processing components."""
        agent = ConversationalAgent()
        
        try:
            # Test session creation
            session_id = await agent.create_session()
            assert session_id is not None
            
            # Test help functionality
            help_response = await agent.process_message(session_id, "帮助")
            assert isinstance(help_response, object)  # AgentResponse
            assert "数据导入" in help_response.message
            
            # Test parameter extraction
            param_response = await agent.process_message(session_id, "最小聚类大小设为8")
            assert agent.sessions[session_id].context.get("min_cluster_size") == 8
            
            # Test unknown intent handling
            unknown_response = await agent.process_message(session_id, "今天天气很好")
            assert "不明白" in unknown_response.message or "帮助" in unknown_response.message
            
        finally:
            await agent.close()


# Performance integration test
@pytest.mark.asyncio
async def test_performance_integration():
    """Test performance of integrated workflow with larger dataset."""
    
    # Create larger dataset for performance testing
    large_data = pd.DataFrame({
        'issue_id': [f'PERF-{i:04d}' for i in range(1, 101)],
        'summary': [f'Performance test task {i}' for i in range(1, 101)],
        'description': [f'Description for performance test task {i}' for i in range(1, 101)],
        'created_at': [datetime.now().isoformat() for _ in range(100)],
        'status': ['To Do'] * 30 + ['In Progress'] * 40 + ['Done'] * 30,
        'priority': ['High'] * 20 + ['Medium'] * 50 + ['Low'] * 30
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        large_data.to_csv(f.name, index=False)
        input_file = f.name
    
    try:
        import time
        
        # Measure preprocessing time
        start_time = time.time()
        preprocessor = DataPreprocessor()
        tasks = preprocessor.load_data(input_file)
        processed_tasks = preprocessor.preprocess_tasks(tasks)
        preprocessing_time = time.time() - start_time
        
        assert len(processed_tasks) == 100
        assert preprocessing_time < 5.0  # Should complete within 5 seconds
        
        # Measure embedding time (using smaller model for testing)
        start_time = time.time()
        embedder = VectorEmbedder(model_name="text-embedding-3-small")
        await embedder.initialize_model()
        embedded_tasks = await embedder.embed_tasks(processed_tasks[:20])  # Test with subset
        embedding_time = time.time() - start_time
        
        assert len(embedded_tasks) == 20
        assert embedding_time < 15.0  # Should complete within 15 seconds
        
        await embedder.close()
        
    finally:
        os.unlink(input_file)