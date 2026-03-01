"""Unit tests for data preprocessing module."""

import pytest
import pandas as pd
import tempfile
import os
from datetime import datetime
from app.core.preprocessing import DataPreprocessor
from app.models import JiraTask, ProcessedTask


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance for testing."""
        return DataPreprocessor(max_description_length=100)
    
    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data for testing."""
        data = {
            'issue_id': ['TASK-001', 'TASK-002', 'TASK-003'],
            'summary': ['Fix login bug', 'Add user profile', 'Update documentation'],
            'description': [
                'User cannot login after password reset',
                'Need to add profile picture upload feature',
                'Update API documentation with new endpoints'
            ],
            'created_at': ['2024-01-01T10:00:00Z', '2024-01-02T14:30:00Z', '2024-01-03T09:15:00Z'],
            'updated_at': ['2024-01-01T10:00:00Z', '2024-01-02T14:30:00Z', '2024-01-03T09:15:00Z'],
            'cluster_label': ['用户认证', '功能开发', '文档更新']
            # status field removed
            # priority field removed
        }
        return pd.DataFrame(data)
    
    def test_clean_description_basic(self, preprocessor):
        """Test basic description cleaning."""
        text = "This is a   test   description with    extra spaces"
        cleaned = preprocessor.clean_description(text)
        assert cleaned == "This is a test description with extra spaces"
    
    def test_clean_description_remove_code_blocks(self, preprocessor):
        """Test removal of code blocks."""
        text = "Here is some code: ```python\nprint('hello')\n``` and more text"
        cleaned = preprocessor.clean_description(text)
        assert "```" not in cleaned
        assert "print('hello')" not in cleaned
    
    def test_clean_description_remove_html_tags(self, preprocessor):
        """Test removal of HTML tags (content preserved)."""
        text = "This has <b>bold</b> and <i>italic</i> tags"
        cleaned = preprocessor.clean_description(text)
        assert "<b>" not in cleaned
        assert "<i>" not in cleaned
        assert "bold" in cleaned  # Content should be preserved
        assert "italic" in cleaned
    
    def test_clean_description_length_limit(self, preprocessor):
        """Test description length limiting."""
        long_text = "A" * 150  # 150 characters
        cleaned = preprocessor.clean_description(long_text)
        assert len(cleaned) <= 103  # 100 + "..."
        assert cleaned.endswith("...")
    
    def test_clean_description_empty_input(self, preprocessor):
        """Test handling of empty or invalid input."""
        assert preprocessor.clean_description("") == ""
        assert preprocessor.clean_description(None) == ""
        assert preprocessor.clean_description(123) == "123"  # Converts to string
    
    def test_row_to_task_valid(self, preprocessor, sample_csv_data):
        """Test conversion of valid DataFrame row to JiraTask."""
        row = sample_csv_data.iloc[0]
        task = preprocessor._row_to_task(row)
        
        assert isinstance(task, JiraTask)
        assert task.issue_id == "TASK-001"
        assert task.summary == "Fix login bug"
        # status field removed
        # priority field removed
    
    def test_row_to_task_missing_required_fields(self, preprocessor):
        """Test handling of rows with missing required fields."""
        invalid_row = pd.Series({
            'summary': 'Test task',  # Missing issue_id
            'description': 'Test description'
        })
        
        task = preprocessor._row_to_task(invalid_row)
        assert task is None
    
    def test_row_to_task_invalid_enums(self, preprocessor):
        """Test handling of invalid enum values."""
        row = pd.Series({
            'issue_id': 'TEST-001',
            'summary': 'Test task',
            'cluster_label': '测试聚类'
            # status field removed
            # priority field removed
        })
        
        task = preprocessor._row_to_task(row)
        assert isinstance(task, JiraTask)
        # status field removed
        # priority field removed
    
    def test_preprocess_tasks(self, preprocessor, sample_csv_data):
        """Test preprocessing of multiple tasks."""
        tasks = []
        for _, row in sample_csv_data.iterrows():
            task = preprocessor._row_to_task(row)
            if task:
                tasks.append(task)
        
        processed_tasks = preprocessor.preprocess_tasks(tasks)
        
        assert len(processed_tasks) == len(tasks)
        for task in processed_tasks:
            assert isinstance(task, ProcessedTask)
            assert hasattr(task, 'cleaned_description')
            assert hasattr(task, 'processed_at')
    
    def test_save_and_load_processed_data(self, preprocessor):
        """Test saving and loading processed data."""
        # Create sample processed tasks
        tasks = [
            ProcessedTask(
                issue_id="TEST-001",
                summary="Test task 1",
                description="Test description 1",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                cluster_label="用户认证",
                # status field removed
                # priority field removed
                cleaned_description="Cleaned desc 1"
            ),
            ProcessedTask(
                issue_id="TEST-002",
                summary="Test task 2",
                description="Test description 2",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                cluster_label="功能开发",
                # status field removed
                # priority field removed
                cleaned_description="Cleaned desc 2"
            )
        ]
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            preprocessor.save_processed_data(tasks, temp_file)
            
            # Load from file
            loaded_tasks = preprocessor.load_processed_data(temp_file)
            
            assert len(loaded_tasks) == len(tasks)
            for original, loaded in zip(tasks, loaded_tasks):
                assert original.issue_id == loaded.issue_id
                assert original.summary == loaded.summary
                assert original.cleaned_description == loaded.cleaned_description
                
        finally:
            # Clean up temporary file
            os.unlink(temp_file)
    
    def test_load_data_csv(self, preprocessor, sample_csv_data):
        """Test loading data from CSV file."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_csv_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            tasks = preprocessor.load_data(temp_file)
            assert len(tasks) == 3
            assert all(isinstance(task, JiraTask) for task in tasks)
            
        finally:
            os.unlink(temp_file)
    
    def test_load_data_unsupported_format(self, preprocessor):
        """Test handling of unsupported file formats."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            preprocessor.load_data("test.txt")