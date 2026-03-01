"""Data preprocessing utilities for Jira task analysis."""

import re
import pandas as pd
from typing import List, Optional
from datetime import datetime
import logging
from app.models import JiraTask, ProcessedTask

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data cleaning and preprocessing for Jira tasks."""
    
    def __init__(self, max_description_length: int = 500):
        self.max_description_length = max_description_length
        
    def load_data(self, file_path: str) -> List[JiraTask]:
        """Load Jira data from CSV or Excel file."""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel.")
            
            tasks = []
            for _, row in df.iterrows():
                task = self._row_to_task(row)
                if task:
                    tasks.append(task)
            
            logger.info(f"Loaded {len(tasks)} tasks from {file_path}")
            return tasks
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise
    
    def _row_to_task(self, row: pd.Series) -> Optional[JiraTask]:
        """Convert DataFrame row to JiraTask object."""
        try:
            # Required fields validation
            if pd.isna(row.get('issue_id')) or pd.isna(row.get('summary')):
                return None
            
            # Handle description (can be empty)
            description = str(row.get('description', '')) if not pd.isna(row.get('description')) else ''
            
            # Parse datetime
            created_at_str = row.get('created_at')
            if isinstance(created_at_str, str):
                created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
            elif isinstance(created_at_str, datetime):
                created_at = created_at_str
            else:
                created_at = datetime.now()
            
            # Parse updated_at datetime
            updated_at_str = row.get('updated_at')
            if isinstance(updated_at_str, str):
                updated_at = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))
            elif isinstance(updated_at_str, datetime):
                updated_at = updated_at_str
            else:
                updated_at = created_at  # Default to created_at if not provided
            
            # Status field has been removed
            
            # Handle cluster label
            cluster_label = row.get('cluster_label')
            if cluster_label is not None:
                cluster_label = str(cluster_label).strip() if cluster_label else None
            
            return JiraTask(
                issue_id=str(row['issue_id']),
                summary=str(row['summary']).strip(),
                description=description.strip(),
                created_at=created_at,
                updated_at=updated_at,
                cluster_label=cluster_label
            )
            
        except Exception as e:
            logger.warning(f"Error processing row: {e}")
            return None
    
    def clean_description(self, text: str) -> str:
        """Clean and preprocess task description."""
        if text is None:
            return ""
        
        # Convert to string
        text = str(text)
        
        # Return empty for empty strings
        if not text.strip():
            return ""
        
        # Remove code blocks
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`[^`]*`', '', text)
        
        # Remove log lines and stack traces
        log_patterns = [
            r'(ERROR|WARN|DEBUG|INFO|TRACE).*',
            r'\tat\s+[\w\.$]+\(.*?\)',
            r'Exception in thread.*',
            r'\[.*?\].*'
        ]
        
        for pattern in log_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)
        
        # Remove HTML tags but keep content
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Limit length
        if len(text) > self.max_description_length:
            text = text[:self.max_description_length] + "..."
        
        return text
    
    def preprocess_tasks(self, tasks: List[JiraTask]) -> List[ProcessedTask]:
        """Preprocess a list of Jira tasks."""
        processed_tasks = []
        
        for task in tasks:
            try:
                cleaned_desc = self.clean_description(task.description)
                
                processed_task = ProcessedTask(
                    issue_id=task.issue_id,
                    summary=task.summary,
                    description=task.description,
                    created_at=task.created_at,
                    updated_at=task.updated_at,
                    cluster_label=task.cluster_label,
                    cleaned_description=cleaned_desc
                )
                
                processed_tasks.append(processed_task)
                
            except Exception as e:
                logger.warning(f"Error preprocessing task {task.issue_id}: {e}")
                continue
        
        logger.info(f"Preprocessed {len(processed_tasks)} out of {len(tasks)} tasks")
        return processed_tasks
    
    def save_processed_data(self, tasks: List[ProcessedTask], output_path: str) -> None:
        """Save processed tasks to JSON file."""
        try:
            import json
            
            # Convert to serializable format
            data = []
            for task in tasks:
                task_dict = task.model_dump()
                # Convert datetime to ISO format
                if 'created_at' in task_dict and task_dict['created_at']:
                    task_dict['created_at'] = task_dict['created_at'].isoformat()
                if 'updated_at' in task_dict and task_dict['updated_at']:
                    task_dict['updated_at'] = task_dict['updated_at'].isoformat()
                if 'processed_at' in task_dict and task_dict['processed_at']:
                    task_dict['processed_at'] = task_dict['processed_at'].isoformat()
                data.append(task_dict)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(tasks)} processed tasks to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise
    
    def load_processed_data(self, input_path: str) -> List[ProcessedTask]:
        """Load processed tasks from JSON file."""
        try:
            import json
            
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            tasks = []
            for item in data:
                # Convert ISO strings back to datetime
                item['created_at'] = datetime.fromisoformat(item['created_at'])
                item['processed_at'] = datetime.fromisoformat(item['processed_at'])
                task = ProcessedTask(**item)
                tasks.append(task)
            
            logger.info(f"Loaded {len(tasks)} processed tasks from {input_path}")
            return tasks
            
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            raise