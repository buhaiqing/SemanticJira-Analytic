"""Unit tests for conversational agent module."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from app.core.conversation import ConversationalAgent, AgentResponse
from app.models import ChatSession, ProcessedTask
from datetime import datetime


class TestConversationalAgent:
    """Test suite for ConversationalAgent class."""
    
    @pytest.fixture
    def agent(self):
        """Create conversational agent instance for testing."""
        return ConversationalAgent()
    
    @pytest.mark.asyncio
    async def test_create_session(self, agent):
        """Test session creation."""
        session_id = await agent.create_session()
        
        assert session_id is not None
        assert isinstance(session_id, str)
        assert session_id in agent.sessions
        assert isinstance(agent.sessions[session_id], ChatSession)
    
    @pytest.mark.asyncio
    async def test_process_message_help_intent(self, agent):
        """Test processing help requests."""
        session_id = await agent.create_session()
        response = await agent.process_message(session_id, "帮助")
        
        assert isinstance(response, AgentResponse)
        assert "您好" in response.message
        assert "数据导入" in response.message
        assert response.action_required is None
    
    @pytest.mark.asyncio
    async def test_process_message_data_import_intent(self, agent):
        """Test processing data import requests."""
        session_id = await agent.create_session()
        response = await agent.process_message(session_id, "我想导入Jira数据")
        
        assert isinstance(response, AgentResponse)
        assert "上传" in response.message or "导入" in response.message
        assert response.action_required == "file_upload"
    
    @pytest.mark.asyncio
    async def test_process_message_clustering_intent(self, agent):
        """Test processing clustering requests."""
        # First create session with processed data
        session_id = await agent.create_session()
        agent.sessions[session_id].context["data_processed"] = True
        agent.sessions[session_id].context["processed_tasks"] = []
        
        response = await agent.process_message(session_id, "进行聚类分析")
        
        assert isinstance(response, AgentResponse)
        # Should prompt for data processing first
        assert "先上传" in response.message or "处理数据" in response.message
    
    @pytest.mark.asyncio
    async def test_process_message_results_intent(self, agent):
        """Test processing results requests."""
        session_id = await agent.create_session()
        agent.sessions[session_id].context["analysis_completed"] = True
        agent.sessions[session_id].context["clustering_result"] = Mock()
        
        response = await agent.process_message(session_id, "查看分析结果")
        
        assert isinstance(response, AgentResponse)
        assert response.action_required == "export_options"
    
    @pytest.mark.asyncio
    async def test_process_message_parameters_intent(self, agent):
        """Test processing parameter adjustment requests."""
        session_id = await agent.create_session()
        response = await agent.process_message(session_id, "最小聚类大小设为15")
        
        assert isinstance(response, AgentResponse)
        assert "调整参数" in response.message
        assert agent.sessions[session_id].context.get("min_cluster_size") == 15
    
    @pytest.mark.asyncio
    async def test_process_message_unknown_intent(self, agent):
        """Test processing unknown intents."""
        session_id = await agent.create_session()
        response = await agent.process_message(session_id, "这是什么意思？")
        
        assert isinstance(response, AgentResponse)
        assert "不明白" in response.message or "帮助" in response.message
    
    @pytest.mark.asyncio
    async def test_extract_parameter(self, agent):
        """Test parameter extraction from messages."""
        # Test min_size extraction
        min_size = agent._extract_parameter("最小聚类大小设为20", "min_size")
        assert min_size == 20
        
        min_size = agent._extract_parameter("min size 15", "min_size")
        assert min_size == 15
        
        # Test epsilon extraction
        epsilon = agent._extract_parameter("epsilon设为0.3", "epsilon")
        assert epsilon == 0.3
        
        # Test default values
        default_min_size = agent._extract_parameter("随便说点什么", "min_size", 10)
        assert default_min_size == 10
    
    @pytest.mark.asyncio
    async def test_session_management(self, agent):
        """Test session creation and management."""
        # Create multiple sessions
        session_id1 = await agent.create_session()
        session_id2 = await agent.create_session()
        
        assert session_id1 != session_id2
        assert len(agent.sessions) == 2
        
        # Test session cleanup
        await agent.cleanup_session(session_id1)
        assert session_id1 not in agent.sessions
        assert len(agent.sessions) == 1
    
    @pytest.mark.asyncio
    async def test_context_updates(self, agent):
        """Test context updates in responses."""
        session_id = await agent.create_session()
        
        # Process message that should update context
        response = await agent.process_message(session_id, "最小聚类大小设为25")
        
        assert response.context_updates is not None
        assert "min_cluster_size" in response.context_updates
        assert response.context_updates["min_cluster_size"] == 25
        
        # Verify context was updated in session
        assert agent.sessions[session_id].context.get("min_cluster_size") == 25


class TestIntentRecognition:
    """Test intent recognition functionality."""
    
    @pytest.fixture
    def agent(self):
        return ConversationalAgent()
    
    def test_recognize_data_import_intent(self, agent):
        """Test recognition of data import intents."""
        test_messages = [
            "导入Jira数据",
            "上传CSV文件",
            "分析任务数据",
            "处理Jira导出",
            "load data"
        ]
        
        for message in test_messages:
            intent = agent._recognize_intent(message)
            assert intent == "data_import"
    
    def test_recognize_clustering_intent(self, agent):
        """Test recognition of clustering intents."""
        test_messages = [
            "聚类分析",
            "分类任务",
            "cluster analysis",
            "分组任务"
        ]
        
        for message in test_messages:
            intent = agent._recognize_intent(message)
            assert intent == "clustering"
    
    def test_recognize_results_intent(self, agent):
        """Test recognition of results intents."""
        test_messages = [
            "查看结果",
            "显示分析",
            "show results",
            "导出结果"
        ]
        
        for message in test_messages:
            intent = agent._recognize_intent(message)
            assert intent == "results"
    
    def test_recognize_help_intent(self, agent):
        """Test recognition of help intents."""
        test_messages = [
            "帮助",
            "怎么用",
            "help",
            "如何使用"
        ]
        
        for message in test_messages:
            intent = agent._recognize_intent(message)
            assert intent == "help"
    
    def test_recognize_unknown_intent(self, agent):
        """Test recognition of unknown intents."""
        test_messages = [
            "今天天气怎么样",
            "随便聊聊",
            "123456"
        ]
        
        for message in test_messages:
            intent = agent._recognize_intent(message)
            assert intent == "unknown"


class TestResponseGeneration:
    """Test response generation functionality."""
    
    @pytest.fixture
    def agent(self):
        return ConversationalAgent()
    
    def test_generate_clustering_summary(self, agent):
        """Test clustering result summary generation."""
        # Mock analysis result
        mock_result = Mock()
        mock_result.total_tasks = 100
        mock_result.clusters_found = 5
        mock_result.processing_time = 2.5
        mock_result.cluster_details = {
            0: {"size": 30, "avg_confidence": 0.85},
            1: {"size": 25, "avg_confidence": 0.78},
            2: {"size": 20, "avg_confidence": 0.92},
            -1: {"size": 25, "avg_confidence": 0.0}  # Noise cluster
        }
        
        summary = agent._generate_clustering_summary(mock_result)
        
        assert "聚类分析完成" in summary
        assert "100" in summary  # Total tasks
        assert "5" in summary   # Clusters found
        assert "2.5" in summary # Processing time
        assert "噪声" in summary # Noise cluster
    
    def test_generate_detailed_results(self, agent):
        """Test detailed results generation."""
        # Mock analysis result with sample tasks
        mock_result = Mock()
        mock_result.total_tasks = 50
        mock_result.clusters_found = 3
        mock_result.processing_time = 1.8
        mock_result.generated_at = datetime.now()
        mock_result.cluster_details = {
            0: {
                "size": 20,
                "avg_confidence": 0.88,
                "sample_tasks": [
                    {"summary": "Frontend development task"},
                    {"summary": "UI implementation"}
                ]
            },
            1: {
                "size": 15,
                "avg_confidence": 0.75,
                "sample_tasks": [
                    {"summary": "Backend API development"}
                ]
            }
        }
        
        detailed_results = agent._generate_detailed_results(mock_result)
        
        assert "详细分析报告" in detailed_results
        assert "50" in detailed_results  # Total tasks
        assert "3" in detailed_results   # Clusters
        assert "Frontend" in detailed_results
        assert "Backend" in detailed_results


# Integration test for complete conversation flow
@pytest.mark.asyncio
async def test_complete_conversation_flow():
    """Test complete conversation flow from start to finish."""
    agent = ConversationalAgent()
    
    try:
        # 1. Create session
        session_id = await agent.create_session()
        assert session_id in agent.sessions
        
        # 2. Send help request
        help_response = await agent.process_message(session_id, "帮助")
        assert "您好" in help_response.message
        
        # 3. Request data import
        import_response = await agent.process_message(session_id, "导入数据")
        assert "上传" in import_response.message
        assert import_response.action_required == "file_upload"
        
        # 4. Simulate file upload context
        agent.sessions[session_id].context["file_uploaded"] = True
        agent.sessions[session_id].context["uploaded_file_path"] = "test.csv"
        
        # 5. Process "uploaded file" message
        file_response = await agent.process_message(session_id, "已上传文件")
        # Should indicate need for actual file processing
        
        # 6. Test parameter adjustment
        param_response = await agent.process_message(session_id, "最小聚类大小设为12")
        assert "调整参数" in param_response.message
        assert agent.sessions[session_id].context.get("min_cluster_size") == 12
        
    finally:
        await agent.close()