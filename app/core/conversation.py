"""Conversational Agent Skill for natural language data analysis."""

import asyncio
import uuid
import re
from typing import Dict, Any, Optional
from datetime import datetime
import logging
from dataclasses import dataclass

from app.models import ChatMessage, ChatSession
from app.core.preprocessing import DataPreprocessor
from app.core.embedding import VectorEmbedder
from app.core.clustering import TaskClusterer, ClusteringConfig

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Response from the conversational agent."""
    message: str
    action_required: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    context_updates: Optional[Dict[str, Any]] = None


class ConversationalAgent:
    """Natural language conversational agent for Jira task analysis."""
    
    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}
        self.preprocessor = DataPreprocessor()
        self.embedder: Optional[VectorEmbedder] = None
        self.clusterer: Optional[TaskClusterer] = None
        
        # Intent recognition patterns
        self.intent_patterns = {
            "data_import": [
                r"导入.*数据",
                r"上传.*文件",
                r"分析.*任务",
                r"处理.*jira",
                r"load.*data"
            ],
            "clustering": [
                r"聚类.*分析",
                r"分类.*任务",
                r"cluster.*analysis",
                r"分组.*任务"
            ],
            "results": [
                r"查看.*结果",
                r"显示.*分析",
                r"show.*results",
                r"导出.*结果"
            ],
            "parameters": [
                r"调整.*参数",
                r"修改.*设置",
                r"change.*parameters"
            ],
            "help": [
                r"帮助",
                r"怎么用",
                r"help",
                r"如何.*使用"
            ]
        }
    
    async def create_session(self) -> str:
        """Create a new chat session."""
        session_id = str(uuid.uuid4())
        session = ChatSession(session_id=session_id)
        self.sessions[session_id] = session
        logger.info(f"Created new chat session: {session_id}")
        return session_id
    
    async def process_message(self, session_id: str, user_message: str) -> AgentResponse:
        """Process user message and generate appropriate response."""
        # Get or create session
        if session_id not in self.sessions:
            session_id = await self.create_session()
        
        session = self.sessions[session_id]
        
        # Add user message to session
        user_msg = ChatMessage(role="user", content=user_message)
        session.messages.append(user_msg)
        session.last_activity = datetime.now()
        
        try:
            # Recognize intent
            intent = self._recognize_intent(user_message)
            logger.info(f"Recognized intent: {intent}")
            
            # Generate response based on intent
            response = await self._generate_response(intent, user_message, session)
            
            # Add assistant message to session
            assistant_msg = ChatMessage(role="assistant", content=response.message)
            session.messages.append(assistant_msg)
            
            # Update context if needed
            if response.context_updates:
                session.context.update(response.context_updates)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_response = AgentResponse(
                message="抱歉，我在处理您的请求时遇到了问题。请稍后再试或尝试重新表述您的需求。"
            )
            # Add error message to session
            error_msg = ChatMessage(role="assistant", content=error_response.message)
            session.messages.append(error_msg)
            return error_response
    
    def _recognize_intent(self, message: str) -> str:
        """Recognize user intent from message."""
        message_lower = message.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return intent
        
        return "unknown"
    
    async def _generate_response(self, intent: str, message: str, session: ChatSession) -> AgentResponse:
        """Generate appropriate response based on intent."""
        
        if intent == "help":
            return self._handle_help_request()
        
        elif intent == "data_import":
            return await self._handle_data_import(message, session)
        
        elif intent == "clustering":
            return await self._handle_clustering_request(message, session)
        
        elif intent == "results":
            return await self._handle_results_request(session)
        
        elif intent == "parameters":
            return await self._handle_parameters_request(message, session)
        
        else:
            return self._handle_unknown_intent(message)
    
    def _handle_help_request(self) -> AgentResponse:
        """Handle help requests."""
        help_text = """您好！我是您的Jira任务分析助手。我可以帮您：

1. 📤 **数据导入** - 上传Jira导出的CSV文件进行分析
2. 🧠 **智能分析** - 自动对任务进行聚类分类
3. 📊 **结果查看** - 查看和导出分析结果
4. ⚙️ **参数调整** - 根据需要调整分析参数

您可以直接告诉我您想做什么，比如：
- "请帮我分析这个月的Jira任务"
- "我想查看聚类结果"
- "调整聚类参数"

随时都可以问我任何问题！"""
        
        return AgentResponse(message=help_text)
    
    async def _handle_data_import(self, message: str, session: ChatSession) -> AgentResponse:
        """Handle data import requests."""
        # Check if file attachment is mentioned
        if "文件" in message or "上传" in message or "导入" in message:
            if "已上传" in message or session.context.get("file_uploaded"):
                # File already uploaded, proceed with processing
                return await self._process_uploaded_file(session)
            else:
                # Request file upload
                return AgentResponse(
                    message="请上传您的Jira导出文件（CSV格式）。您可以直接拖拽文件到这里。",
                    action_required="file_upload"
                )
        else:
            return AgentResponse(
                message="我理解您想要导入数据进行分析。请上传Jira导出的CSV文件，我会帮您处理。"
            )
    
    async def _process_uploaded_file(self, session: ChatSession) -> AgentResponse:
        """Process uploaded file."""
        file_path = session.context.get("uploaded_file_path")
        if not file_path:
            return AgentResponse(
                message="我没有找到上传的文件。请先上传Jira导出的CSV文件。"
            )
        
        try:
            # Load and preprocess data
            with asyncio.Lock():  # Ensure thread safety
                tasks = self.preprocessor.load_data(file_path)
                processed_tasks = self.preprocessor.preprocess_tasks(tasks)
            
            # Store processed tasks in session context
            session.context["processed_tasks"] = processed_tasks
            session.context["total_tasks"] = len(processed_tasks)
            
            return AgentResponse(
                message=f"✅ 已成功加载并预处理 {len(processed_tasks)} 个任务。现在可以进行聚类分析了！",
                context_updates={"data_processed": True}
            )
            
        except Exception as e:
            logger.error(f"Error processing uploaded file: {e}")
            return AgentResponse(
                message=f"处理文件时出现错误：{str(e)}。请检查文件格式是否正确。"
            )
    
    async def _handle_clustering_request(self, message: str, session: ChatSession) -> AgentResponse:
        """Handle clustering analysis requests."""
        if not session.context.get("data_processed"):
            return AgentResponse(
                message="请先上传并处理数据文件，然后我才能进行聚类分析。"
            )
        
        try:
            # Get processed tasks
            processed_tasks = session.context.get("processed_tasks", [])
            if not processed_tasks:
                return AgentResponse(message="没有找到待分析的任务数据。")
            
            # Initialize embedder if not already done
            if self.embedder is None:
                self.embedder = VectorEmbedder()
                await self.embedder.initialize_model()
            
            # Generate embeddings
            embedded_tasks = await self.embedder.embed_tasks(processed_tasks)
            
            # Configure clustering based on user preferences or defaults
            min_size = self._extract_parameter(message, "min_size", 10)
            algorithm = "hdbscan"  # Default algorithm
            
            config = ClusteringConfig(
                algorithm=algorithm,
                min_cluster_size=min_size,
                cluster_selection_epsilon=0.5
            )
            
            # Perform clustering
            if self.clusterer is None:
                self.clusterer = TaskClusterer(config)
            
            result = await self.clusterer.cluster_tasks(embedded_tasks)
            
            # Store results
            session.context["clustering_result"] = result
            session.context["embedded_tasks"] = embedded_tasks
            
            # Generate summary response
            summary = self._generate_clustering_summary(result)
            
            return AgentResponse(
                message=summary,
                context_updates={"analysis_completed": True}
            )
            
        except Exception as e:
            logger.error(f"Error during clustering: {e}")
            return AgentResponse(
                message=f"聚类分析过程中出现错误：{str(e)}"
            )
    
    async def _handle_results_request(self, session: ChatSession) -> AgentResponse:
        """Handle results viewing/export requests."""
        if not session.context.get("analysis_completed"):
            return AgentResponse(
                message="还没有完成分析。请先上传数据并执行聚类分析。"
            )
        
        result = session.context.get("clustering_result")
        if not result:
            return AgentResponse(message="没有找到分析结果。")
        
        # Generate detailed results summary
        detailed_summary = self._generate_detailed_results(result)
        
        return AgentResponse(
            message=detailed_summary,
            action_required="export_options"
        )
    
    async def _handle_parameters_request(self, message: str, session: ChatSession) -> AgentResponse:
        """Handle parameter adjustment requests."""
        # Extract parameters from message
        min_size = self._extract_parameter(message, "min_size")
        epsilon = self._extract_parameter(message, "epsilon")
        
        adjustments = []
        if min_size:
            session.context["min_cluster_size"] = min_size
            adjustments.append(f"最小聚类大小设为 {min_size}")
        
        if epsilon:
            session.context["cluster_epsilon"] = epsilon
            adjustments.append(f"聚类epsilon设为 {epsilon}")
        
        if adjustments:
            return AgentResponse(
                message=f"已调整参数：{'，'.join(adjustments)}。下次分析时将使用新参数。"
            )
        else:
            return AgentResponse(
                message="请告诉我您想要调整什么参数，比如：'最小聚类大小设为15' 或 'epsilon设为0.3'"
            )
    
    def _handle_unknown_intent(self, message: str) -> AgentResponse:
        """Handle unrecognized intents."""
        return AgentResponse(
            message="我不太明白您的意思。您可以问我关于数据导入、聚类分析、查看结果或调整参数的问题。输入'帮助'可以查看所有可用功能。"
        )
    
    def _extract_parameter(self, message: str, param_name: str, default=None):
        """Extract parameter value from message."""
        patterns = {
            "min_size": [r"最小.*?(\d+)", r"min.*?size.*?(\d+)", r"(\d+).*?最小"],
            "epsilon": [r"epsilon.*?(\d+\.?\d*)", r"ε.*?(\d+\.?\d*)"]
        }
        
        if param_name in patterns:
            for pattern in patterns[param_name]:
                match = re.search(pattern, message, re.IGNORECASE)
                if match:
                    value = match.group(1)
                    try:
                        return int(value) if param_name == "min_size" else float(value)
                    except ValueError:
                        continue
        
        return default
    
    def _generate_clustering_summary(self, result) -> str:
        """Generate summary of clustering results."""
        summary = f"""📊 聚类分析完成！

📈 **总体统计**：
• 总任务数：{result.total_tasks}
• 识别聚类数：{result.clusters_found}
• 处理耗时：{result.processing_time:.2f}秒

🎯 **主要发现**：
"""
        
        # Sort clusters by size
        sorted_clusters = sorted(
            result.cluster_details.items(),
            key=lambda x: x[1]["size"],
            reverse=True
        )
        
        for i, (cluster_id, details) in enumerate(sorted_clusters[:3]):
            if cluster_id == -1:  # Noise cluster
                summary += f"• 噪声点：{details['size']} 个任务\n"
            else:
                confidence = details['avg_confidence']
                summary += f"• 聚类 {cluster_id}：{details['size']} 个任务 (置信度: {confidence:.2f})\n"
        
        summary += "\n💡 您可以要求查看详细结果或导出分析报告。"
        return summary
    
    def _generate_detailed_results(self, result) -> str:
        """Generate detailed results description."""
        detailed = f"""📋 **详细分析报告**

📊 **分析概览**：
- 总处理任务：{result.total_tasks} 个
- 识别聚类数量：{result.clusters_found} 个
- 分析处理时间：{result.processing_time:.2f} 秒
- 报告生成时间：{result.generated_at.strftime('%Y-%m-%d %H:%M:%S')}

🔍 **聚类详情**：
"""
        
        sorted_clusters = sorted(
            result.cluster_details.items(),
            key=lambda x: x[1]["size"],
            reverse=True
        )
        
        for cluster_id, details in sorted_clusters:
            if cluster_id == -1:
                detailed += f"\n🚫 **噪声/未分类任务** ({details['size']} 个)\n"
                detailed += "   描述：无法归类到明确聚类的任务\n"
            else:
                detailed += f"\n🎯 **聚类 {cluster_id}** ({details['size']} 个任务)\n"
                detailed += f"   平均置信度：{details['avg_confidence']:.3f}\n"
                if "sample_tasks" in details:
                    detailed += "   样本任务：\n"
                    for task in details["sample_tasks"]:
                        detailed += f"   • {task['summary']}\n"
        
        detailed += "\n📤 **导出选项**："
        detailed += "\n- CSV格式：包含聚类统计信息"
        detailed += "\n- JSON格式：完整分析结果"
        detailed += "\n- 详细报告：包含所有聚类详情"
        
        return detailed
    
    async def cleanup_session(self, session_id: str):
        """Clean up session resources."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleaned up session: {session_id}")
    
    async def close(self):
        """Clean up all resources."""
        if self.embedder:
            await self.embedder.close()
        if self.clusterer:
            await self.clusterer.close()
        self.sessions.clear()
        logger.info("Conversational agent closed")