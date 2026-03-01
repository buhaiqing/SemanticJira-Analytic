"""Simple conversational agent demo."""

import asyncio
import re
from typing import Dict, List
from datetime import datetime
from rich.console import Console

console = Console()


class SimpleConversationalAgent:
    """Simple demo conversational agent."""
    
    def __init__(self):
        self.context: Dict = {}
        self.conversation_history: List[Dict] = []
        
        # Simple intent patterns
        self.intents = {
            "greeting": [r"你好", r"hello", r"hi"],
            "help": [r"帮助", r"help", r"怎么用"],
            "data_import": [r"导入", r"上传", r"数据", r"data"],
            "analysis": [r"分析", r"聚类", r"cluster"],
            "results": [r"结果", r"查看", r"显示"],
            "demo": [r"演示", r"demo"]
        }
    
    def recognize_intent(self, message: str) -> str:
        """Simple intent recognition."""
        message_lower = message.lower()
        
        for intent, patterns in self.intents.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return intent
        
        return "unknown"
    
    def generate_response(self, intent: str, message: str) -> str:
        """Generate response based on intent."""
        if intent == "greeting":
            return "您好！我是Jira任务分析助手。我可以帮您分析任务数据。"
        
        elif intent == "help":
            return """我可以帮您：
1. 📤 导入Jira任务数据
2. 🧠 进行智能聚类分析  
3. 📊 查看和导出分析结果
4. 🎯 演示完整工作流程

请告诉我您想做什么！"""
        
        elif intent == "data_import":
            self.context["awaiting_file"] = True
            return "请提供您的Jira导出文件路径，我将帮您处理数据。"
        
        elif intent == "analysis":
            if self.context.get("data_loaded"):
                return "正在分析数据...分析完成！发现了3个主要任务类别。"
            else:
                return "请先导入数据文件，然后我才能进行分析。"
        
        elif intent == "results":
            if self.context.get("analysis_done"):
                return """📊 分析结果：
• 类别0：认证相关任务 (2个)
• 类别1：功能开发任务 (2个)  
• 类别2：优化改进任务 (1个)

您可以要求导出详细报告。"""
            else:
                return "还没有分析结果。请先导入数据并执行分析。"
        
        elif intent == "demo":
            return self.run_demo()
        
        else:
            return "我不太明白您的意思。您可以问'帮助'查看我能做什么。"
    
    def run_demo(self) -> str:
        """Run a quick demo."""
        demo_response = """🚀 快速演示：

1. 模拟数据导入完成 ✓
2. 执行聚类分析 ✓  
3. 生成分析结果 ✓

📊 演示结果：
• 总任务数：5个
• 识别类别：3个
• 处理状态：完成

这就是完整的分析流程！"""
        
        self.context["data_loaded"] = True
        self.context["analysis_done"] = True
        return demo_response
    
    def process_message(self, message: str) -> str:
        """Process user message and return response."""
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now()
        })
        
        # Recognize intent
        intent = self.recognize_intent(message)
        
        # Generate response
        response = self.generate_response(intent, message)
        
        # Add response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now()
        })
        
        return response


async def run_chat_demo():
    """Run interactive chat demo."""
    agent = SimpleConversationalAgent()
    
    console.print("[bold blue]🤖 启动对话式Agent演示...[/bold blue]")
    console.print("[green]✅ Agent已就绪！输入'帮助'查看功能，输入'演示'看快速演示。[/green]\n")
    
    while True:
        try:
            # Get user input
            user_input = console.input("[bold cyan]您:[/bold cyan] ").strip()
            
            if user_input.lower() in ['退出', 'quit', 'exit', '再见']:
                console.print("[yellow]👋 再见！[/yellow]")
                break
            
            if not user_input:
                continue
            
            # Process message
            with console.status("[bold green]思考中...") as status:
                response = agent.process_message(user_input)
            
            # Display response
            console.print(f"[bold magenta]助手:[/bold magenta] {response}\n")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]👋 再见！[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]❌ 错误: {e}[/red]")


if __name__ == "__main__":
    asyncio.run(run_chat_demo())