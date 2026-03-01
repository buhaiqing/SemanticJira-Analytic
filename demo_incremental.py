"""Demonstration of incremental update functionality."""

import json
import pandas as pd
from datetime import datetime, timedelta
from app.core.incremental import IncrementalUpdateHandler
from app.core.preprocessing import DataPreprocessor
from app.models import JiraTask, ProcessedTask, TaskStatus

def demonstrate_incremental_updates():
    """Demonstrate the incremental update functionality."""
    
    print("🔄 演示增量更新功能")
    print("=" * 50)
    
    # Create incremental update handler
    handler = IncrementalUpdateHandler()
    
    # Step 1: Load initial data (simulate existing database)
    print("\n📥 步骤1: 加载初始数据")
    initial_data = create_sample_data(num_tasks=5, base_date=datetime(2024, 1, 15))
    preprocessor = DataPreprocessor()
    initial_tasks = [preprocessor._row_to_task(row) for _, row in initial_data.iterrows()]
    initial_tasks = [task for task in initial_tasks if task is not None]
    
    # Process initial tasks
    processed_initial = preprocessor.preprocess_tasks(initial_tasks)
    handler.load_existing_data(processed_initial)
    
    print(f"✅ 加载了 {len(processed_initial)} 个初始任务")
    
    # Step 2: Process new batch with some updates
    print("\n📥 步骤2: 处理新的数据批次")
    new_batch_data = create_updated_sample_data()
    new_tasks = [preprocessor._row_to_task(row) for _, row in new_batch_data.iterrows()]
    new_tasks = [task for task in new_tasks if task is not None]
    
    print(f"📊 新批次包含 {len(new_tasks)} 个任务")
    
    # Step 3: Categorize updates
    print("\n🔍 步骤3: 分类更新类型")
    new_tasks_only, updated_task_pairs = handler.categorize_updates(new_tasks)
    
    print(f"🆕 新任务: {len(new_tasks_only)} 个")
    print(f"🔄 更新任务: {len(updated_task_pairs)} 个")
    
    # Show details of updated tasks
    if updated_task_pairs:
        print("\n📝 更新任务详情:")
        for new_task, existing_task in updated_task_pairs:
            print(f"  • {new_task.issue_id}: "
                  f"创建时间 {new_task.created_at.strftime('%Y-%m-%d')} → "
                  f"更新时间 {new_task.updated_at.strftime('%Y-%m-%d %H:%M')}")
    
    # Step 4: Process and merge
    print("\n⚙️  步骤4: 处理并合并数据")
    processed_new = preprocessor.preprocess_tasks(new_tasks_only)
    
    # Process updated tasks
    processed_updated_pairs = []
    for new_task, existing_task in updated_task_pairs:
        processed_new_task = preprocessor.preprocess_tasks([new_task])[0]
        processed_updated_pairs.append((processed_new_task, existing_task))
    
    # Merge all updates
    merged_tasks = handler.merge_updates(processed_new, processed_updated_pairs)
    
    print(f"✅ 合并完成，总计 {len(merged_tasks)} 个任务")
    
    # Step 5: Show statistics
    print("\n📈 步骤5: 更新统计信息")
    stats = handler.get_update_statistics()
    print(f"📊 总任务数: {stats['total_existing_tasks']}")
    
    # Step 6: Export results
    print("\n💾 步骤6: 导出结果")
    export_filename = "incremental_demo_results.json"
    preprocessor.save_processed_data(merged_tasks, export_filename)
    print(f"✅ 结果已保存到: {export_filename}")
    
    return merged_tasks

def create_sample_data(num_tasks=5, base_date=None):
    """Create sample Jira task data."""
    if base_date is None:
        base_date = datetime.now()
    
    data = {
        'issue_id': [f'TASK-{i+1:03d}' for i in range(num_tasks)],
        'summary': [
            '修复用户登录认证问题',
            '添加密码重置功能', 
            '更新用户个人资料页面',
            '实现数据库查询优化',
            '添加API限流机制'
        ][:num_tasks],
        'description': [
            '用户在修改密码后无法登录，需要调查认证流程并修复相关bug',
            '实现通过邮箱验证链接重置用户密码的功能，包括邮件发送和验证逻辑',
            '重新设计用户个人资料界面，使用现代化UI组件提升用户体验',
            '优化慢查询SQL语句，添加适当的索引以提高数据库性能',
            '实现API请求频率限制，防止恶意请求和滥用'
        ][:num_tasks],
        'created_at': [(base_date + timedelta(days=i)).isoformat() for i in range(num_tasks)],
        'updated_at': [(base_date + timedelta(days=i)).isoformat() for i in range(num_tasks)],
        'status': ['To Do', 'In Progress', 'Done', 'To Do', 'In Progress'][:num_tasks]
    }
    
    return pd.DataFrame(data)

def create_updated_sample_data():
    """Create sample data with some updates for demonstration."""
    # Base data (some existing, some new)
    base_date = datetime(2024, 1, 20)
    
    data = {
        'issue_id': ['TASK-001', 'TASK-002', 'TASK-006', 'TASK-007', 'TASK-016'],
        'summary': [
            '修复用户登录认证问题',  # Existing task - updated
            '添加密码重置功能',      # Existing task - no change
            '集成第三方支付网关',    # New task
            '实现用户行为日志记录',  # Existing task - updated
            '添加实时聊天功能'       # New task
        ],
        'description': [
            '用户在修改密码后无法登录，已修复认证流程中的安全漏洞',  # Updated description
            '实现通过邮箱验证链接重置用户密码的功能，包括邮件发送和验证逻辑',  # Same
            '集成支付宝和微信支付功能，支持多种支付方式',  # New task
            '记录用户关键操作日志，便于审计和问题追踪，增加实时监控',  # Updated description
            '实现实时聊天功能，支持用户间即时通讯'  # New task
        ],
        'created_at': [
            '2024-01-15T09:30:00Z',  # Original creation
            '2024-01-16T14:20:00Z',  # Original creation
            base_date.isoformat(),   # New task creation
            '2024-01-21T13:45:00Z',  # Original creation
            base_date.isoformat()    # New task creation
        ],
        'updated_at': [
            '2024-01-20T10:30:00Z',  # Updated timestamp
            '2024-01-16T14:20:00Z',  # Same as creation (no update)
            base_date.isoformat(),   # New task
            '2024-01-20T15:30:00Z',  # Updated timestamp
            base_date.isoformat()    # New task
        ],
        'status': ['In Progress', 'In Progress', 'To Do', 'Done', 'To Do']
    }
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    demonstrate_incremental_updates()