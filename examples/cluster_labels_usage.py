"""
任务分类标签使用示例
演示如何在SemanticJira-Analytic项目中使用配置文件
"""

import json
import sys
from pathlib import Path

# 添加config目录到路径
config_path = Path(__file__).parent.parent / "config"
sys.path.append(str(config_path))

try:
    from cluster_labels import (
        CLUSTER_LABELS, 
        get_label_by_keywords, 
        get_all_labels, 
        get_label_description
    )
except ImportError:
    print("请确保cluster_labels.py文件在config目录中")
    sys.exit(1)

def load_json_config():
    """加载JSON格式的配置文件"""
    config_path = Path(__file__).parent.parent / "config" / "cluster_labels.json"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"配置文件未找到: {config_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return None

def demonstrate_usage():
    """演示配置文件的各种使用方式"""
    
    print("=== SemanticJira-Analytic 任务分类标签使用演示 ===\n")
    
    # 1. 显示所有可用标签
    print("1. 所有可用的分类标签:")
    print("-" * 50)
    all_labels = get_all_labels()
    for i, label in enumerate(all_labels, 1):
        print(f"{i:2d}. {label}")
    print()
    
    # 2. 显示标签详细信息
    print("2. 标签详细信息示例:")
    print("-" * 50)
    sample_labels = ["数据库运维", "安全管理", "应用部署", "故障排查"]
    for label in sample_labels:
        description = get_label_description(label)
        keywords = CLUSTER_LABELS[label]["keywords"]
        print(f"标签: {label}")
        print(f"描述: {description}")
        print(f"关键词: {', '.join(keywords)}")
        print()
    
    # 3. 文本分类演示
    print("3. 文本自动分类演示:")
    print("-" * 50)
    test_texts = [
        "今天需要处理泡泡玛特数据库的巡检工作",
        "qianfan123的ssl证书快要过期了，需要及时更新",
        "为dly-portal项目准备生产环境的部署方案",
        "海鼎测试环境的k8s cpu配置需要调整",
        "开发一个新的JIRA changelog API接口",
        "梅尼项目的custom组件出现报错，需要紧急排查",
        "设置诚信志远生产环境的CPU使用率监控告警",
        "为客户酒廷1990制定云资源降配方案"
    ]
    
    for text in test_texts:
        label = get_label_by_keywords(text)
        print(f"文本: {text}")
        print(f"分类: {label if label else '未匹配到合适标签'}")
        if label:
            print(f"描述: {get_label_description(label)}")
        print("-" * 80)
    
    # 4. JSON配置文件使用
    print("\n4. JSON配置文件使用:")
    print("-" * 50)
    json_config = load_json_config()
    if json_config:
        print(f"配置文件版本: {json_config['metadata']['version']}")
        print(f"创建时间: {json_config['metadata']['created_at']}")
        print(f"标签总数: {len(json_config['cluster_labels'])}")
        
        # 显示JSON中的前3个标签作为示例
        print("\n前3个标签示例:")
        for i, (label, config) in enumerate(list(json_config['cluster_labels'].items())[:3]):
            print(f"{i+1}. {label}")
            print(f"   关键词: {', '.join(config['keywords'])}")
            print(f"   描述: {config['description']}")
            print()

def integrate_with_existing_workflow():
    """演示如何与现有工作流集成"""
    
    print("=== 与现有工作流集成示例 ===\n")
    
    # 模拟从CSV文件读取任务数据
    sample_tasks = [
        {
            "issue_id": "DOPS-001",
            "summary": "泡泡玛特数据库巡检",
            "description": "执行泡泡玛特数据库周巡检工作"
        },
        {
            "issue_id": "DOPS-002", 
            "summary": "qianfan123 ssl证书更新",
            "description": "处理qianfan123 ssl证书到期问题"
        },
        {
            "issue_id": "DOPS-003",
            "summary": "dly-portal生产部署",
            "description": "为dly-portal项目提供部署作业支持"
        }
    ]
    
    print("原始任务数据:")
    print("-" * 50)
    for task in sample_tasks:
        print(f"Issue ID: {task['issue_id']}")
        print(f"Summary: {task['summary']}")
        print(f"Description: {task['description']}")
        print()
    
    print("添加分类标签后:")
    print("-" * 50)
    for task in sample_tasks:
        # 合并summary和description进行分类
        full_text = f"{task['summary']} {task['description']}"
        label = get_label_by_keywords(full_text)
        
        print(f"Issue ID: {task['issue_id']}")
        print(f"Summary: {task['summary']}")
        print(f"Description: {task['description']}")
        print(f"Cluster Label: {label if label else '未分类'}")
        if label:
            print(f"Label Description: {get_label_description(label)}")
        print("-" * 50)

if __name__ == "__main__":
    demonstrate_usage()
    print("\n" + "="*80 + "\n")
    integrate_with_existing_workflow()