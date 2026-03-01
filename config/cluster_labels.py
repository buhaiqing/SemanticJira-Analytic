"""
Jira任务分类标签配置文件
用于SemanticJira-Analytic项目的任务语义分类
"""

import json

with open("./cluster_labels.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    # 任务分类标签配置
    CLUSTER_LABELS = data["cluster_labels"]


# 反向映射：关键词到标签
KEYWORD_TO_LABEL = {}
for label, config in CLUSTER_LABELS.items():
    for keyword in config["keywords"]:
        if keyword not in KEYWORD_TO_LABEL:
            KEYWORD_TO_LABEL[keyword] = []
        KEYWORD_TO_LABEL[keyword].append(label)

# 配置元数据
CONFIG_METADATA = {
    "created_at": "2026-03-01T09:59:00+08:00",
    "version": "1.0",
    "description": "Jira任务分类标签配置文件，用于SemanticJira-Analytic项目的任务语义分类",
}


def get_label_by_keywords(text):
    """
    根据文本内容匹配最合适的分类标签

    Args:
        text (str): 要分类的文本内容

    Returns:
        str: 匹配到的分类标签，如果没有匹配则返回None
    """
    if not text:
        return None

    text_lower = text.lower()
    matched_labels = set()

    for keyword, labels in KEYWORD_TO_LABEL.items():
        if keyword.lower() in text_lower:
            matched_labels.update(labels)

    # 如果匹配到多个标签，返回第一个（可以根据需要调整优先级）
    return list(matched_labels)[0] if matched_labels else None


def get_all_labels():
    """
    获取所有可用的分类标签

    Returns:
        list: 所有分类标签的列表
    """
    return list(CLUSTER_LABELS.keys())


def get_label_description(label):
    """
    获取指定标签的详细描述

    Args:
        label (str): 分类标签

    Returns:
        str: 标签的详细描述，如果标签不存在则返回None
    """
    return CLUSTER_LABELS.get(label, {}).get("description")


# 用于测试的示例数据
if __name__ == "__main__":
    # 测试关键词匹配功能
    test_cases = [
        "处理数据库连接问题",
        "SSL证书即将到期",
        "部署新的docker容器",
        "调整k8s资源配置",
        "开发监控API工具",
    ]

    print("=== 任务分类标签测试 ===")
    for case in test_cases:
        label = get_label_by_keywords(case)
        print(f"文本: {case}")
        print(f"分类: {label}")
        if label:
            print(f"描述: {get_label_description(label)}")
        print("-" * 50)
