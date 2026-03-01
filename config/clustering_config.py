"""
聚类分析配置文件
用于管理SemanticJira-Analytic项目的聚类相关参数
"""

import os
from typing import Dict, Any
from pydantic import BaseModel, Field

class ClusteringParameters(BaseModel):
    """聚类参数配置模型"""
    
    # HDBSCAN参数
    min_cluster_size: int = Field(
        default=10, 
        ge=2, 
        le=100,
        description="最小聚类大小 - 聚类中所需的最少样本数"
    )
    
    cluster_selection_epsilon: float = Field(
        default=0.5, 
        ge=0.1, 
        le=2.0,
        description="聚类选择epsilon - 控制聚类紧密度的参数"
    )
    
    # K-Means参数
    max_clusters: int = Field(
        default=20,
        ge=2,
        le=50,
        description="K-Means算法的最大聚类数"
    )
    
    # 通用参数
    algorithm: str = Field(
        default="hdbscan",
        pattern="^(hdbscan|kmeans)$",
        description="聚类算法选择"
    )
    
    # 预期聚类数量范围
    expected_cluster_range: tuple = Field(
        default=(3, 15),
        description="预期的聚类数量范围，用于参数调优参考"
    )

# 默认配置实例
DEFAULT_CLUSTERING_CONFIG = ClusteringParameters()

# 从环境变量加载配置
def load_clustering_config_from_env() -> ClusteringParameters:
    """
    从环境变量加载聚类配置
    
    Returns:
        ClusteringParameters: 聚类配置对象
    """
    config_dict = {
        "min_cluster_size": int(os.getenv("MIN_CLUSTER_SIZE", "10")),
        "cluster_selection_epsilon": float(os.getenv("CLUSTER_SELECTION_EPSILON", "0.5")),
        "max_clusters": int(os.getenv("MAX_CLUSTERS", "20")),
        "algorithm": os.getenv("CLUSTERING_ALGORITHM", "hdbscan")
    }
    
    return ClusteringParameters(**config_dict)

# 预定义的配置模板
CLUSTERING_PRESETS = {
    "conservative": ClusteringParameters(
        min_cluster_size=15,
        cluster_selection_epsilon=0.8,
        max_clusters=10,
        algorithm="hdbscan",
        expected_cluster_range=(2, 8)
    ),
    
    "balanced": ClusteringParameters(
        min_cluster_size=10,
        cluster_selection_epsilon=0.5,
        max_clusters=15,
        algorithm="hdbscan",
        expected_cluster_range=(3, 12)
    ),
    
    "aggressive": ClusteringParameters(
        min_cluster_size=5,
        cluster_selection_epsilon=0.3,
        max_clusters=25,
        algorithm="hdbscan",
        expected_cluster_range=(5, 20)
    ),
    
    "kmeans_focused": ClusteringParameters(
        min_cluster_size=8,
        cluster_selection_epsilon=0.5,
        max_clusters=12,
        algorithm="kmeans",
        expected_cluster_range=(4, 15)
    )
}

def get_preset_config(preset_name: str = "balanced") -> ClusteringParameters:
    """
    获取预定义的配置模板
    
    Args:
        preset_name (str): 配置模板名称
        
    Returns:
        ClusteringParameters: 对应的配置对象
        
    Raises:
        ValueError: 当preset_name不存在时
    """
    if preset_name not in CLUSTERING_PRESETS:
        available_presets = list(CLUSTERING_PRESETS.keys())
        raise ValueError(f"未知的配置模板 '{preset_name}'。可用模板: {available_presets}")
    
    return CLUSTERING_PRESETS[preset_name]

def suggest_parameters(task_count: int) -> ClusteringParameters:
    """
    根据任务数量建议合适的聚类参数
    
    Args:
        task_count (int): 任务总数
        
    Returns:
        ClusteringParameters: 建议的参数配置
    """
    if task_count < 20:
        return get_preset_config("conservative")
    elif task_count < 100:
        return get_preset_config("balanced")
    elif task_count < 500:
        return get_preset_config("aggressive")
    else:
        # 大数据集使用保守参数
        return ClusteringParameters(
            min_cluster_size=max(10, task_count // 50),
            cluster_selection_epsilon=0.6,
            max_clusters=min(30, task_count // 10),
            algorithm="hdbscan"
        )

# 配置验证和帮助信息
def validate_clustering_config(config: ClusteringParameters) -> Dict[str, Any]:
    """
    验证聚类配置的有效性
    
    Args:
        config (ClusteringParameters): 要验证的配置
        
    Returns:
        Dict[str, Any]: 验证结果，包含警告和建议
    """
    issues = []
    suggestions = []
    
    # 检查min_cluster_size相对于任务数量是否合理
    if hasattr(config, '_task_count'):
        if config.min_cluster_size > config._task_count // 2:
            issues.append("最小聚类大小过大，可能导致大部分数据被标记为噪声")
            suggestions.append(f"建议将min_cluster_size设置为{config._task_count // 10}左右")
    
    # 检查参数组合
    if config.algorithm == "kmeans" and config.max_clusters < 3:
        issues.append("K-Means算法的聚类数过少")
        suggestions.append("建议max_clusters至少为3")
    
    if config.cluster_selection_epsilon > 1.0:
        suggestions.append("较大的epsilon值可能导致过度合并聚类")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "suggestions": suggestions
    }

# 使用示例和文档
CLUSTERING_GUIDELINES = """
聚类参数配置指南:

1. min_cluster_size (最小聚类大小):
   - 控制形成聚类所需的最少样本数
   - 值越大，聚类越少但更稳定
   - 建议范围：任务总数的2%-10%

2. cluster_selection_epsilon (聚类选择epsilon):
   - 控制聚类的紧密程度
   - 值越小，聚类越精细
   - 建议范围：0.1-1.0

3. algorithm (算法选择):
   - hdbscan: 自动确定聚类数，处理噪声点好
   - kmeans: 需要指定聚类数，适合球形聚类

4. max_clusters (最大聚类数):
   - 仅对K-Means算法有效
   - 根据业务需求和数据特点设定

推荐配置策略:
- 小数据集(<50任务): 保守配置
- 中等数据集(50-200任务): 平衡配置  
- 大数据集(>200任务): 激进配置
"""

if __name__ == "__main__":
    # 演示配置使用
    print("=== 聚类配置演示 ===\n")
    
    # 显示默认配置
    print("默认配置:")
    print(f"  算法: {DEFAULT_CLUSTERING_CONFIG.algorithm}")
    print(f"  最小聚类大小: {DEFAULT_CLUSTERING_CONFIG.min_cluster_size}")
    print(f"  Epsilon: {DEFAULT_CLUSTERING_CONFIG.cluster_selection_epsilon}")
    print(f"  最大聚类数: {DEFAULT_CLUSTERING_CONFIG.max_clusters}")
    print()
    
    # 显示预设配置
    print("预设配置模板:")
    for name, config in CLUSTERING_PRESETS.items():
        print(f"  {name}: min_size={config.min_cluster_size}, epsilon={config.cluster_selection_epsilon}")
    print()
    
    # 演示参数建议
    test_sizes = [15, 50, 150, 300]
    print("根据任务数量的参数建议:")
    for size in test_sizes:
        suggested = suggest_parameters(size)
        print(f"  {size}个任务 -> min_size={suggested.min_cluster_size}, epsilon={suggested.cluster_selection_epsilon}")