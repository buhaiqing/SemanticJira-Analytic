"""
聚类配置使用示例
演示如何在SemanticJira-Analytic项目中使用聚类配置
"""

import sys
from pathlib import Path

# 添加config目录到路径
sys.path.append(str(Path(__file__).parent.parent / "config"))

try:
    from clustering_config import (
        ClusteringParameters,
        DEFAULT_CLUSTERING_CONFIG,
        CLUSTERING_PRESETS,
        get_preset_config,
        suggest_parameters,
        validate_clustering_config,
        load_clustering_config_from_env
    )
except ImportError as e:
    print(f"导入配置模块失败: {e}")
    sys.exit(1)

def demonstrate_basic_usage():
    """演示基本配置使用"""
    print("=== 基本配置使用演示 ===\n")
    
    # 1. 使用默认配置
    print("1. 默认配置:")
    print(f"   算法: {DEFAULT_CLUSTERING_CONFIG.algorithm}")
    print(f"   最小聚类大小: {DEFAULT_CLUSTERING_CONFIG.min_cluster_size}")
    print(f"   Epsilon: {DEFAULT_CLUSTERING_CONFIG.cluster_selection_epsilon}")
    print(f"   最大聚类数: {DEFAULT_CLUSTERING_CONFIG.max_clusters}")
    print()
    
    # 2. 创建自定义配置
    print("2. 自定义配置:")
    custom_config = ClusteringParameters(
        min_cluster_size=15,
        cluster_selection_epsilon=0.3,
        algorithm="hdbscan",
        max_clusters=25
    )
    print(f"   自定义最小聚类大小: {custom_config.min_cluster_size}")
    print(f"   自定义Epsilon: {custom_config.cluster_selection_epsilon}")
    print()
    
    # 3. 使用预设配置
    print("3. 预设配置模板:")
    for preset_name in ["conservative", "balanced", "aggressive", "kmeans_focused"]:
        try:
            preset = get_preset_config(preset_name)
            print(f"   {preset_name}: min_size={preset.min_cluster_size}, "
                  f"epsilon={preset.cluster_selection_epsilon}, "
                  f"algorithm={preset.algorithm}")
        except ValueError as e:
            print(f"   {preset_name}: 错误 - {e}")
    print()

def demonstrate_parameter_suggestion():
    """演示参数建议功能"""
    print("=== 参数建议演示 ===\n")
    
    test_task_counts = [10, 25, 75, 200, 500, 1000]
    
    print("根据任务数量建议的参数:")
    print("-" * 60)
    for count in test_task_counts:
        suggested = suggest_parameters(count)
        print(f"任务数: {count:4d} -> "
              f"min_size: {suggested.min_cluster_size:2d}, "
              f"epsilon: {suggested.cluster_selection_epsilon:.1f}, "
              f"max_clusters: {suggested.max_clusters:2d}")
    print()

def demonstrate_validation():
    """演示配置验证功能"""
    print("=== 配置验证演示 ===\n")
    
    # 创建一些测试配置
    test_configs = [
        ("正常配置", ClusteringParameters(min_cluster_size=10, cluster_selection_epsilon=0.5)),
        ("过小的聚类大小", ClusteringParameters(min_cluster_size=2, cluster_selection_epsilon=0.5)),
        ("过大的epsilon", ClusteringParameters(min_cluster_size=10, cluster_selection_epsilon=2.0)),
        ("K-Means聚类数过少", ClusteringParameters(min_cluster_size=10, algorithm="kmeans", max_clusters=2))
    ]
    
    for name, config in test_configs:
        print(f"配置: {name}")
        # 临时添加任务数量用于验证
        config._task_count = 100
        validation_result = validate_clustering_config(config)
        
        if validation_result["valid"]:
            print("  ✓ 配置有效")
        else:
            print("  ✗ 配置存在问题:")
            for issue in validation_result["issues"]:
                print(f"    - {issue}")
        
        if validation_result["suggestions"]:
            print("  建议:")
            for suggestion in validation_result["suggestions"]:
                print(f"    - {suggestion}")
        print()

def demonstrate_env_loading():
    """演示从环境变量加载配置"""
    print("=== 环境变量配置演示 ===\n")
    
    # 注意：这会使用实际的环境变量
    try:
        env_config = load_clustering_config_from_env()
        print("从环境变量加载的配置:")
        print(f"  最小聚类大小: {env_config.min_cluster_size}")
        print(f"  Epsilon: {env_config.cluster_selection_epsilon}")
        print(f"  最大聚类数: {env_config.max_clusters}")
        print(f"  算法: {env_config.algorithm}")
    except Exception as e:
        print(f"加载环境变量配置失败: {e}")
    print()

def demonstrate_integration_example():
    """演示与实际工作流的集成"""
    print("=== 实际工作流集成示例 ===\n")
    
    # 模拟不同类型的任务数据集
    scenarios = [
        {
            "name": "小型团队日常任务",
            "task_count": 30,
            "description": "一个小型开发团队一个月的任务量"
        },
        {
            "name": "中型企业项目",
            "task_count": 150,
            "description": "中等规模企业的季度任务量"
        },
        {
            "name": "大型组织年度任务",
            "task_count": 800,
            "description": "大型组织一年的任务总量"
        }
    ]
    
    print("不同场景的聚类配置建议:")
    print("=" * 80)
    
    for scenario in scenarios:
        print(f"\n场景: {scenario['name']}")
        print(f"任务数量: {scenario['task_count']}")
        print(f"描述: {scenario['description']}")
        
        # 获取建议配置
        suggested_config = suggest_parameters(scenario['task_count'])
        
        print(f"\n推荐配置:")
        print(f"  • 最小聚类大小: {suggested_config.min_cluster_size}")
        print(f"  • 聚类选择epsilon: {suggested_config.cluster_selection_epsilon}")
        print(f"  • 算法: {suggested_config.algorithm}")
        print(f"  • 预期聚类范围: {suggested_config.expected_cluster_range[0]}-{suggested_config.expected_cluster_range[1]}个")
        
        # 验证配置
        suggested_config._task_count = scenario['task_count']
        validation = validate_clustering_config(suggested_config)
        
        if validation["valid"]:
            print("  ✓ 配置验证通过")
        else:
            print("  ⚠ 配置需要注意:")
            for issue in validation["issues"]:
                print(f"    - {issue}")

def main():
    """主函数 - 运行所有演示"""
    print("🎯 SemanticJira-Analytic 聚类配置使用演示\n")
    print("=" * 80)
    
    demonstrate_basic_usage()
    demonstrate_parameter_suggestion()
    demonstrate_validation()
    demonstrate_env_loading()
    demonstrate_integration_example()
    
    print("\n" + "=" * 80)
    print("💡 使用建议:")
    print("1. 根据数据规模选择合适的预设配置")
    print("2. 使用suggest_parameters()函数获得针对特定任务数量的建议")
    print("3. 通过validate_clustering_config()验证配置的合理性")
    print("4. 在.env文件中设置环境变量来自定义默认行为")
    print("5. 对于生产环境，建议先用小数据集测试配置效果")

if __name__ == "__main__":
    main()