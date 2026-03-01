# JiraVector-Analytics 智能代理开发指南

## 📋 文档导航

- [项目概述](#项目概述)
- [DevOps自动化操作](#devops自动化操作)
- [构建与测试](#构建与测试)
- [开发最佳实践与质量要求](#开发最佳实践与质量要求)
- [API设计规范](#api设计规范)
- [数据处理规范](#数据处理规范)
- [AI代码审查代理](#ai代码审查代理-code-review-agent)
- [对话式数据分析Agent Skill](#对话式数据分析agent-skill)
- [智能代理开发指南](#智能代理开发指南)
- [用户使用手册](#用户使用手册)
- [参考资源](#参考资源)

---

## 项目概述

JiraVector-Analytics是一个基于向量嵌入(LanceDB)和无监督聚类的智能任务分类系统，旨在解决Jira任务单语义模糊和分类混乱问题。采用FastAPI(Python)和React构建，通过先进的AI技术实现企业级项目管理数据治理。

### 技术栈

- **后端框架**: FastAPI + Python 3.10+ 集成 LanceDB
- **嵌入模型**: BGE-M3 (中文优化) 或 text-embedding-3-small
- **聚类算法**: HDBSCAN (主) + K-Means (备选)
- **前端框架**: React + Ant Design + AG-Grid + Redux Toolkit
- **部署方案**: Docker + Docker Compose，可选 Kubernetes
- **CLI工具**: 自定义 `jira-vector` 命令行工具

## DevOps自动化操作

### Makefile配置

项目根目录包含完整的Makefile，集成各阶段开发运维操作：

```
# Makefile - JiraVector-Analytics DevOps自动化

# 环境变量
export COMPOSE_FILE = docker-compose.yml\export PROJECT_NAME = jira-vector

# 开发环境
.PHONY: dev stop restart

dev:
	@echo "🚀 启动开发环境..."
	docker-compose up -d
	@echo "✅ 开发环境已启动，访问 http://localhost:3000"

stop:
	@echo "🛑 停止所有服务..."
	docker-compose down

restart: stop dev

# 测试相关
.PHONY: test test-unit test-integration coverage

test:
	@echo "🧪 运行所有测试..."
	docker-compose run --rm backend pytest tests/ -v

test-unit:
	@echo "🔬 运行单元测试..."
	docker-compose run --rm backend pytest tests/unit/ -v

test-integration:
	@echo "🔗 运行集成测试..."
	docker-compose run --rm backend pytest tests/integration/ -v

coverage:
	@echo "📊 生成测试覆盖率报告..."
	docker-compose run --rm backend pytest tests/ --cov=app --cov-report=html
	@echo "📋 覆盖率报告已生成: htmlcov/index.html"

# 代码质量
.PHONY: lint format security

lint:
	@echo "🔍 代码检查..."
	docker-compose run --rm backend ruff check .
	docker-compose run --rm frontend npm run lint

format:
	@echo "🎨 代码格式化..."
	docker-compose run --rm backend ruff format .
	docker-compose run --rm frontend npm run format

security:
	@echo "🛡️ 安全扫描..."
	docker-compose run --rm backend bandit -r app/
	docker-compose run --rm backend safety check

# 构建部署
.PHONY: build deploy backup

build:
	@echo "🏗️ 构建Docker镜像..."
	docker-compose build

deploy:
	@echo "🚀 部署到生产环境..."
	# 生产环境部署脚本	echo "TODO: 实现生产环境部署逻辑"

backup:
	@echo "💾 数据备份..."
	mkdir -p backups
	cp -r data/lancedb backups/lancedb_$(shell date +%Y%m%d_%H%M%S)
	cp .env backups/env_$(shell date +%Y%m%d_%H%M%S)
	@echo "✅ 备份完成"

# 清理维护
.PHONY: clean clean-data logs

clean:
	@echo "🧹 清理构建文件..."
	rm -rf build/ dist/ *.egg-info
	docker-compose down -v

clean-data:
	@echo "🗑️ 清理数据文件..."
	rm -rf data/lancedb/*

logs:
	@echo "📋 查看服务日志..."
	docker-compose logs -f

# 帮助信息
.PHONY: help

help:
	@echo "🎯 JiraVector-Analytics DevOps 命令帮助"
	@echo ""
	@echo "开发环境:"
	@echo "  make dev              启动开发环境"
	@echo "  make stop             停止所有服务"
	@echo "  make restart          重启服务"
	@echo ""
	@echo "测试相关:"
	@echo "  make test             运行所有测试"
	@echo "  make test-unit        运行单元测试"
	@echo "  make test-integration 运行集成测试"
	@echo "  make coverage         生成测试覆盖率报告"
	@echo ""
	@echo "代码质量:"
	@echo "  make lint             代码检查"
	@echo "  make format           代码格式化"
	@echo "  make security         安全扫描"
	@echo ""
	@echo "构建部署:"
	@echo "  make build            构建Docker镜像"
	@echo "  make deploy           部署到生产环境"
	@echo "  make backup           数据备份"
	@echo ""
	@echo "清理维护:"
	@echo "  make clean            清理构建文件"
	@echo "  make clean-data       清理数据文件"
	@echo "  make logs             查看日志"
```

## 构建与测试

### Python 后端

```bash
pip install -r requirements.txt
pytest tests/ -v
ruff check .
mypy app/
uvicorn app.main:app --reload
```

### 前端

```bash
npm install
npm test -- --watch
npm run build
npm start
```

### Docker

```bash
docker-compose build
docker-compose up -d
docker-compose run backend pytest
curl http://localhost:8000/health
```

### Makefile快捷操作

项目提供Makefile简化开发运维操作：

```makefile
# 开发环境
make dev              # 启动开发环境
make stop             # 停止所有服务
make restart          # 重启服务

# 测试相关
make test             # 运行所有测试
make test-unit        # 运行单元测试
make test-integration # 运行集成测试
make coverage         # 生成测试覆盖率报告

# 代码质量
make lint             # 代码检查
make format           # 代码格式化
make security         # 安全扫描

# 构建部署
make build            # 构建Docker镜像
make deploy          # 部署到生产环境
make backup          # 数据备份

# 清理维护
make clean           # 清理构建文件
make clean-data      # 清理数据文件
make logs            # 查看日志
```

### CLI工具

```
# 向量化处理
jira-vector vectorize --input cleaned.json --model bge-m3

# 聚类分析
jira-vector cluster --algorithm hdbscan --min-size 10

# 结果导出
jira-vector export --format csv --output results.csv

# 系统状态检查
jira-vector health

# 系统自动增量识别（无需特殊命令）
# 使用常规预处理命令，系统自动识别增量更新
jira-vector preprocess --input new_data.csv --output processed_data.json

```

### 对话式Agent Skill

**新增功能**：支持通过自然语言对话方式完成数据导入与分析操作

#### 🚀 快速开始

```bash
# 启动对话式Agent
jira-vector chat

# 或者使用详细模式
jira-vector chat --verbose --log-file agent.log
```


#### 💬 示例对话场景

**场景1：数据分析全流程**
```
User: "请帮我分析这个月的Jira任务数据"
Agent: "好的，请上传您的Jira导出文件，我将自动进行数据处理和分析。"

User: "这是我的数据文件：jira_tasks.csv"
Agent: "正在处理您的数据文件...
       ✓ 数据预处理完成 (156条记录)
       ✓ 向量嵌入生成完成 
       ✓ 聚类分析完成
       分析已完成，共识别出5个主要任务类别。"

User: "我想查看聚类结果"
Agent: "正在为您生成聚类分析报告，请稍等...
       📊 分析结果：
       - 数据库运维: 32条 (20.5%)
       - 应用部署: 28条 (17.9%)
       - 故障处理: 25条 (16.0%)
       - 安全管理: 22条 (14.1%)
       - 性能优化: 19条 (12.2%)
       其他类别: 30条 (19.3%)"

User: "能否导出分析结果？"
Agent: "已为您导出CSV格式的分析结果到results_20260301.csv文件。
       同时生成了详细的PDF报告：analysis_report_20260301.pdf"
```

**场景2：参数调整交互**
```
User: "我觉得聚类结果不够准确，能调整一下参数吗？"
Agent: "当然可以！我可以帮您调整聚类参数：
       1. 最小聚类大小 (当前: 10) - 控制每个簇的最小任务数
       2. 聚类紧密度 (当前: 0.5) - 控制簇的紧密程度
       3. 算法选择 (当前: hdbscan) - 可切换到kmeans
       
       您希望调整哪个参数？"

User: "把最小聚类大小调整为5"
Agent: "正在重新进行聚类分析...
       ✓ 参数已更新: min_cluster_size = 5
       ✓ 重新聚类完成
       新的分析结果显示7个任务类别，细粒度更高。"
```

#### 🎯 核心能力详解

**1. 自然语言理解与处理**
- 支持中文自然语言指令解析
- 智能识别用户意图和上下文
- 自动参数提取和验证
- 错误输入的友好提示

**2. 对话式数据导入指导**
- 智能文件格式识别
- 数据质量自动检查
- 缺失字段智能补全
- 导入进度实时反馈

**3. 智能分析结果解释**
- 统计数据可视化描述
- 业务含义智能解读
- 异常模式自动识别
- 趋势分析和建议

**4. 交互式参数调整**
- 实时参数修改反馈
- 效果预览和对比
- 最佳实践建议
- 参数影响说明

**5. 多轮对话状态保持**
- 会话上下文记忆
- 操作历史追踪
- 中断恢复支持
- 个性化偏好学习

## 开发最佳实践与质量要求

### 后端开发最佳实践

**代码质量要求：**
- **类型安全**: 所有公共函数/类必须包含完整类型提示，使用mypy进行静态类型检查
- **错误处理**: 统一异常处理机制，自定义异常继承自基类，使用结构化日志记录
- **性能优化**: 数据库查询使用连接池，API调用实现缓存机制，避免N+1查询问题
- **安全性**: 输入验证使用Pydantic模型，SQL注入防护，JWT令牌刷新机制
- **可维护性**: 函数单一职责原则，模块解耦，清晰的接口设计

**架构约束：**
- **分层架构**: 严格遵循Controller-Service-Repository三层架构
- **依赖注入**: 使用FastAPI Depends实现依赖注入
- **配置管理**: 环境变量集中管理，敏感信息加密存储
- **API设计**: RESTful风格，版本控制，统一响应格式

**性能基准：**
- API响应时间 < 200ms (95th percentile)
- 数据库查询平均耗时 < 50ms
- 内存使用率峰值 < 75%
- 系统可用性 ≥ 99.9%

### 前端开发最佳实践

**代码质量要求：**
- **类型安全**: 启用TypeScript严格模式，禁用隐式any类型，完整类型定义
- **组件设计**: 遵循单一职责原则，优先使用函数组件和Hooks
- **状态管理**: Redux Toolkit统一状态管理，合理划分state slices
- **性能优化**: 虚拟滚动处理大数据列表，React.memo优化重渲染，代码分割按路由懒加载
- **用户体验**: 加载状态提示，错误边界处理，键盘导航支持

**架构约束：**
- **组件层级**: 页面组件 → 容器组件 → 展示组件三级结构
- **数据流**: 单向数据流，避免props drilling
- **样式管理**: CSS-in-JS或CSS Modules，避免全局样式污染
- **国际化**: 使用i18next实现多语言支持

**性能基准：**
- 首屏加载时间 < 2s
- Bundle大小 < 2MB (gzip压缩后)
- Lighthouse性能评分 ≥ 90
- React组件重渲染次数控制在合理范围内

## 企业级代码质量标准

### 高质量代码要求

**可读性标准：**
- 函数长度 ≤ 50行
- 参数数量 ≤ 3个
- 变量命名具有描述性，避免缩写
- 复杂逻辑必须添加注释说明
- 统一的代码风格和格式化

**高性能要求：**
- 算法时间复杂度控制在O(n log n)以内
- 数据库索引合理使用
- 内存泄漏预防和检测
- 异步操作避免阻塞主线程

**安全性规范：**
- 输入验证和净化
- 输出编码防止XSS攻击
- CORS策略配置
- 安全头设置(Helmet)
- 敏感信息加密传输

### 代码审查标准

**必须检查项：**
- 安全漏洞扫描
- 性能瓶颈识别
- 代码覆盖率验证
- 兼容性测试
- 异常处理完整性

## API设计规范

- 基础URL: `/api/v1`
- 认证方式: Bearer JWT令牌（24小时有效期）
- 速率限制: 100请求/分钟，最大50MB请求
- 错误格式:
```
{"error":{"code":"ERROR_CODE","message":"人类可读的错误信息","details":"..."}}
```

### 数据处理规范

#### 输入格式

- **CSV**: `issue_id,summary,description,created_at,updated_at,cluster_label`
- **JSON**: 包含 `issue_id`, `summary`, `description`, `created_at`, `updated_at`, `cluster_label` 的数组

#### 增量更新机制

**核心能力**：
- ✅ **智能自动识别**：系统在常规数据导入过程中自动判断新增/更新数据
- ✅ 支持多批次CSV文件导入
- ✅ 基于issue_id智能识别已存在任务单
- ✅ 对已存在任务进行数据更新
- ✅ 对新任务单进行新增处理
- ✅ 保持历史聚类结果的连续性

**处理逻辑**：
1. 系统维护任务单唯一标识(issue_id)映射表
2. **自动识别机制**：任何数据导入操作都会自动触发增量识别
3. 已存在任务：更新字段内容，保留历史分类信息
4. 新任务：执行完整的数据处理流程
5. 增量数据可选择重新参与聚类分析

**用户无需额外操作**：
- 不需要使用特殊的增量导入命令
- 常规的`jira-vector preprocess`命令即可完成智能增量处理
- 系统后台自动完成数据比对和分类处理

#### 清洗流程

1. 移除代码块(\`\`\`)、日志行(ERROR/WARN/DEBUG)、堆栈跟踪信息
2. 清除HTML标签
3. 清洗后将描述限制在500字符以内
4. UTF-8编码
5. 合并 `summary` + 清洗后的 `description` 前缀

## 项目结构

```
.
├── app/                  # 后端代码
│   ├── api/             # API端点
│   ├── core/            # 核心逻辑 (preprocessing.py, vector_store.py, clustering.py)
│   ├── models/          # Pydantic模型
│   ├── exceptions.py    # 自定义异常
│   └── main.py          # FastAPI入口
├── frontend/            # React应用
│   └── src/
├── tests/               # 测试文件
├── docs/                # 文档
├── scripts/             # CLI工具
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

## 测试要求

- 单元测试: 覆盖率80%以上，模拟外部依赖
- 集成测试: API端点、CLI工具、Docker环境
- 性能测试: 5000条任务<10分钟，内存<2GB，API<200ms

## 开发工作流

1. 从 `main` 分支创建新分支
2. 推送前运行 `pytest`
3. 修复所有 `ruff` 错误
4. 提交格式: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`
5. 向 `main` 发起PR并附带测试结果

## 代码检查与格式化

- Python: Ruff
- TypeScript: ESLint + Prettier
- 使用pre-commit钩子进行代码检查

## CI/CD流程

- PR到 `main` 时触发构建
- 使用pytest + 覆盖率进行测试
- 安全扫描
- Docker镜像构建和推送

## 部署配置

### 环境变量

- `LANCEDB_URI`, `EMBEDDING_MODEL`, `JWT_SECRET`
- 环境: `development`, `staging`, `production`

### 监控指标

- 健康检查: `/api/v1/health`
- 性能指标: 内存<80%，CPU<70%
- 日志格式: Apache CloudTrail格式

## 常见开发任务

### 新增API端点

1. 在 `app/api/routes/` 中添加路由
2. 在 `app/models/` 中创建Pydantic模型
3. 在 `tests/integration/` 中添加集成测试
4. 更新 `docs/api.md` 文档

### 修改聚类算法

1. 更新 `app/core/clustering.py`
2. 添加测试用例
3. 更新 `scripts/cluster.py` 中的CLI参数
4. 更新相关文档

## AI代码审查代理 (Code Review Agent)

### 代理职责

**核心功能：**
- 自动化代码质量检查和评审
- 安全漏洞识别和风险评估
- 性能优化建议和瓶颈分析
- 代码规范符合性验证
- 技术债务识别和改进建议

**审查范围：**
- 静态代码分析
- 安全扫描(SAST)
- 性能基准测试
- 代码复杂度评估
- 最佳实践合规性检查

### 审查流程

**自动化审查：**
1. 代码提交触发审查流水线
2. 执行静态分析工具(Ruff, ESLint, SonarQube)
3. 运行安全扫描和漏洞检测
4. 性能基准测试和回归分析
5. 生成详细的审查报告

**人工复核：**
- 复杂业务逻辑的深度审查
- 架构设计合理性评估
- 技术选型适当性检查
- 团队编码规范一致性验证

### 审查标准

**质量门禁：**
- 代码覆盖率 ≥ 85%
- 安全扫描无高危漏洞
- 性能测试通过基准要求
- 代码复杂度指数 < 10
- 重复代码率 < 5%

**审查维度：**
- **功能性**: 代码正确性和完整性
- **可维护性**: 代码结构和可读性
- **安全性**: 漏洞防护和数据保护
- **性能**: 执行效率和资源使用
- **可扩展性**: 架构弹性和适应性

### 集成配置

```
# .code-review-config.yaml
enabled: true
quality_gate:
  coverage: 85
  security_level: HIGH
  performance_threshold: 200ms
review_rules:
  - name: "security_scan"
    tool: "bandit,safety"
    severity: "HIGH,CRITICAL"
  - name: "performance_test"
    tool: "pytest-benchmark"
    threshold: "95th_percentile"
```



## 智能代理开发指南

### 代码生成规范

- 严格遵循现有代码模式
- FastAPI模型使用Pydantic
- 所有函数必须包含类型提示
- 包含错误处理和日志记录
- I/O操作使用异步编程

### 调试指南

- 检查 `docker-compose logs`
- 验证LanceDB存储: `ls -la data/`
- 隔离测试各个组件

### 文档维护

- 以PRD.md作为权威来源
- API端点变更时及时更新API文档
- 为CLI命令提供详细示例文档
- **用户手册同步**: 每次功能或代码变更后必须检查并更新用户手册

### 用户手册同步机制

**同步检查清单：**
- [ ] 新增功能是否在用户手册中有对应说明
- [ ] 接口变更是否更新了使用示例
- [ ] 配置参数调整是否同步到安装指南
- [ ] CLI命令变更是否更新命令列表
- [ ] 界面改动是否更新操作指南

**同步流程：**
1. 代码变更完成后，执行 `make docs-check`
2. 对比变更内容与用户手册覆盖度
3. 更新相关文档章节
4. 验证文档准确性
5. 提交文档更新与代码变更

**自动化检查脚本：**
```
#!/bin/bash
# scripts/check-docs-sync.sh

echo "🔍 检查用户手册同步状态..."

# 检查最近一次提交的变更
CHANGED_FILES=$(git diff --name-only HEAD~1 HEAD)

# 检查是否涉及需要同步的变更
for file in $CHANGED_FILES; do
  case $file in
    app/api/*)
      echo "⚠️  API接口变更，需要更新用户手册中的接口说明"
      ;;
    cli/*)
      echo "⚠️  CLI工具变更，需要更新用户手册中的命令说明"
      ;;
    frontend/src/*)
      echo "⚠️  前端界面变更，需要更新用户手册中的操作指南"
      ;;
  esac
done

echo "✅ 文档同步检查完成"
```

## 文档版本管理

### 版本同步策略

**文档与代码版本对应关系：**
- 主版本号：重大功能变更时同步更新
- 次版本号：新增功能时同步更新
- 修订号：bug修复和小幅改进时同步更新

**版本标记规范：**
```
文档版本: v1.2.3 (对应代码版本 v1.2.3)
更新时间: 2024-01-15
更新内容: 新增批量导入功能说明
```

### 快速入门指南

#### 系统安装

**前提条件：**
- Python 3.10+ 环境
- Node.js 16+ 环境
- Docker (推荐)
- 4GB以上内存

**安装步骤：**
```bash
# 1. 克隆项目
git clone https://github.com/your-org/JiraVector-Analytics.git
cd JiraVector-Analytics

# 2. 后端环境安装
pip install -r requirements.txt

# 3. 前端环境安装
cd frontend
npm install
cd ..

# 4. 启动服务
# 方式一：Docker启动（推荐）
docker-compose up -d

# 方式二：本地启动
# 后端：uvicorn app.main:app --reload
# 前端：cd frontend && npm start
```

#### 数据准备

**Jira数据导出：**
1. 登录Jira系统
2. 进入项目页面
3. 选择"导出"功能
4. 选择CSV或JSON格式
5. 确保包含以下字段：
   - issue_id (任务ID)
   - summary (任务标题)
   - description (任务描述)
   - created_at (创建时间)
   - status (状态)
   - priority (优先级)

#### 基本操作流程

**1. 首次数据导入**
```bash
# 使用CLI工具上传初始数据
jira-vector preprocess --input initial_data.csv --output processed_data.json
```

**2. 增量数据处理**
```bash
# 系统自动识别增量更新（无需特殊命令）
jira-vector preprocess --input batch_002.csv --output processed_data.json

# 或者使用Web界面上传文件
# 系统后台自动识别新增/更新任务
```

**3. 执行聚类分析**
```bash
# 向量化处理
jira-vector vectorize --input processed_data.json --model bge-m3

# 聚类分析（可选择是否包含增量数据）
jira-vector cluster --algorithm hdbscan --min-size 10 --include-incremental
```

**4. 查看结果**
```bash
# 导出分析结果
jira-vector export --format csv --output cluster_results.csv

# 查看系统状态
jira-vector health
```

### Web界面操作指南

#### 登录系统
1. 打开浏览器访问 `http://localhost:3000`
2. 使用默认账号登录：
   - 用户名：admin
   - 密码：password123

#### 主要功能模块

**数据管理页面：**
- 上传Jira导出文件
- 系统自动识别增量更新
- 查看数据处理进度
- 管理历史数据集
- 查看数据更新日志

**分析结果页面：**
- 查看聚类分析结果
- 浏览各类别任务分布
- 下载分析报告

**人工复核页面：**
- 对比AI分类与原分类
- 批量确认分类结果
- 导入复核决策文件

### 常见问题解答

**Q: 系统支持哪些数据格式？**
A: 支持CSV和JSON两种格式的Jira导出数据。

**Q: 处理大量数据需要多长时间？**
A: 5000条数据约需8-10分钟，具体取决于硬件配置。

**Q: 如何提高分类准确率？**
A: 确保数据质量，提供完整的任务描述，适当调整聚类参数。

**Q: 支持中文任务单分析吗？**
A: 完全支持，系统使用中文优化的BGE-M3嵌入模型。

**Q: 如何批量处理多个项目的数据？**
A: 可以将多个项目的导出文件合并后一次性处理。

### 系统维护

**日常检查：**
```bash
# 检查系统健康状态
jira-vector health

# 查看系统日志
docker-compose logs -f

# 监控资源使用
docker stats
```

**数据备份：**
```bash
# 备份向量数据库
mkdir -p backups
cp -r data/lancedb backups/lancedb_$(date +%Y%m%d)

# 备份配置文件
cp .env backups/env_$(date +%Y%m%d)
```

**系统升级：**
```bash
# 拉取最新代码
git pull origin main

# 重建Docker镜像
docker-compose build

# 重启服务
docker-compose down
docker-compose up -d
```

最新的开发进度同步更新TODO.md文件，要求清晰简洁。

## 用户使用手册

> 📝 **版本信息**: v1.0.0 | 更新时间: 2024-01-15 | 对应代码版本: v1.0.0
>
> ⚠️ **注意**: 本文档与代码保持同步更新，如有差异请以最新版本为准。
>
> 🔄 **同步状态**: ✅ 最新版本已同步所有功能变更

### 🚀 快速开始

请参考上方的[快速入门指南](#快速入门指南)获取详细的安装和使用说明。

### 📖 详细操作指南

#### Web界面操作

**登录系统**
1. 打开浏览器访问 `http://localhost:3000`
2. 使用默认账号登录：
   - 用户名：admin
   - 密码：password123

**主要功能模块**

**数据管理页面：**
- 上传Jira导出文件
- 系统自动识别增量更新
- 查看数据处理进度
- 管理历史数据集
- 查看数据更新日志

**分析结果页面：**
- 查看聚类分析结果
- 浏览各类别任务分布
- 下载分析报告

**人工复核页面：**
- 对比AI分类与原分类
- 批量确认分类结果
- 导入复核决策文件

### ❓ 常见问题解答

**Q: 系统支持哪些数据格式？**
A: 支持CSV和JSON两种格式的Jira导出数据。

**Q: 处理大量数据需要多长时间？**
A: 5000条数据约需8-10分钟，具体取决于硬件配置。

**Q: 如何提高分类准确率？**
A: 确保数据质量，提供完整的任务描述，适当调整聚类参数。

**Q: 支持中文任务单分析吗？**
A: 完全支持，系统使用中文优化的BGE-M3嵌入模型。

**Q: 如何批量处理多个项目的数据？**
A: 可以将多个项目的导出文件合并后一次性处理。

### 🔧 系统维护

**日常检查：**
```bash
# 检查系统健康状态
jira-vector health

# 查看系统日志
docker-compose logs -f

# 监控资源使用
docker stats
```

**数据备份：**
```bash
# 备份向量数据库
mkdir -p backups
cp -r data/lancedb backups/lancedb_$(date +%Y%m%d)

# 备份配置文件
cp .env backups/env_$(date +%Y%m%d)
```

**系统升级：**
```bash
# 拉取最新代码
git pull origin main

# 重建Docker镜像
docker-compose build

# 重启服务
docker-compose down
docker-compose up -d
```

## 参考资源

### 核心文档
- [PRD.md](./PRD.md) - 完整需求文档
- [README.md](./README.md) - 项目概述
- [API Documentation](./docs/api.md) - API参考文档（待完善）
- [Deployment Guide](./docs/deployment.md) - 部署指南（待完善）

### 代码质量工具
- [Ruff](https://beta.ruff.rs/docs/) - Python代码检查工具
- [ESLint](https://eslint.org/) - JavaScript/TypeScript代码检查
- [SonarQube](https://www.sonarqube.org/) - 代码质量管理平台
- [Bandit](https://bandit.readthedocs.io/) - Python安全扫描工具

### 性能监控
- [Pytest-Benchmark](https://pytest-benchmark.readthedocs.io/) - Python性能测试
- [Lighthouse](https://developer.chrome.com/docs/lighthouse/) - Web性能审计
- [Prometheus](https://prometheus.io/) - 系统监控和告警

### 安全工具
- [Safety](https://pyup.io/safety/) - Python依赖安全检查
- [OWASP ZAP](https://www.zaproxy.org/) - Web应用安全测试
- [Snyk](https://snyk.io/) - 开源组件安全监控
