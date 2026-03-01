---
name: "jira-cli-agent"
description: "Natural language interface for Jira data analysis using CLI commands. Invoke when user wants to analyze Jira tasks, perform clustering, or needs conversational guidance for data operations."
---

# Jira CLI Agent Skill

This skill provides a natural language conversational interface for Jira Vector Analytics CLI operations, enabling users to interact with the system through intuitive dialogue instead of memorizing complex command-line syntax.

## Core Capabilities

### 1. Data Import & Preprocessing
- Guide users through Jira data file upload (CSV/JSON format)
- **File format validation**: Strict validation against demo_sample.csv format
  - Required CSV fields: `issue_id,summary,description,created_at,updated_at,cluster_label`
  - Field order must match exactly
  - UTF-8 encoding required
- Automate data preprocessing pipeline
- Support incremental data updates with automatic detection
- Handle batch processing of multiple files

### 2. Vector Embedding Generation
- Generate semantic embeddings using BGE-M3 or text-embedding-3-small models
- Process large datasets efficiently
- Store vectors in LanceDB for fast retrieval

### 3. Clustering Analysis
- Execute HDBSCAN or K-Means clustering algorithms
- Automatically identify task categories
- Generate cluster labels with business context
- Provide confidence scores for each classification

### 4. Results Export & Visualization
- Export analysis results in CSV/JSON formats
- Generate comprehensive analysis reports
- Display clustering statistics and insights
- Support batch review and confirmation workflows

### 5. System Health Monitoring
- Check system status and dependencies
- Validate data directory accessibility
- Monitor resource usage and performance metrics

## Usage Scenarios

### Scenario 1: First-time Data Analysis
**User**: "请帮我分析这个月的Jira任务数据"
**Agent**: 
1. Request file upload
2. Guide through preprocessing
3. Execute vectorization
4. Perform clustering analysis
5. Present results summary

### Scenario 2: Incremental Updates
**User**: "我有新的任务数据需要更新"
**Agent**:
1. Accept new data file
2. Automatically detect incremental changes
3. Update existing vectors
4. Re-run clustering if requested
5. Compare with previous results

### Scenario 3: Parameter Tuning
**User**: "调整聚类参数，最小簇大小设为15"
**Agent**:
1. Update clustering configuration
2. Re-execute analysis with new parameters
3. Compare results with previous run
4. Provide recommendations

### Scenario 4: Results Review
**User**: "查看聚类分析结果并导出"
**Agent**:
1. Display clustering summary
2. Show cluster details and sample tasks
3. Offer export options (CSV/JSON)
4. Generate comprehensive report

## CLI Command Mapping

The agent translates natural language requests to CLI commands:

| User Intent | CLI Command |
|------------|-------------|
| Import data | `jira-vector preprocess --input <file> --output <output>` |
| Vectorize | `jira-vector vectorize --input <file> --model <model>` |
| Cluster | `jira-vector cluster --algorithm <algo> --min-size <size>` |
| Export | `jira-vector export --format <format> --output <file>` |
| Health check | `jira-vector health` |
| Chat mode | `jira-vector chat` |

## Intent Recognition

The agent recognizes the following intent patterns:

- **Data Import**: "导入数据", "上传文件", "分析任务", "load data"
- **Clustering**: "聚类分析", "分类任务", "cluster analysis", "分组任务"
- **Results**: "查看结果", "显示分析", "show results", "导出结果"
- **Parameters**: "调整参数", "修改设置", "change parameters"
- **Help**: "帮助", "怎么用", "help", "如何使用"

## Session Management

Each conversation session maintains:
- User message history
- Processing context (uploaded files, processed data)
- Analysis results
- Parameter preferences
- Session state (data_processed, analysis_completed)

## Error Handling

The agent provides graceful error handling for:
- Invalid file formats
- Missing dependencies
- Processing failures
- Network issues
- Insufficient resources

## Best Practices

1. **Always validate file paths** before processing
2. **Provide clear progress feedback** during long operations
3. **Offer alternative solutions** when errors occur
4. **Maintain conversation context** across multiple interactions
5. **Generate human-readable summaries** for technical results

## Integration Points

The agent integrates with:
- `app.core.preprocessing.DataPreprocessor` - Data cleaning and preparation
- `app.core.embedding.VectorEmbedder` - Vector generation
- `app.core.clustering.TaskClusterer` - Clustering algorithms
- `app.models.ChatSession` - Session management
- `app.models.ChatMessage` - Message handling

## Example Conversations

### Example 1: Complete Analysis Workflow
```
User: "我想分析这个季度的Jira任务"
Agent: "好的，请上传您的Jira导出文件（CSV格式）"
User: [uploads file]
Agent: "✅ 已成功加载并预处理 1250 个任务。现在可以进行聚类分析了！"
User: "开始聚类分析"
Agent: "📊 聚类分析完成！识别出 8 个主要任务类别，处理耗时 45.3 秒"
User: "查看详细结果"
Agent: [displays detailed clustering report with sample tasks]
User: "导出为CSV格式"
Agent: "✅ 结果已导出到 exported_results.csv"
```

### Example 2: Parameter Optimization
```
User: "聚类结果太多了，我想合并一些小类别"
Agent: "建议将最小聚类大小从 10 调整到 15。要重新分析吗？"
User: "好的，调整参数并重新分析"
Agent: "✅ 已调整参数并重新聚类。现在识别出 6 个主要类别"
```

## Technical Notes

- **Concurrency**: Uses asyncio for non-blocking operations
- **Resource Management**: Proper cleanup of embedder and clusterer instances
- **Logging**: Comprehensive logging for debugging and monitoring
- **Type Safety**: Full type hints for all public methods
- **Error Recovery**: Graceful degradation on failures

## Performance Considerations

- **Vector Generation**: ~100-200 tasks/second depending on model
- **Clustering**: HDBSCAN scales well to 10,000+ tasks
- **Memory Usage**: Typical < 2GB for 5,000 tasks
- **Session Storage**: In-memory with configurable persistence

## Extension Points

The agent can be extended to support:
- Custom clustering algorithms
- Additional embedding models
- Real-time webhook integration
- Multi-language support
- Advanced visualization options
- Automated report scheduling

## Microservice Architecture Guidelines

### Service Decomposition

The system is designed with microservice architecture principles:

1. **Data Preprocessing Service**
   - Service ID: `jira-vector-preprocessor`
   - Responsibility: Data cleaning, validation, and feature extraction
   - API: RESTful API at `/api/v1/preprocessor`
   - Queue: RabbitMQ queue for async jobs
   - Health check: `/health` endpoint
   - Metrics: Prometheus metrics at `/metrics`

2. **Vector Embedding Service**
   - Service ID: `jira-vector-embedder`
   - Responsibility: Generate semantic embeddings using BGE-M3
   - API: RESTful API at `/api/v1/embedder`
   - Queue: RabbitMQ queue for embedding jobs
   - GPU acceleration: Optional CUDA support
   - Model caching: In-memory model cache

3. **Clustering Analysis Service**
   - Service ID: `jira-vector-clusterer`
   - Responsibility: HDBSCAN/K-Means clustering algorithms
   - API: RESTful API at `/api/v1/clusterer`
   - Queue: RabbitMQ queue for clustering jobs
   - Algorithm selection: Configurable via environment variables

4. **Vector Storage Service**
   - Service ID: `jira-vector-storage`
   - Responsibility: LanceDB vector database operations
   - API: RESTful API at `/api/v1/storage`
   - Replication: Multi-node LanceDB cluster
   - Backup: Automated hourly backups

5. **Conversation Agent Service**
   - Service ID: `jira-vector-chatbot`
   - Responsibility: Natural language conversation handling
   - API: RESTful API at `/api/v1/chatbot`
   - WebSocket: Real-time chat interface
   - Session management: Redis for session storage

### Inter-Service Communication

- **Synchronous**: REST/HTTP with retries and circuit breakers
- **Asynchronous**: RabbitMQ for background tasks
- **Service Discovery**: Consul or Kubernetes DNS
- **API Gateway**: Traefik or Kong for routing and rate limiting

### Configuration Management

- **Environment Variables**:
  - `SERVICE_NAME`: Unique service identifier
  - `LOG_LEVEL`: Logging verbosity (DEBUG/INFO/WARN/ERROR)
  - `RABBITMQ_URL`: Message broker connection string
  - `REDIS_URL`: Session storage connection string
  - `LANCEDB_URI`: Vector database connection
  - `EMBEDDING_MODEL`: Model selection (BGE-M3/text-embedding-3-small)

### Deployment Specification

```yaml
# Kubernetes deployment example
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jira-vector-preprocessor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: jira-vector-preprocessor
  template:
    metadata:
      labels:
        app: jira-vector-preprocessor
    spec:
      containers:
      - name: preprocessor
        image: jira-vector/preprocessor:latest
        ports:
        - containerPort: 8000
        env:
        - name: SERVICE_NAME
          value: "jira-vector-preprocessor"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Monitoring & Observability

- **Logging**: Structured JSON logs to ELK stack
- **Metrics**: Prometheus + Grafana dashboards
- **Tracing**: OpenTelemetry for distributed tracing
- **Alerting**: Prometheus Alertmanager for critical issues

### Security Requirements

- **Authentication**: JWT tokens for API access
- **Authorization**: RBAC for service-to-service communication
- **Encryption**: TLS 1.3 for all network traffic
- **Secrets**: HashiCorp Vault or Kubernetes Secrets
