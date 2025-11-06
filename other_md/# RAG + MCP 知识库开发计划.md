# RAG + MCP 知识库开发计划

## 项目目标
将现有RAG模块整合成完整流程，并封装为MCP服务，作为Agent项目的外挂知识库。

## 技术栈确认
- **文档处理**: PDF, Excel
- **向量模型**: BAAI/bge-large-zh-v1.5
- **Rerank模型**: BAAI/bge-reranker-large
- **向量数据库**: Chroma (推荐) / Faiss
- **LLM**: DeepSeek / Qwen
- **框架**: LangChain
- **接口协议**: MCP (Model Context Protocol)

---

## 第一阶段：最小可行流程 (2-3天)

### 目标
跑通单文档的完整问答流程，验证核心功能。

### 任务清单
- [ ] **环境准备**
  - 整理现有代码，确认依赖包版本
  - 准备测试文档（1个PDF）
  - 创建项目目录结构

- [ ] **文档处理模块**
  - 集成现有PDF分块代码
  - 固定参数：chunk_size=500, overlap=50
  - 输出：文本块列表 + 元数据

- [ ] **向量化与存储**
  - 加载 bge-large-zh-v1.5 模型
  - 批量向量化文本块
  - 初始化 Chroma 数据库
  - 存储向量 + 元数据

- [ ] **检索模块**
  - 实现简单的向量检索
  - 固定 top_k=5
  - 返回相关文本块

- [ ] **问答生成**
  - 设计基础 Prompt 模板
  - 集成 DeepSeek API
  - 输入：问题 + 检索到的上下文
  - 输出：答案

- [ ] **端到端测试**
  - 准备 5-10 个测试问题
  - 验证答案准确性
  - 记录问题和改进点

### 关键参数（第一阶段固定值）
```python
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
MAX_CONTEXT_LENGTH = 2000  # tokens
```

### 交付物
- 可运行的 Python 脚本
- 测试结果文档（问题-答案对）

---

## 第二阶段：功能完善 (3-4天)

### 目标
增加多文档、对话历史、Rerank等核心功能。

### 任务清单
- [ ] **多文档支持**
  - 支持批量导入 PDF + Excel
  - 为每个文档添加来源标识
  - 实现文档管理接口（增删查）

- [ ] **Excel处理优化**
  - 集成现有 Excel 分块代码
  - 保留表格结构信息
  - 统一文本块格式

- [ ] **Rerank集成**
  - 加载 bge-reranker-large 模型
  - 两阶段检索：向量检索(top_k=20) → Rerank(top_n=5)
  - 对比开启/关闭 Rerank 的效果

- [ ] **对话历史管理**
  - 使用 LangChain 的 ConversationBufferMemory
  - 限制历史轮数（如最近3轮）
  - 实现历史清除功能

- [ ] **上下文优化**
  - 实现智能截断（按 token 限制）
  - 优先保留高分文本块
  - 添加来源引用信息

- [ ] **错误处理**
  - 文档解析失败捕获
  - LLM 超时重试机制
  - 空结果友好提示

- [ ] **配置文件**
  - 抽离所有超参数到 config.yaml
  - 支持环境变量覆盖

### 参数调优范围
```python
CHUNK_SIZE: 300-800 (需实验)
CHUNK_OVERLAP: 50-150
VECTOR_TOP_K: 10-30
RERANK_TOP_N: 3-8
MAX_CONTEXT_TOKENS: 1500-3000
```

### 交付物
- 完整的 RAG 系统代码
- 配置文件模板
- 性能对比报告（有无Rerank）

---

## 第三阶段：MCP服务封装 (2-3天)

### 目标
将RAG系统封装为标准MCP服务，对接Agent项目。

### 任务清单
- [ ] **MCP协议设计**
  - 定义接口规范（JSON Schema）
  - 请求格式：`{"query": "问题", "session_id": "会话ID", "top_k": 5}`
  - 响应格式：`{"answer": "答案", "sources": [...], "confidence": 0.8}`

- [ ] **服务端实现**
  - 使用 FastAPI / Flask 搭建 HTTP 服务
  - 实现接口：
    - `POST /query` - 问答
    - `POST /upload` - 上传文档
    - `GET /documents` - 文档列表
    - `DELETE /documents/{id}` - 删除文档
    - `POST /reset` - 清除会话历史

- [ ] **模型常驻优化**
  - 服务启动时加载所有模型
  - 实现模型单例模式
  - 减少冷启动时间

- [ ] **并发处理**
  - 实现请求队列
  - 限制并发数（如3-5）
  - 添加请求超时机制

- [ ] **日志与监控**
  - 记录所有请求/响应
  - 统计调用次数、耗时
  - 错误告警机制

- [ ] **Docker化**
  - 编写 Dockerfile
  - 配置 GPU 支持（如需要）
  - 提供 docker-compose.yml

- [ ] **文档与测试**
  - API 文档（Swagger）
  - 编写集成测试脚本
  - Agent 对接示例代码

### MCP接口示例
```json
// 请求
{
  "query": "公司去年的营收是多少？",
  "session_id": "user123-session456",
  "options": {
    "top_k": 5,
    "use_rerank": true
  }
}

// 响应
{
  "answer": "根据财报，公司去年营收为10亿元。",
  "sources": [
    {
      "document": "2023年报.pdf",
      "page": 5,
      "content": "...营收达到10亿元...",
      "score": 0.89
    }
  ],
  "confidence": 0.85,
  "elapsed_time": 1.2
}
```

### 交付物
- MCP 服务代码
- Docker 镜像
- API 文档
- 对接指南

---

## 第四阶段：优化与迭代 (持续)

### 优化方向
- [ ] **检索质量优化**
  - 实验不同分块策略
  - 调优 Rerank 阈值
  - A/B 测试不同参数组合

- [ ] **性能优化**
  - 向量检索加速（ANN索引）
  - 批量处理优化
  - 缓存热门查询

- [ ] **功能增强**
  - 支持更多文档格式（Word, TXT, Markdown）
  - 多模态支持（图片OCR）
  - 流式输出答案

- [ ] **用户体验**
  - 添加 Web UI（Streamlit / Gradio）
  - 实时反馈检索进度
  - 答案质量评分功能

---

## 技术风险与应对

### 风险1：内存占用过高
- **影响**: BGE模型(2-3GB) + Reranker(2GB) + Chroma索引
- **应对**: 
  - 使用量化模型（INT8）
  - 考虑模型分离部署
  - 限制并发数

### 风险2：检索效果不佳
- **影响**: 答案不准确，引用错误
- **应对**:
  - 准备评测数据集
  - 系统化调参实验
  - 考虑混合检索（BM25 + 向量）

### 风险3：LLM调用成本/延迟
- **影响**: 响应慢，费用高
- **应对**:
  - 优化上下文长度
  - 使用更小的模型（如qwen-turbo）
  - 本地部署方案（如Ollama）

### 风险4：MCP协议兼容性
- **影响**: Agent对接困难
- **应对**:
  - 预留协议版本字段
  - 提供降级方案（标准HTTP）
  - 充分的集成测试

---

## 时间线总览

| 阶段 | 工作日 | 关键里程碑 |
|------|--------|-----------|
| 阶段一 | 2-3天 | 单文档问答可用 |
| 阶段二 | 3-4天 | 完整RAG系统 |
| 阶段三 | 2-3天 | MCP服务上线 |
| 阶段四 | 持续 | 优化迭代 |
| **总计** | **7-10天** | **可用的MCP知识库** |

---

## 开发建议

### 优先级原则
1. **先通后优**: 先跑通流程，再优化性能
2. **小步快跑**: 每完成一个模块就测试
3. **记录问题**: 遇到的坑和解决方案都记录下来

### 代码管理
```
rag-mcp-project/
├── src/
│   ├── document_loader/    # 文档处理
│   ├── vectorstore/         # 向量存储
│   ├── retriever/           # 检索+Rerank
│   ├── llm/                 # LLM调用
│   └── mcp_server/          # MCP服务
├── config/
│   └── config.yaml          # 配置文件
├── tests/                   # 测试用例
├── docs/                    # 文档
└── docker/                  # Docker配置
```

### 测试策略
- 单元测试：每个模块独立测试
- 集成测试：端到端流程测试
- 压力测试：并发请求测试
- 准确性测试：准备标准问答对

---

## 后续规划

完成MCP服务后，可以考虑：
1. **知识库管理界面**: 文档上传、查看、删除
2. **多租户支持**: 不同用户独立知识库
3. **增量更新**: 新增文档无需重建索引
4. **分布式部署**: 向量库、LLM、服务分离
5. **开源发布**: 整理代码，发布到GitHub

---

## 参考资源

- LangChain文档: https://python.langchain.com/
- BGE模型: https://huggingface.co/BAAI
- Chroma文档: https://docs.trychroma.com/
- MCP协议: https://modelcontextprotocol.io/
- DeepSeek API: https://platform.deepseek.com/
- Qwen API: https://help.aliyun.com/zh/dashscope/

---

**最后提醒**: 
- 每个阶段结束后总结经验
- 遇到问题及时调整计划
- 保持代码简洁，避免过早优化
- 优先保证功能可用，再追求完美

祝开发顺利！🚀