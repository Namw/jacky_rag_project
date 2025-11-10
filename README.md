# RAG Project API

一个基于 FastAPI 的**检索增强生成（RAG）系统**，支持多个大语言模型提供商,支持离线运行本地 Embedding,具备语义缓存、使用限额管理和定时任务调度等高级功能。

## 主要特性

### 核心 RAG 能力
- 🤖 **多模型支持** - 支持 4 大 LLM 提供商（DeepSeek、OpenAI、阿里巴巴通义千问）及本地模型动态切换
- 📚 **文档管理** - 支持 PDF、Excel(实现中...) 等多种文档格式的上传、解析和管理
- 🔍 **向量检索** - 基于 ChromaDB 的向量数据库，支持语义相似度搜索和元数据过滤
- 🧠 **智能分块** - 基于 Embedding 的语义分块算法，自动合并相似内容段落

### 缓存与性能优化
- ⚡ **多层缓存系统**
  - 语义缓存：基于 Redis 的智能查询缓存，相似度阈值达 0.8（可配置）
  - 向量存储缓存：自动缓存 ChromaDB 实例，支持 LRU 淘汰和 TTL
  - 支持 Rerank 兼容性识别，提升缓存命中率
- 🎯 **精排机制** - 支持本地 CrossEncoder 和阿里云 TextReRank 两种精排实现

### 本地化支持
- 🖥️ **本地模型调用** - 支持离线运行本地 Embedding（BAAI/bge-large-zh-v1.5）和 Reranker（BAAI/bge-reranker-large）模型
- 🌍 **多语言优化** - 特别针对中文文本优化的 Embedding 和分块算法
- 💾 **离线运行** - 可完全离线部署，支持不依赖外部 API 的本地Embedding数据处理

### 管理与安全
- 📊 **使用限额管理** - 每日上传和问答次数限制，可自定义配置，自动每日重置
- 🔐 **身份认证与授权** - 基于 JWT 的用户认证，支持基于角色的访问控制（RBAC）
- 🔄 **定时任务调度** - APScheduler 支持的自动化任务（限额重置、缓存清理、文件清理等）
- 📖 **自动化文档** - FastAPI 内置的 Swagger/OpenAPI 自动文档和可视化界面

### 架构与设计
- 🏗️ **设计模式** - 采用单例、工厂、策略等设计模式，确保模型管理的高效性和可维护性
- 📦 **模块化设计** - 清晰的分层架构（API 层、服务层、数据层），易于扩展和维护
- 🔌 **可扩展架构** - 灵活的加载器、处理器和提供商接口，支持快速集成新模型和功能

## 快速开始

### 前置要求

- Python >= 3.13
- Redis（用于语义缓存和使用限额管理）
- 各大模型提供商的 API Key

### 安装

1. **克隆项目**
```bash
git clone <repository-url>
cd jakcy_rag_mcp
```

2. **安装依赖**
```bash
uv sync
```

3. **配置环境变量**

复制 `.env` 模板文件：
```bash
cp .env_copy .env
```

编辑 `.env` 文件，添加必要的配置。**两种工作模式选择**：

**模式 1：使用云端 API（推荐）**
```env
# LLM API Keys
DEEPSEEK_API_KEY=your_deepseek_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
DASHSCOPE_API_KEY=your_dashscope_api_key

# Embedding & Reranker 提供商选择（使用阿里云）
MODEL_PROVIDER=aliyun

# JWT 密钥
JWT_SECRET_KEY=your_secret_key

# Redis 配置
REDIS_HOST=localhost
REDIS_PORT=6379
```

**模式 2：使用本地模型（离线，需要更多磁盘空间）**
```env
# LLM API Keys - 仍需至少一个 LLM API Key（用于聊天模型）
DEEPSEEK_API_KEY=your_deepseek_api_key

# Embedding & Reranker 提供商选择（使用本地模型）
MODEL_PROVIDER=local

# JWT 密钥
JWT_SECRET_KEY=your_secret_key

# Redis 配置
REDIS_HOST=localhost
REDIS_PORT=6379

# 本地模型缓存目录（可选，默认使用 HuggingFace 官方缓存）
HF_HOME=/path/to/your/huggingface/cache
```

**重要说明**：
- 本地模型模式会在首次启动时自动下载 BAAI Embedding 和 Reranker 模型（总大小约 2-3GB）
- 下载的模型会缓存在本地，后续启动时直接使用缓存，无需重新下载
- 可完全离线运行本地模型，但仍需 LLM API Key 用于聊天回复生成
- 关于本地模型，如果无法访问 huggingface，可以用我百度网盘：https://pan.baidu.com/s/1mWX7BYTdX8pLfPHJTIYDnA 提取码: 3enr

4. **启动应用**

**开发环境（带自动重载）**
```bash
uv run uvicorn main:app --reload --port 8000
```

**生产环境**
```bash
uv run uvicorn main:app --port 8000
```

5. **访问 API 文档**

打开浏览器访问：`http://localhost:8000/docs`

## 项目结构

```
jakcy_rag_mcp/
├── main.py                          # 应用入口
├── pyproject.toml                   # 项目配置和依赖声明
├── uv.lock                          # 锁定的依赖版本
├── .env_copy                        # 环境变量模板
│
├── config/
│   └── model_config.py              # 模型配置（支持的 LLM 提供商）
│
├── models/
│   └── model_factory.py             # 模型工厂类
│
├── src/
│   ├── api/                         # FastAPI 路由
│   │   ├── auth.py                  # 用户认证和授权
│   │   ├── chat.py                  # 聊天接口
│   │   ├── documents.py             # 文档管理接口
│   │   ├── chromadb_lib.py          # 向量库管理
│   │   └── usage_limits.py          # 使用限额管理
│   │
│   ├── services/                    # 业务逻辑层
│   │   ├── retrieval_service.py     # 检索和语义缓存
│   │   ├── semantic_cache.py        # 语义相似度缓存实现
│   │   ├── usage_limiter.py         # 使用限额管理
│   │   ├── scheduler_service.py     # 定时任务调度
│   │   └── vector_store_cache.py    # 向量存储缓存
│   │
│   ├── loaders/                     # 文档加载器
│   │   └── document_loader.py       # PDF、Excel 等格式的加载
│   │
│   ├── processors/                  # 文本处理
│   │   └── text_processor.py        # 文本清洗和预处理
│   │
│   ├── vectorstore/                 # 向量数据库
│   │   └── chroma_store.py          # ChromaDB 集成
│   │
│   └── tools/                       # 工具函数
│       └── pdf_handler.py           # PDF 处理工具
│
└── tests/                           # 测试文件
    ├── test_rag_pipeline.py         # RAG 流程测试
    ├── test_usage_limiter.py        # 使用限额测试
    └── check_chroma_02.py           # 向量库检查
```

## API 接口概览

### 认证接口 (`/auth`)

- `POST /auth/register` - 用户注册
- `POST /auth/login` - 用户登录
- `POST /auth/refresh` - 刷新 JWT Token

### 文档管理 (`/documents`)

- `POST /documents/upload` - 上传文档
- `GET /documents/list` - 获取文档列表
- `DELETE /documents/{doc_id}` - 删除文档
- `GET /documents/{doc_id}` - 获取文档详情

### 聊天接口 (`/chat`)

- `POST /chat/query` - 发送查询
- `GET /chat/history` - 获取聊天历史

### 向量库管理 (`/chromadb`)

- `POST /chromadb/clean` - 清理向量库
- `GET /chromadb/status` - 获取向量库状态

### 使用限额 (`/usage-limits`)

- `GET /usage-limits/stats` - 获取使用统计
- `GET /usage-limits/remaining` - 获取剩余额度

## 核心功能说明

### 多层缓存系统

项目实现了**分层缓存架构**，在不同层级优化查询性能：

#### 1. 语义缓存层（Semantic Cache）
基于查询语义相似度的智能缓存系统：
- **工作原理**：计算新查询的 Embedding，与 Redis 缓存中的历史查询比对相似度
- **相似度阈值**：0.8（可配置），超过阈值直接返回缓存的历史结果
- **缓存过期时间**：5 小时（可配置）
- **隔离机制**：按 `collection_name` + `top_k` 维度隔离；Rerank 结果和普通结果分离存储
- **兼容性识别**：识别非 Rerank 缓存不能服务 Rerank 请求的情况

**缓存命中流程**：
```
新查询 → 计算 Embedding → 与历史查询比对相似度 → 相似度 > 0.8 → 返回缓存结果
                                                  ↓
                                         否 → 执行新查询 → 缓存结果
```

#### 2. 向量存储缓存层（VectorStore Cache）
自动缓存 ChromaDB 实例和相关数据：
- **缓存对象**：Chroma 连接实例，减少重复初始化开销
- **缓存淘汰**：LRU 机制，最多缓存 100 个不同的 Collection
- **过期时间**：30 分钟（可配置）
- **单例模式**：全局只维护一个缓存实例
- **自动清理**：后台定时清理过期缓存条目

#### 3. 缓存协同工作流
```
用户查询
   ↓
检查语义缓存
   ├─ 命中（相似度 > 0.8）✓ → 返回缓存结果
   │
   └─ 未命中 ↓

      获取向量存储（检查向量存储缓存）
      ├─ 缓存命中 ✓ → 使用缓存的 Chroma 实例
      │
      └─ 未命中 → 新建 Chroma 连接 → 缓存新实例
         ↓

         执行检索和 Rerank（可选）
         ↓

         缓存查询和结果到语义缓存
         ↓

         返回结果
```

### 文档处理与分块

项目采用**基于 Embedding 的语义分块**算法，确保文档块的语义连贯性：

#### 处理流程：
1. **文本提取** - 从原始文档（PDF/Excel）提取文本
2. **初步分块** - 固定长度分块（300 字符）
3. **语义计算** - 计算每个块的 Embedding
4. **相似度分析** - 计算相邻块的相似度
5. **动态合并** - 相似的相邻块自动合并
6. **元数据丰富** - 添加文件名、页码、位置等元数据
7. **向量化存储** - 存储到 ChromaDB

**支持格式**：
- **PDF** - 完整支持，按页面处理，自动检测空白页，大小限制 5MB
- **Excel** - 支持，按行列处理

### 使用限额管理

支持**细粒度的使用限制**，保障资源公平分配：

#### 限制类型：
- **上传限制**：每用户每日上传次数限制（默认 10 次）
- **问答限制**：每用户每日问答次数限制（默认 10 次）

#### 特性：
- **自动重置**：每日指定时间自动重置（默认早上 6 点，可配置）
- **自定义配置**：管理员接口可修改限额和重置时间
- **实时统计**：API 提供实时的使用统计和剩余额度查询
- **实现方式**：基于 Redis 原子操作，支持分布式场景

### 支持的模型

#### LLM 提供商（聊天模型）

| 提供商 | 模型名称 | 环境变量 | 模型类型 |
|-------|---------|---------|---------|
| **DeepSeek** | `deepseek-chat` | `DEEPSEEK_API_KEY` | API 远程模型（默认） |
| **OpenAI** | `gpt-3.5-turbo` | `OPENAI_API_KEY` | API 远程模型 |
| **Claude** | `claude-3-sonnet-20240229` | `ANTHROPIC_API_KEY` | API 远程模型 |
| **阿里云通义千问** | `qwen3-coder-plus` | `DASHSCOPE_API_KEY` | API 远程模型 |

#### 本地模型（Embedding & Reranker）

| 组件 | 模型名称 | 来源 | 离线支持 | 环境变量 |
|-----|---------|------|---------|---------|
| **Embedding** | `BAAI/bge-large-zh-v1.5` | HuggingFace | ✓ 支持 | `MODEL_PROVIDER=local` |
| **Reranker** | `BAAI/bge-reranker-large` | HuggingFace | ✓ 支持 | `MODEL_PROVIDER=local` |

**提供商选择**：
- 设置 `MODEL_PROVIDER=aliyun` 使用阿里云 DashScope API（需要 `DASHSCOPE_API_KEY`）
- 设置 `MODEL_PROVIDER=local` 使用本地模型（自动下载到本地缓存，支持离线运行）

## 架构与设计模式

本项目采用多种经典设计模式，确保代码的可维护性、可扩展性和高效性：

### 单例模式（Singleton Pattern）

#### 1. VectorStoreCache 类
**位置**：`src/services/vector_store_cache.py`

**目的**：确保全局只有一个向量存储缓存实例，避免重复初始化 Chroma 连接

**实现**：
```python
class VectorStoreCache:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

**优势**：
- 减少内存占用和初始化开销
- 确保缓存的全局一致性
- 支持 TTL 和 LRU 淘汰机制

#### 2. ModelFactoryEbd 模型工厂
**位置**：`models/model_factory_ebd.py`

**目的**：懒加载 Embedding 和 Reranker 模型，全局共享同一个模型实例

**特性**：
- 支持本地和云端模型的自动选择
- 第一次调用时自动下载并缓存模型
- 后续调用直接使用缓存的模型实例

#### 3. 全局模型管理器
**位置**：`models/model_factory.py`

**目的**：管理 LLM 模型的生命周期，支持运行时动态切换

**提供接口**：
```python
get_model_manager()        # 获取全局唯一的模型管理器
model_manager.set_model()  # 切换模型
model_manager.get_model()  # 获取当前模型
```

### 工厂模式（Factory Pattern）

#### 1. ModelFactory - LLM 模型工厂
**位置**：`models/model_factory.py`

**功能**：根据提供商和参数创建 LLM 模型实例

**特点**：
- 支持多个提供商（DeepSeek、OpenAI、Claude、阿里云）
- 自动处理 API Key 验证
- 支持自定义参数（温度、最大令牌数等）
- 统一返回 ChatOpenAI 接口

```python
model = ModelFactory.create_model(
    provider=ModelProvider.DEEPSEEK,
    temperature=0.7
)
```

#### 2. ModelFactoryEbd - Embedding 工厂
**位置**：`models/model_factory_ebd.py`

**功能**：创建和管理 Embedding 及 Reranker 模型

**特点**：
- 自动选择本地或云端 Embedding 模型
- 两种 Reranker 实现（本地 CrossEncoder 和云端 TextReRank）
- 统一的模型接口

#### 3. 快速创建函数
```python
create_pdf_loader()           # 创建 PDF 加载器
create_vectorstore_manager()  # 创建向量存储管理器
create_semantic_chunker()     # 创建语义分块器
```

### 策略模式（Strategy Pattern）

#### 1. 多个 Embedding 提供商
- **本地策略**：BAAI/bge-large-zh-v1.5（HuggingFace）
- **云端策略**：阿里云 DashScope API

根据 `MODEL_PROVIDER` 环境变量在运行时选择策略

#### 2. 多个 Reranker 实现
- **本地策略**：BAAI/bge-reranker-large（CrossEncoder）
- **云端策略**：阿里云 TextReRank API

#### 3. 多个 LLM 提供商
- 每个提供商有独立的初始化和调用逻辑
- 通过工厂方法统一接口

### 分层架构设计

```
┌─────────────────────────────────────────┐
│        FastAPI 路由层 (API Layer)        │
│  ├─ /auth       - 认证管理              │
│  ├─ /chat       - 聊天接口              │
│  ├─ /documents  - 文档管理              │
│  ├─ /chromadb   - 向量库管理            │
│  └─ /usage-limits - 使用限额             │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│      业务服务层 (Service Layer)          │
│  ├─ retrieval_service    - 检索服务     │
│  ├─ semantic_cache       - 语义缓存     │
│  ├─ usage_limiter        - 使用限额     │
│  ├─ vector_store_cache   - 存储缓存     │
│  ├─ semantic_chunker     - 语义分块     │
│  └─ scheduler_service    - 任务调度     │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│      数据访问层 (Data Layer)             │
│  ├─ vectorstore/chroma_store    - 向量库│
│  ├─ loaders/document_loader     - 加载器│
│  ├─ processors/text_processor   - 处理器│
│  └─ tools/pdf_handler           - 工具  │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│      外部依赖 (External Dependencies)    │
│  ├─ ChromaDB    - 向量数据库            │
│  ├─ Redis       - 缓存和限额             │
│  ├─ LLM APIs    - 模型服务              │
│  └─ HuggingFace - 本地模型加载           │
└─────────────────────────────────────────┘
```

### 模块化设计特性

1. **清晰的职责划分**
   - API 层：处理 HTTP 请求和响应
   - 服务层：实现核心业务逻辑
   - 数据层：管理数据存储和访问

2. **高内聚、低耦合**
   - 各模块通过接口依赖，而非具体实现
   - 支持快速替换和扩展

3. **易于扩展**
   - 添加新 LLM 提供商：在工厂类中增加新分支
   - 添加新文档格式：创建新的加载器
   - 添加新缓存策略：实现新的缓存类

## 开发指南

### 调试技巧

1. **查看实时日志**
```bash
uv run uvicorn main:app --reload --port 8000
```

2. **查看已运行的 API 进程**
```bash
lsof -ti:8000
```

3. **杀死进程**
```bash
lsof -ti:8000 | xargs kill -9
```

### 添加新的 LLM 提供商

由于项目采用工厂模式，添加新的 LLM 提供商非常便捷：

1. **在 `config/model_config.py` 中添加模型配置**
   ```python
   # 新增提供商的环境变量和默认模型名称
   {
       "provider_name": "new_provider",
       "api_key_env": "NEW_PROVIDER_API_KEY",
       "default_model": "model-name"
   }
   ```

2. **在 `models/model_factory.py` 中实现工厂逻辑**
   ```python
   @staticmethod
   def create_model(provider: ModelProvider, ...):
       if provider == ModelProvider.NEW_PROVIDER:
           # 初始化新提供商的模型
           return ChatOpenAI(
               model_name="...",
               base_url="...",
               api_key="..."
           )
   ```

3. **更新环境变量文档**
   - 在 `.env_copy` 中添加 `NEW_PROVIDER_API_KEY`
   - 更新 README 的模型支持表格

4. **（可选）添加提供商特定的参数**
   - 在 ModelFactory 中添加提供商特定的初始化参数

### 扩展文档处理能力

项目的模块化设计支持快速扩展文档处理能力：

**添加新的文档格式**（例如：Docx、Markdown）：

1. **创建新的加载器**
   ```python
   # src/loaders/docx_loader.py
   from langchain_community.document_loaders import Docx2txtLoader

   class DocxLoader:
       def load(self, file_path: str) -> List[Document]:
           loader = Docx2txtLoader(file_path)
           return loader.load()
   ```

2. **在文档上传接口中注册加载器**
   ```python
   # src/api/documents.py
   LOADERS = {
       ".pdf": PDFLoader,
       ".docx": DocxLoader,
       ".xlsx": ExcelLoader
   }
   ```

3. **更新前端和文档**
   - 更新 API 文档中的支持格式列表
   - 在 README 中更新"支持的文档格式"

**支持的文档处理流程**：
- 所有加载器返回统一的 `List[Document]` 格式
- 后续的语义分块、向量化等流程对所有格式保持一致
- 可根据需要在加载器中自定义文本提取逻辑

## 依赖管理

本项目使用 **uv** 作为包管理工具，所有依赖在 `pyproject.toml` 中声明：

```bash
# 安装依赖
uv sync

# 添加新依赖
uv add package-name

# 更新依赖
uv sync
```

### 主要依赖

- **FastAPI** - Web 框架
- **LangChain** - LLM 集成框架
- **ChromaDB** - 向量数据库
- **Redis** - 缓存和消息队列
- **APScheduler** - 定时任务调度
- **Pydantic** - 数据验证
- **PyMuPDF** - PDF 处理

## 故障排查

### Redis 连接失败

```
❌ Redis 连接失败
⚠️  语义缓存功能将不可用
```

**解决方案**：确保 Redis 服务已启动
```bash
# macOS (使用 Homebrew)
brew services start redis

# Linux
sudo systemctl start redis-server
```

### 端口被占用

```
❌ Address already in use
```

**解决方案**：杀死占用端口的进程或更换端口
```bash
# 杀死进程
lsof -ti:8000 | xargs kill -9

# 或使用不同的端口
uv run uvicorn main:app --port 8001
```

### 模型 API Key 缺失

**解决方案**：检查 `.env` 文件中是否配置了正确的 API Key

## 性能优化建议

1. **语义缓存** - 启用缓存以减少重复查询的成本
2. **批量操作** - 使用批量接口处理多个文档
3. **向量库清理** - 定期清理废弃的文档和向量
4. **Redis 监控** - 监控 Redis 内存使用情况

## 安全建议

- 🔐 **不要提交 `.env` 文件** - 使用 `.env_copy` 作为模板
- 🔑 **定期轮换 API Key** - 定期更新各服务的 API Key
- 🛡️ **启用 HTTPS** - 生产环境必须使用 HTTPS
- 📝 **审计日志** - 记录所有用户操作

## 部署

### 使用 uv 部署到新环境

```bash
# 在新服务器上
git clone <repository-url>
cd jakcy_rag_mcp

# 安装依赖（会使用 uv.lock 确保版本一致）
uv sync

# 配置环境变量
cp .env_copy .env
# 编辑 .env 文件

# 启动应用
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker 部署

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# 安装 uv
RUN pip install uv

# 复制项目文件
COPY . .

# 安装依赖
RUN uv sync

# 暴露端口
EXPOSE 8000

# 启动应用
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 贡献指南

欢迎提交 Issue 和 Pull Request！

## 许可证

本项目采用 MIT 许可证

## 联系方式

如有问题，请创建 Issue 或联系项目维护者。
