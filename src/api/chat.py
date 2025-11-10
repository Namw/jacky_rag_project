from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List, Tuple
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from src.services.retrieval_service import get_semantic_cache
from src.services.vector_store_cache import vectorstore_cache
from src.vectorstore.chroma_store import VectorStoreManager
from models.model_factory import ModelFactory
from config.model_config import ModelProvider
from src.api.auth import get_current_user, User
from src.services.retrieval_service import retrieve_with_cache
from src.services.usage_limiter import get_usage_limiter
from langchain_core.documents import Document

# --- 1. 全局变量和配置 ---

# 向量数据库和 LLM 实例（延迟初始化）
_vectorstore: Optional[VectorStoreManager] = None
_llm: Optional[ChatOpenAI] = None
_current_provider: ModelProvider = ModelProvider.DEEPSEEK
_default_top_k: int = 5
_default_temperature: float = 0.7
# --- 全局缓存实例 ---
_vectorstore_cache = None
# 默认系统提示词
DEFAULT_SYSTEM_PROMPT = """你是一个专业的知识助手。

你的任务是根据提供的参考资料回答用户的问题。

要求：
1. 仔细阅读参考资料，基于资料内容回答
2. 如果资料中没有相关信息，诚实地告诉用户"根据提供的资料，我无法回答这个问题"
3. 回答要准确、简洁、有条理
4. 可以适当引用资料中的关键信息
5. 不要编造资料中没有的内容"""

def get_llm() -> ChatOpenAI:
    """获取 LLM 实例"""
    global _llm
    if _llm is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM not initialized. Please initialize the service first."
        )
    return _llm


def init_chat_service(
    model_provider: ModelProvider = ModelProvider.DEEPSEEK,
    top_k: int = 5,
    temperature: float = 0.7
):
    """
    初始化聊天服务

    :param vectorstore_manager: 向量数据库管理器
    :param model_provider: LLM 提供商
    :param top_k: 默认检索文档数量
    :param temperature: LLM 温度
    """
    global _llm, _vectorstore_cache, _current_provider, _default_top_k, _default_temperature
    _vectorstore_cache = vectorstore_cache
    _current_provider = model_provider
    _default_top_k = top_k
    _default_temperature = temperature

    # 初始化 LLM
    _llm = ModelFactory.create_model(
        provider=model_provider,
        temperature=temperature
    )


# --- 2. Pydantic 模型 ---

class ChatRequest(BaseModel):
    """聊天请求模型"""
    question: str = Field(..., min_length=1, max_length=1000)
    document_id: Optional[str] = Field(None, description="文档ID，为空则检索所有文档")
    top_k: Optional[int] = Field(None, ge=1, le=20)
    use_rerank: bool = Field(False, description="是否启用 rerank")  # ⭐️ 新增
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)  # ⭐️ 新增
    return_sources: bool = Field(True)
    system_prompt: Optional[str] = Field(None)

class SourceInfo(BaseModel):
    """来源信息模型"""
    source: str = Field(..., description="文档来源")
    page: str = Field(..., description="页码")
    content: str = Field(..., description="相关内容摘要")
    score: float = Field(..., description="相似度分数")

class ChatResponse(BaseModel):
    """聊天响应模型"""
    answer: str = Field(..., description="回答内容")
    question: str = Field(..., description="原始问题")
    sources: Optional[List[SourceInfo]] = Field(None, description="参考来源列表")
    timestamp: str = Field(..., description="响应时间戳")
    model_provider: str = Field(..., description="使用的模型提供商")

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "根据劳动合同，期限为三年...",
                "question": "劳动合同的期限是多久？",
                "sources": [
                    {
                        "source": "labor_contract.pdf",
                        "page": "1",
                        "content": "合同期限为三年...",
                        "score": 0.85
                    }
                ],
                "timestamp": "2025-11-05T10:30:00",
                "model_provider": "deepseek"
            }
        }


class ModelSwitchRequest(BaseModel):
    """模型切换请求"""
    provider: str = Field(..., description="模型提供商名称")
    temperature: Optional[float] = Field(None, description="LLM 温度", ge=0.0, le=2.0)

    class Config:
        json_schema_extra = {
            "example": {
                "provider": "deepseek",
                "temperature": 0.7
            }
        }


class ModelInfo(BaseModel):
    """模型信息"""
    current_provider: str = Field(..., description="当前使用的模型提供商")
    available_providers: List[str] = Field(..., description="可用的模型提供商列表")
    temperature: float = Field(..., description="当前温度设置")



# --- 3. 核心功能函数 ---
def retrieve_documents(
        document_id: Optional[str],
        query: str,
        k: int,
        use_rerank: bool = False,
        threshold: Optional[float] = None
) -> List[Tuple[Document, float]]:
    """检索文档（使用带 TTL 的缓存）"""

    # 1. 获取客户端
    client = _vectorstore_cache.get_client()

    # 2. 获取 collection 列表
    if document_id:
        collection_names = [f"doc_{document_id.replace('-', '_')}"]
    else:
        all_collections = client.list_collections()
        collection_names = [col.name for col in all_collections]

    # 3. ⭐️ 从缓存中获取 vectorstore
    all_results = []
    for collection_name in collection_names:
        try:
            vectorstore = _vectorstore_cache.get(collection_name)

            results = retrieve_with_cache(
                vectorstore=vectorstore,
                query=query,
                collection_name=collection_name,
                top_k=k,
                use_rerank=use_rerank,
                threshold=threshold
            )

            all_results.extend(results)

        except Exception as e:
            print(f"Warning: Failed to retrieve from {collection_name}: {str(e)}")
            continue

    all_results.sort(key=lambda x: x[1], reverse=True)
    return all_results[:k] if document_id else all_results[:k * 2]


def build_prompt(
        query: str,
        documents: List[tuple],
        system_prompt: Optional[str] = None
) -> tuple[str, str]:
    """构建 Prompt"""
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    context_parts = []
    for i, (doc, score) in enumerate(documents, 1):
        source = doc.metadata.get('category', 'Unknown')  # 👈 改用 category

        # 👇 处理页码列表
        page_numbers = doc.metadata.get('page_numbers', [])
        if isinstance(page_numbers, list) and page_numbers:
            if len(page_numbers) == 1:
                page_str = f"第{format_page_numbers(doc.metadata.get('page_numbers'))}页"
            else:
                page_str = f"第{page_numbers[0]}-{page_numbers[-1]}页"
        else:
            page_str = "页码未知"

        content = doc.page_content

        context_parts.append(
            f"【参考资料 {i}】\n"
            f"来源: {source} ({page_str})\n"  # 👈 格式化后的页码
            f"内容: {content}\n"
        )

    context = "\n".join(context_parts)

    user_message = f"""参考资料：
{context}

---

用户问题：{query}

请基于上述参考资料回答问题。"""

    return sys_prompt, user_message

def generate_answer(
    query: str,
    documents: List[tuple],
    system_prompt: Optional[str] = None
) -> str:
    """
    生成答案

    :param query: 用户问题
    :param documents: 检索到的文档
    :param system_prompt: 自定义系统提示词
    :return: LLM 生成的答案
    """
    llm = get_llm()

    # 构建 Prompt
    sys_msg, user_msg = build_prompt(query, documents, system_prompt)

    # 调用 LLM
    messages = [
        SystemMessage(content=sys_msg),
        HumanMessage(content=user_msg)
    ]

    try:
        response = llm.invoke(messages)
        answer = response.content
        return answer
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM generation failed: {str(e)}"
        )


def format_page_numbers(page_numbers) -> str:
    """
    格式化页码显示

    :param page_numbers: 页码列表或其他格式
    :return: 格式化的页码字符串

    示例:
        [5] -> "5"
        [3, 4] -> "3-4"
        [1, 2, 3] -> "1-3"
        [] -> "N/A"
    """
    if not page_numbers:
        return "N/A"

    if isinstance(page_numbers, list):
        if len(page_numbers) == 1:
            return str(page_numbers[0])
        else:
            return f"{page_numbers[0]}-{page_numbers[-1]}"

    # 兼容旧数据（如果是单个数字）
    return str(page_numbers)

# --- 4. 路由 ---

router = APIRouter(
    prefix="/api/chat",
    tags=["Chat"],
)


@router.post("/query", response_model=ChatResponse)
async def chat_query(
        request: ChatRequest,
        current_user: User = Depends(get_current_user)
):
    """RAG 问答接口 - 使用统一的召回方案"""
    # 检查问答限额 ⭐️ 新增
    limiter = get_usage_limiter()
    can_query, error_msg = limiter.check_can_query(current_user.id)
    if not can_query:
        raise HTTPException(
            status_code=429,
            detail=error_msg
        )

    try:
        k = request.top_k or _default_top_k

        # 1. 检索文档（使用统一的召回逻辑）
        documents = retrieve_documents(
            document_id=request.document_id,  # ⭐️
            query=request.question,
            k=k,
            use_rerank=request.use_rerank,  # ⭐️
            threshold=request.threshold  # ⭐️
        )

        if not documents:
            return ChatResponse(
                answer="抱歉，没有找到相关的参考资料。",
                question=request.question,
                sources=None,
                timestamp=datetime.now().isoformat(),
                model_provider=_current_provider.value
            )

        # 2. 生成答案（逻辑不变）
        answer = generate_answer(
            query=request.question,
            documents=documents,
            system_prompt=request.system_prompt
        )

        # 3. 构建响应
        sources = None
        if request.return_sources:
            sources = []
            for doc, score in documents:
                # 👇 格式化页码
                sources.append(SourceInfo(
                    source=doc.metadata.get('category', 'Unknown'),
                    page=format_page_numbers(doc.metadata.get('page_numbers')),
                    content=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    score=float(score)
                ))

        # 增加问答计数 ⭐️ 新增
        limiter.increment_query(current_user.id)

        return ChatResponse(
            answer=answer,
            question=request.question,
            sources=sources,
            timestamp=datetime.now().isoformat(),
            model_provider=_current_provider.value
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )

@router.post("/switch-model")
async def switch_model(
    request: ModelSwitchRequest,
    current_user: User = Depends(get_current_user)
):
    """
    切换 LLM 模型提供商

    需要有效的 Token 才能访问
    仅管理员可以切换模型
    """
    # 检查权限
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin can switch models"
        )

    try:
        global _llm, _current_provider, _default_temperature

        # 验证提供商
        try:
            provider = ModelProvider(request.provider)
        except ValueError:
            available = [p.value for p in ModelProvider]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid provider. Available providers: {available}"
            )

        # 使用请求中的温度或当前温度
        temperature = request.temperature if request.temperature is not None else _default_temperature

        # 创建新的 LLM 实例
        _llm = ModelFactory.create_model(
            provider=provider,
            temperature=temperature
        )
        _current_provider = provider
        _default_temperature = temperature

        return {
            "message": f"Successfully switched to {provider.value}",
            "current_provider": provider.value,
            "temperature": temperature
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to switch model: {str(e)}"
        )

@router.get("/model-info", response_model=ModelInfo)
async def get_model_info(
    current_user: User = Depends(get_current_user)
):
    """
    获取当前模型信息

    需要有效的 Token 才能访问
    """
    return ModelInfo(
        current_provider=_current_provider.value,
        available_providers=[p.value for p in ModelProvider],
        temperature=_default_temperature
    )

@router.get("/cache/stats")
async def get_cache_stats(current_user: User = Depends(get_current_user)):
    """获取缓存统计"""
    from src.services.retrieval_service import get_semantic_cache

    cache = get_semantic_cache()
    if not cache:
        return {"error": "Cache not initialized"}

    stats = cache.get_cache_stats()
    return stats


@router.delete("/cache/clear")
async def clear_cache(
        document_id: Optional[str] = None,
        current_user: User = Depends(get_current_user)
):
    """清空缓存"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")

    cache = get_semantic_cache()
    if not cache:
        return {"error": "Cache not initialized"}

    cache.clear_cache(document_id)
    return {"message": "Cache cleared"}