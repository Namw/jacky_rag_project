import json

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
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
from src.services.chat_session_store import get_chat_session_store
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


class SessionCreateRequest(BaseModel):
    """创建会话请求"""
    title: Optional[str] = Field(None, description="会话标题")
    document_id: Optional[str] = Field(None, description="文档ID，为空则检索所有文档")
    system_prompt: Optional[str] = Field(None, description="会话级系统提示词")


class ChatSessionInfo(BaseModel):
    """会话信息"""
    session_id: str
    title: str
    document_id: Optional[str] = None
    system_prompt: Optional[str] = None
    created_at: str
    updated_at: str


class SessionMessageRequest(BaseModel):
    """会话消息请求"""
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: Optional[int] = Field(None, ge=1, le=20)
    use_rerank: bool = Field(False, description="是否启用 rerank")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    return_sources: bool = Field(True)


class ChatMessage(BaseModel):
    """消息记录"""
    message_id: str
    role: str
    content: str
    timestamp: str
    sources: Optional[List[SourceInfo]] = None


class SessionMessagesResponse(BaseModel):
    """会话消息列表响应"""
    session_id: str
    messages: List[ChatMessage]


class SessionMessageResponse(BaseModel):
    """发送会话消息响应"""
    session_id: str
    answer: str
    question: str
    sources: Optional[List[SourceInfo]] = None
    timestamp: str
    model_provider: str


class SessionDetailResponse(BaseModel):
    """会话详情响应"""
    session: ChatSessionInfo
    messages: List[ChatMessage]



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


def build_prompt_with_history(
        query: str,
        documents: List[tuple],
        history_messages: List[dict],
        system_prompt: Optional[str] = None
) -> tuple[str, str]:
    """构建包含多轮历史的 Prompt"""
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    context_parts = []
    for i, (doc, score) in enumerate(documents, 1):
        source = doc.metadata.get('category', 'Unknown')

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
            f"来源: {source} ({page_str})\n"
            f"内容: {content}\n"
        )

    context = "\n".join(context_parts)

    history_parts = []
    for item in history_messages:
        role = item.get("role", "user")
        role_name = "用户" if role == "user" else "助手"
        content = item.get("content", "")
        history_parts.append(f"{role_name}: {content}")

    history_text = "\n".join(history_parts) if history_parts else "（无历史对话）"

    user_message = f"""历史对话：
{history_text}

---

参考资料：
{context}

---

用户当前问题：{query}

请基于历史对话与参考资料回答问题。"""

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


def generate_answer_with_history(
    query: str,
    documents: List[tuple],
    history_messages: List[dict],
    system_prompt: Optional[str] = None
) -> str:
    """生成多轮会话答案"""
    llm = get_llm()

    sys_msg, user_msg = build_prompt_with_history(
        query=query,
        documents=documents,
        history_messages=history_messages,
        system_prompt=system_prompt
    )

    messages = [
        SystemMessage(content=sys_msg),
        HumanMessage(content=user_msg)
    ]

    try:
        response = llm.invoke(messages)
        return response.content
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


def _extract_chunk_text(chunk) -> str:
    """提取流式 chunk 的文本内容"""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)

    return str(content) if content else ""


def _format_sse_event(event: str, data: dict) -> str:
    """格式化 SSE 事件"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _build_sources(documents: List[tuple]) -> List[SourceInfo]:
    """构建来源信息"""
    sources = []
    for doc, score in documents:
        sources.append(SourceInfo(
            source=doc.metadata.get('category', 'Unknown'),
            page=format_page_numbers(doc.metadata.get('page_numbers')),
            content=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            score=float(score)
        ))
    return sources

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


@router.post("/sessions", response_model=ChatSessionInfo)
async def create_chat_session(
        request: SessionCreateRequest,
        current_user: User = Depends(get_current_user)
):
    """创建会话"""
    store = get_chat_session_store()
    session = store.create_session(
        user_id=current_user.id,
        title=request.title,
        document_id=request.document_id,
        system_prompt=request.system_prompt
    )
    return ChatSessionInfo(**session)


@router.get("/sessions", response_model=List[ChatSessionInfo])
async def list_chat_sessions(
        current_user: User = Depends(get_current_user)
):
    """会话列表"""
    store = get_chat_session_store()
    sessions = store.list_sessions(current_user.id)
    return [ChatSessionInfo(**item) for item in sessions]


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_chat_session_detail(
        session_id: str,
        current_user: User = Depends(get_current_user)
):
    """获取会话详情"""
    store = get_chat_session_store()

    session = store.get_session(current_user.id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = store.list_messages(current_user.id, session_id)

    parsed_messages = []
    for item in messages:
        source_items = item.get("sources") or None
        sources = None
        if source_items:
            sources = [SourceInfo(**src) for src in source_items]

        parsed_messages.append(ChatMessage(
            message_id=item["message_id"],
            role=item["role"],
            content=item["content"],
            timestamp=item["timestamp"],
            sources=sources
        ))

    return SessionDetailResponse(
        session=ChatSessionInfo(**session),
        messages=parsed_messages
    )


@router.get("/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
async def list_chat_session_messages(
        session_id: str,
        limit: int = 50,
        current_user: User = Depends(get_current_user)
):
    """获取会话消息列表"""
    store = get_chat_session_store()

    session = store.get_session(current_user.id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = store.list_messages(current_user.id, session_id, limit=limit)

    parsed_messages = []
    for item in messages:
        source_items = item.get("sources") or None
        sources = None
        if source_items:
            sources = [SourceInfo(**src) for src in source_items]

        parsed_messages.append(ChatMessage(
            message_id=item["message_id"],
            role=item["role"],
            content=item["content"],
            timestamp=item["timestamp"],
            sources=sources
        ))

    return SessionMessagesResponse(session_id=session_id, messages=parsed_messages)


@router.post(
    "/sessions/{session_id}/messages",
    response_model=SessionMessageResponse,
    responses={
        200: {
            "description": "普通 JSON 或 SSE 流式响应",
            "content": {
                "application/json": {},
                "text/event-stream": {}
            }
        }
    }
)
async def send_session_message(
        session_id: str,
        request: SessionMessageRequest,
        http_request: Request,
        stream: bool = Query(False, description="是否开启 SSE 流式响应"),
        current_user: User = Depends(get_current_user)
):
    """发送会话消息（多轮 + 上下文记忆）"""
    limiter = get_usage_limiter()
    can_query, error_msg = limiter.check_can_query(current_user.id)
    if not can_query:
        raise HTTPException(status_code=429, detail=error_msg)

    store = get_chat_session_store()
    session = store.get_session(current_user.id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if stream:
        async def event_generator():
            try:
                k = request.top_k or _default_top_k

                documents = retrieve_documents(
                    document_id=session.get("document_id"),
                    query=request.question,
                    k=k,
                    use_rerank=request.use_rerank,
                    threshold=request.threshold
                )

                if not documents:
                    store.append_message(
                        user_id=current_user.id,
                        session_id=session_id,
                        role="user",
                        content=request.question
                    )
                    no_answer = "抱歉，没有找到相关的参考资料。"
                    assistant_message = store.append_message(
                        user_id=current_user.id,
                        session_id=session_id,
                        role="assistant",
                        content=no_answer,
                        sources=None
                    )
                    limiter.increment_query(current_user.id)

                    yield _format_sse_event("message", {"delta": no_answer})
                    yield _format_sse_event("done", {
                        "session_id": session_id,
                        "answer": no_answer,
                        "question": request.question,
                        "sources": None,
                        "timestamp": assistant_message["timestamp"],
                        "model_provider": _current_provider.value
                    })
                    return

                history_messages = store.list_messages(current_user.id, session_id, limit=10)
                sys_msg, user_msg = build_prompt_with_history(
                    query=request.question,
                    documents=documents,
                    history_messages=history_messages,
                    system_prompt=session.get("system_prompt")
                )

                messages = [
                    SystemMessage(content=sys_msg),
                    HumanMessage(content=user_msg)
                ]

                yield _format_sse_event("start", {
                    "session_id": session_id,
                    "question": request.question,
                    "model_provider": _current_provider.value
                })

                answer_chunks = []
                llm = get_llm()
                for chunk in llm.stream(messages):
                    if await http_request.is_disconnected():
                        return

                    delta = _extract_chunk_text(chunk)
                    if not delta:
                        continue

                    answer_chunks.append(delta)
                    yield _format_sse_event("message", {"delta": delta})

                answer = "".join(answer_chunks).strip()
                if not answer:
                    answer = "（模型未返回内容）"

                sources = None
                if request.return_sources:
                    sources = _build_sources(documents)

                source_dicts = [item.model_dump() for item in sources] if sources else None

                store.append_message(
                    user_id=current_user.id,
                    session_id=session_id,
                    role="user",
                    content=request.question
                )
                assistant_message = store.append_message(
                    user_id=current_user.id,
                    session_id=session_id,
                    role="assistant",
                    content=answer,
                    sources=source_dicts
                )

                limiter.increment_query(current_user.id)

                if source_dicts is not None:
                    yield _format_sse_event("sources", {"sources": source_dicts})

                yield _format_sse_event("done", {
                    "session_id": session_id,
                    "answer": answer,
                    "question": request.question,
                    "sources": source_dicts,
                    "timestamp": assistant_message["timestamp"],
                    "model_provider": _current_provider.value
                })

            except HTTPException as e:
                yield _format_sse_event("error", {"detail": e.detail})
            except Exception as e:
                yield _format_sse_event("error", {"detail": f"Session query failed: {str(e)}"})

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    try:
        k = request.top_k or _default_top_k
        print(f'document_id:{session.get("document_id")}')
        documents = retrieve_documents(
            document_id=session.get("document_id"),
            query=request.question,
            k=k,
            use_rerank=request.use_rerank,
            threshold=request.threshold
        )

        if not documents:
            store.append_message(
                user_id=current_user.id,
                session_id=session_id,
                role="user",
                content=request.question
            )
            no_answer = "抱歉，没有找到相关的参考资料。"
            store.append_message(
                user_id=current_user.id,
                session_id=session_id,
                role="assistant",
                content=no_answer,
                sources=None
            )

            limiter.increment_query(current_user.id)

            return SessionMessageResponse(
                session_id=session_id,
                answer=no_answer,
                question=request.question,
                sources=None,
                timestamp=datetime.now().isoformat(),
                model_provider=_current_provider.value
            )

        history_messages = store.list_messages(current_user.id, session_id, limit=10)
        answer = generate_answer_with_history(
            query=request.question,
            documents=documents,
            history_messages=history_messages,
            system_prompt=session.get("system_prompt")
        )

        sources = None
        if request.return_sources:
            sources = _build_sources(documents)

        source_dicts = [item.model_dump() for item in sources] if sources else None

        store.append_message(
            user_id=current_user.id,
            session_id=session_id,
            role="user",
            content=request.question
        )
        assistant_message = store.append_message(
            user_id=current_user.id,
            session_id=session_id,
            role="assistant",
            content=answer,
            sources=source_dicts
        )

        limiter.increment_query(current_user.id)

        return SessionMessageResponse(
            session_id=session_id,
            answer=answer,
            question=request.question,
            sources=sources,
            timestamp=assistant_message["timestamp"],
            model_provider=_current_provider.value
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Session query failed: {str(e)}"
        )


@router.delete("/sessions/{session_id}")
async def delete_chat_session(
        session_id: str,
        current_user: User = Depends(get_current_user)
):
    """删除会话"""
    store = get_chat_session_store()
    deleted = store.delete_session(current_user.id, session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted"}

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