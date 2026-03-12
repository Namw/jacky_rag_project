from fastapi import UploadFile, File, HTTPException, APIRouter, Depends
from fastapi.responses import JSONResponse
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing import List, Optional
import re
import uuid
import os
from pathlib import Path
from datetime import datetime
import fitz  # PyMuPDF
import time
import chromadb
from src.api.auth import get_current_user, User
from src.api.chat import get_llm
from src.services.chroma_cleanup import delete_collection_completely
from src.services.retrieval_service import retrieve_with_rerank, embedding_model, CHROMA_PERMANENT_DIR
from src.services.vector_store_cache import vectorstore_cache
from src.services.usage_limiter import get_usage_limiter
from src.loaders.pdf_loader import create_pdf_loader
from src.processors.semantic_chunker import create_semantic_chunker

# 修改router的prefix和tags
router = APIRouter(
    prefix="/api/documents",
    tags=["Documents"],
)

# 配置
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Chroma存储路径
CHROMA_TEMP_DIR = Path("data/vectorstore/temp")  # 临时库
CHROMA_TEMP_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERMANENT_DIR.mkdir(parents=True, exist_ok=True)

# 简单的内存存储（生产环境应该用数据库）
documents_db = {}


class DocumentMetadata:
    """文档元数据存储类（注意：与 LangChain 的 Document 类区别开）"""
    def __init__(self, document_id: str, filename: str, filepath: str,
                 page_count: int, file_size: int, user_id: str):
        self.document_id = document_id
        self.filename = filename
        self.filepath = filepath
        self.page_count = page_count
        self.file_size = file_size
        self.user_id = user_id
        self.status = "uploaded"
        self.created_at = datetime.now()
        self.text_content = None
        self.chunks = None  # List[Document] - LangChain Document 对象列表
        self.chroma_collection_name = None
        self.permanent_collection_name = None  # 正式库collection名称
        self.confirmed_at = None  # 确认时间
        self.category = None
        self.page_char_ranges = None



def cleanup_temp_collection(collection_name: str):
    """删除Chroma临时collection（包括物理文件）"""
    delete_result = delete_collection_completely(
        collection_name=collection_name,
        persist_dir=str(CHROMA_TEMP_DIR),
        verbose=True
    )

    if not delete_result["collection_deleted"]:
        print(f"⚠️ 删除临时collection失败: {collection_name}")


def verify_document_ownership(document_id: str, user_id: str):
    """验证文档所有权"""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="文档不存在")

    doc = documents_db[document_id]

    if doc.user_id != user_id:
        raise HTTPException(status_code=403, detail="无权访问此文档")

    return doc


# ==================== Pydantic Models ====================

class ChunkRequest(BaseModel):
    """分块请求参数"""
    chunk_size: int = Field(default=500, ge=100, le=2000, description="分块大小（字符数）")
    overlap: int = Field(default=50, ge=0, le=500, description="重叠字符数")
    separator: str = Field(default="\n\n", description="分隔符")


class ChunkItem(BaseModel):
    """单个分块"""
    chunk_id: str
    content: str
    start_pos: int
    end_pos: int
    char_count: int
    index: int


class ChunkResponse(BaseModel):
    """分块响应"""
    document_id: str
    chunks: List[ChunkItem]
    total_chunks: int
    total_chars: int
    chunk_size: int
    overlap: int
    category: str


class VectorizeResponse(BaseModel):
    """向量化响应"""
    document_id: str
    status: str
    total_chunks: int
    embedding_dim: int
    message: str
    category: str


class SearchRequest(BaseModel):
    """搜索请求参数"""
    query: str = Field(..., min_length=1, description="搜索问题")
    top_k: int = Field(default=5, ge=1, le=20, description="返回top-k个最相关的chunks")
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="相似度阈值（可选）")
    use_rerank: bool = Field(default=False, description="是否启用rerank二次精排")  # 新增 ⭐️
    filter_category: Optional[str] = Field(default=None, description="按分类过滤（可选）")  # 👈 新增

class SearchResultItem(BaseModel):
    """单个搜索结果"""
    chunk_id: str
    chunk_index: int
    content: str
    similarity_score: float
    char_count: int
    start_pos: int
    end_pos: int


class SearchResponse(BaseModel):
    """搜索响应"""
    document_id: str
    query: str
    results: List[SearchResultItem]
    total_results: int
    search_time_ms: float


class ConfirmResponse(BaseModel):
    """确认入库响应"""
    document_id: str
    status: str
    permanent_collection_name: str
    total_chunks: int
    confirmed_at: str
    message: str

class PermanentChunkItem(BaseModel):
    """正式库中的单个分块"""
    chunk_id: str
    chunk_index: int
    content: str
    char_count: int
    start_pos: int
    end_pos: int
    metadata: dict

class PermanentDocumentResponse(BaseModel):
    """正式库文档查看响应"""
    document_id: str
    permanent_collection_name: str
    total_chunks: int
    chunks: List[PermanentChunkItem]
    page: int
    page_size: int
    has_more: bool

# ==================== 工具函数 ====================

def _get_pdf_loader():
    """获取 PDFLoader 实例（懒加载）"""
    return create_pdf_loader(
        embedding_model=embedding_model,
        chunk_size=300,
        chunk_overlap=0.1,
        base_threshold=0.8,
        dynamic_threshold=True,
        window_size=2,
        verbose=False
    )

def extract_category_from_chunks(
        chunks: List[Document],
        max_chunks: int = 3
) -> str:
    """
    使用 LLM 进行文档分类（返回分类名称字符串）
    支持 LangChain Document 对象列表

    :param chunks: LangChain Document 对象列表
    :param max_chunks: 使用前几个chunk进行分类
    :return: 分类名称字符串
    """
    llm = get_llm()

    # 提取样本文本（限制长度避免token过多）
    sample_text = "\n".join([
        f"片段{i + 1}: {doc.page_content[:300]}"  # 每个chunk只取300字
        for i, doc in enumerate(chunks[:max_chunks])
    ])

    # System Prompt - 自由分类模式
    system_prompt = """你是一个专业的文档分类助手。
你的任务是根据文档内容，为其确定一个恰当的分类标签。

分类要求：
1. 仔细阅读文档片段，理解其核心内容、用途和性质
2. 给出一个简洁、准确的分类名称（2-6个字）
3. 分类应该是通用的、标准的类型

常见文档类型参考（但不限于）：
- 人事类：简历、入职登记表、离职申请、员工花名册
- 法务类：劳动合同、保密协议、承诺书、授权委托书
- 管理类：管理制度、操作规程、工作流程、通知公告
- 财务类：财务报表、发票、收据、报销单
- 项目类：项目方案、技术文档、需求文档、测试报告
- 会议类：会议纪要、会议通知、会议议程

请仔细分析文档内容后，只返回一个最合适的分类名称，不要有其他内容。
例如：简历、劳动合同、管理制度、会议纪要等。"""

    user_prompt = f"""请对以下文档进行分类，只返回分类名称（2-6个字）：

{sample_text}

分类名称："""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    try:
        response = llm.invoke(messages)
        category = response.content.strip()

        # 清理可能的多余内容
        # 移除可能的标点符号、引号等
        category = category.replace('"', '').replace("'", "").replace('。', '').strip()

        # 如果返回内容过长，尝试提取第一个有效分类词
        if len(category) > 10:
            # 尝试匹配常见模式
            match = re.search(
                r'(简历|合同|制度|报表|纪要|方案|文档|协议|通知|申请|登记|承诺书|委托书|报销单|发票|收据)', category)
            if match:
                category = match.group(1)
            else:
                category = category[:6]  # 截取前6个字

        # 如果为空，返回默认值
        if not category:
            category = "其他文档"

        return category

    except Exception as e:
        print(f"LLM分类失败: {str(e)}")
        return "未分类"  # 👈 错误时返回默认字符串


def get_page_numbers(start_pos: int, end_pos: int, page_char_ranges: List[dict]) -> List[int]:
    """
    根据字符范围确定所有涉及的页码

    返回: [1, 2] 表示分块跨越第1页和第2页
    """
    pages = set()

    for page_info in page_char_ranges:
        page_start = page_info["start_char"]
        page_end = page_info["end_char"]

        # 判断分块与页面是否有重叠
        if not (end_pos <= page_start or start_pos >= page_end):
            pages.add(page_info["page_num"])

    return sorted(list(pages))

# ==================== API Endpoints ====================

@router.post("/upload")
async def upload_document(
        file: UploadFile = File(...),
        current_user: User = Depends(get_current_user)
):
    """上传PDF文件"""

    # 检查上传限额
    limiter = get_usage_limiter()
    can_upload, error_msg = limiter.check_can_upload(current_user.id)
    if not can_upload:
        raise HTTPException(
            status_code=429,
            detail=error_msg
        )

    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="只支持PDF文件")

    content = await file.read()
    file_size = len(content)

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"文件大小超过限制（最大5MB），当前文件: {file_size / 1024 / 1024:.2f}MB"
        )

    try:
        document_id = str(uuid.uuid4())
        filepath = UPLOAD_DIR / f"{document_id}.pdf"

        with open(filepath, "wb") as f:
            f.write(content)

        # 使用 PyMuPDF 提取文本和页码信息
        doc_pdf = fitz.open(filepath)
        page_count = doc_pdf.page_count

        text_content = ""
        page_char_ranges = []

        for page_num in range(page_count):
            page = doc_pdf[page_num]
            page_text = page.get_text()

            start_char = len(text_content)
            text_content += page_text + "\n"
            end_char = len(text_content)

            page_char_ranges.append({
                "page_num": page_num + 1,  # 页码从1开始
                "start_char": start_char,
                "end_char": end_char
            })

        doc_pdf.close()

        if not text_content.strip():
            os.remove(filepath)
            raise HTTPException(status_code=400, detail="PDF文件无法提取文本内容")

    except HTTPException:
        raise
    except Exception as e:
        if filepath.exists():
            os.remove(filepath)
        raise HTTPException(status_code=400, detail=f"PDF文件处理失败: {str(e)}")

    # 创建文档元数据
    doc = DocumentMetadata(
        document_id=document_id,
        filename=file.filename,
        filepath=str(filepath),
        page_count=page_count,
        file_size=file_size,
        user_id=current_user.username  # 使用 username
    )
    doc.text_content = text_content
    doc.page_char_ranges = page_char_ranges  # 保存页码映射

    documents_db[document_id] = doc

    # 增加上传计数
    new_count = limiter.increment_upload(current_user.id)
    stats = limiter.get_usage_stats(current_user.id)

    return JSONResponse(
        status_code=200,
        content={
            "document_id": document_id,
            "filename": file.filename,
            "file_size": file_size,
            "page_count": page_count,
            "status": "uploaded",
            "created_at": doc.created_at.isoformat(),
            "upload_count": new_count,
            "upload_limit": stats["upload_limit"],
            "upload_remaining": stats["upload_remaining"]
        }
    )

@router.post("/{document_id}/chunk", response_model=ChunkResponse)
async def chunk_document(
        document_id: str,
        request: ChunkRequest,
        current_user: User = Depends(get_current_user)
):
    """对文档进行分块（允许重复分块，自动覆盖旧数据）"""
    doc = verify_document_ownership(document_id, current_user.username)

    if doc.status not in ["uploaded", "chunked"]:
        raise HTTPException(
            status_code=400,
            detail=f"文档状态错误，当前状态: {doc.status}，只能对 uploaded 或 chunked 状态的文档进行分块"
        )

    if not doc.text_content:
        raise HTTPException(status_code=400, detail="文档没有文本内容")

    try:
        # 使用 SemanticChunker 进行语义分块
        chunker = create_semantic_chunker(
            embedding_model=embedding_model,
            chunk_size=request.chunk_size,
            chunk_overlap=request.overlap / 100.0 if request.overlap > 1 else request.overlap,  # 转换为比例
            base_threshold=0.8,
            dynamic_threshold=True,
            window_size=2,
            merge_separator=""
        )

        # 返回 Document 对象列表
        lang_chain_documents = chunker.process_text(
            text=doc.text_content,
            metadata={
                "document_id": document_id,
                "filename": doc.filename,
                "file_path": doc.filepath
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分块失败: {str(e)}")

    # 自动提取分类
    category = extract_category_from_chunks(lang_chain_documents, max_chunks=3)

    doc.category = category
    doc.chunks = lang_chain_documents  # 保存 LangChain Document 对象列表
    doc.status = "chunked"

    # 转换为 API 响应格式
    chunk_items = []
    for i, lc_doc in enumerate(lang_chain_documents):
        chunk_items.append(ChunkItem(
            chunk_id=f"chunk_{i}",
            content=lc_doc.page_content,
            start_pos=0,  # 语义分块不追踪字符位置
            end_pos=len(lc_doc.page_content),
            char_count=len(lc_doc.page_content),
            index=i
        ))

    return ChunkResponse(
        document_id=document_id,
        chunks=chunk_items,
        total_chunks=len(lang_chain_documents),
        total_chars=len(doc.text_content),
        chunk_size=request.chunk_size,
        overlap=request.overlap,
        category=category
    )


@router.post("/{document_id}/vectorize", response_model=VectorizeResponse)
async def vectorize_document(
        document_id: str,
        current_user: User = Depends(get_current_user)
):
    """对文档分块进行向量化（允许重复向量化，自动覆盖旧数据）"""
    doc = verify_document_ownership(document_id, current_user.username)

    if doc.status not in ["chunked", "vectorized"]:
        raise HTTPException(
            status_code=400,
            detail=f"文档状态错误，当前状态: {doc.status}，必须先完成分块"
        )

    if not doc.chunks or len(doc.chunks) == 0:
        raise HTTPException(status_code=400, detail="文档没有分块数据")

    # 如果已经向量化过，先清理旧的Chroma collection
    if doc.status == "vectorized" and doc.chroma_collection_name:
        try:
            cleanup_temp_collection(doc.chroma_collection_name)
            print(f"✅ 已清理旧的向量数据: {doc.chroma_collection_name}")
        except Exception as e:
            print(f"⚠️ 清理旧collection失败（忽略）: {e}")

    # 从 LangChain Document 对象中提取数据
    chunk_texts = [chunk.page_content for chunk in doc.chunks]
    chunk_ids = [f"chunk_{i}" for i in range(len(doc.chunks))]

    metadatas = [
        {
            "chunk_index": i,
            "char_count": len(chunk.page_content),
            "start_pos": 0,  # 语义分块不追踪位置
            "end_pos": len(chunk.page_content),
            "document_id": document_id,
            "category": doc.category if doc.category else "未分类",
            **chunk.metadata  # 合并 Document 的元数据
        }
        for i, chunk in enumerate(doc.chunks)
    ]

    try:
        collection_name = f"temp_{document_id.replace('-', '_')}"

        Chroma.from_texts(
            texts=chunk_texts,
            embedding=embedding_model,
            ids=chunk_ids,
            metadatas=metadatas,
            collection_name=collection_name,
            persist_directory=str(CHROMA_TEMP_DIR),
            collection_metadata={"hnsw:space": "cosine"}
        )

        doc.chroma_collection_name = collection_name

        sample_embedding = embedding_model.embed_query(chunk_texts[0])
        embedding_dim = len(sample_embedding)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"向量化失败: {str(e)}"
        )

    doc.status = "vectorized"
    return VectorizeResponse(
        document_id=document_id,
        status="vectorized",
        total_chunks=len(doc.chunks),
        embedding_dim=embedding_dim,
        category=doc.category if doc.category else "未分类",
        message=f"成功向量化 {len(doc.chunks)} 个文本块并存入Chroma临时库"
    )


@router.post("/{document_id}/search", response_model=SearchResponse)
async def search_document(
        document_id: str,
        request: SearchRequest,
        current_user: User = Depends(get_current_user)
):
    """文档召回测试 - 使用统一的召回方案"""
    start_time = time.time()

    doc = verify_document_ownership(document_id, current_user.username)

    if doc.status != "vectorized":
        raise HTTPException(status_code=400, detail="必须先完成向量化")

    try:
        # 加载临时库
        vectorstore = Chroma(
            collection_name=doc.chroma_collection_name,
            embedding_function=embedding_model,
            persist_directory=str(CHROMA_TEMP_DIR)
        )

        # ⭐️ 使用统一的召回函数
        results = retrieve_with_rerank(
            vectorstore=vectorstore,
            query=request.query,
            top_k=request.top_k,
            use_rerank=request.use_rerank,
            threshold=request.threshold
        )

        # 转换为响应格式
        search_results = []
        for doc_result, similarity in results:
            metadata = doc_result.metadata
            search_results.append(SearchResultItem(
                chunk_id=f"chunk_{metadata['chunk_index']}",
                chunk_index=metadata["chunk_index"],
                content=doc_result.page_content,
                similarity_score=round(similarity, 4),
                char_count=metadata["char_count"],
                start_pos=metadata["start_pos"],
                end_pos=metadata["end_pos"]
            ))

        search_time = (time.time() - start_time) * 1000

        return SearchResponse(
            document_id=document_id,
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            search_time_ms=round(search_time, 2)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


@router.post("/{document_id}/confirm", response_model=ConfirmResponse)
async def confirm_document(
        document_id: str,
        current_user: User = Depends(get_current_user)
):
    """
    确认入库 - 将临时数据迁移到正式库

    流程：
    1. 验证文档状态（必须是 vectorized）
    2. 从临时库读取所有数据
    3. 写入正式库
    4. 删除临时库
    5. 更新文档状态为 confirmed
    """
    doc = verify_document_ownership(document_id, current_user.username)

    # 1. 检查文档状态
    if doc.status != "vectorized":
        raise HTTPException(
            status_code=400,
            detail=f"文档状态错误，当前状态: {doc.status}，必须先完成向量化"
        )

    if not doc.chroma_collection_name:
        raise HTTPException(status_code=400, detail="文档未创建向量库")

    try:
        # 2. 加载临时collection
        temp_vectorstore = Chroma(
            collection_name=doc.chroma_collection_name,
            embedding_function=embedding_model,
            persist_directory=str(CHROMA_TEMP_DIR)
        )

        # 3. 获取所有数据
        temp_collection = temp_vectorstore._collection
        all_data = temp_collection.get(include=['documents', 'metadatas', 'embeddings'])

        if not all_data['ids']:
            raise HTTPException(status_code=400, detail="临时库中没有数据")

        # 4. 创建正式库collection
        permanent_collection_name = f"doc_{document_id.replace('-', '_')}"

        permanent_client = chromadb.PersistentClient(path=str(CHROMA_PERMANENT_DIR))

        try:
            # 删除已存在的同名collection（如果有）
            permanent_client.delete_collection(name=permanent_collection_name)
        except:
            pass

        # 创建新的正式collection（带时间戳元数据）
        now = datetime.now().isoformat()
        permanent_collection = permanent_client.create_collection(
            name=permanent_collection_name,
            metadata={
                "hnsw:space": "cosine",
                "created_at": doc.created_at.isoformat(),
                "confirmed_at": now,
                "document_id": document_id,
                "category": doc.category if doc.category else "未分类"
            }
        )

        # 5. 添加数据到正式库
        permanent_collection.add(
            ids=all_data['ids'],
            documents=all_data['documents'],
            metadatas=all_data['metadatas'],
            embeddings=all_data['embeddings']
        )

        print(f"✅ 数据已迁移到正式库: {permanent_collection_name}")

        # 6. 删除临时collection
        cleanup_temp_collection(doc.chroma_collection_name)

        # 7. 更新文档状态
        doc.status = "confirmed"
        doc.permanent_collection_name = permanent_collection_name
        doc.confirmed_at = datetime.now()

        # 8. 清理临时数据（可选，保留chunks便于查看）
        doc.chroma_collection_name = None
        vectorstore_cache.clear_client()

        return ConfirmResponse(
            document_id=document_id,
            status="confirmed",
            permanent_collection_name=permanent_collection_name,
            total_chunks=len(all_data['ids']),
            confirmed_at=doc.confirmed_at.isoformat(),
            message=f"文档已成功入库，共 {len(all_data['ids'])} 个文本块"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"确认入库失败: {str(e)}"
        )


@router.get("/{document_id}/permanent", response_model=PermanentDocumentResponse)
async def get_permanent_document(
        document_id: str,
        page: int = 1,
        page_size: int = 10,
        current_user: User = Depends(get_current_user)
):
    """
    查看正式库中的文档内容
    - 支持分页查看
    - 返回文本内容和metadata
    """
    # 验证文档所有权（注意：doc 是 DocumentMetadata 类）
    doc = verify_document_ownership(document_id, current_user.username)

    # 检查文档状态
    if doc.status != "confirmed":
        raise HTTPException(
            status_code=400,
            detail=f"文档状态错误，当前状态: {doc.status}，必须先确认入库（状态为confirmed）"
        )

    if not doc.permanent_collection_name:
        raise HTTPException(status_code=400, detail="文档未创建正式库")

    try:

        # 连接正式库
        client = chromadb.PersistentClient(path=str(CHROMA_PERMANENT_DIR))
        collection = client.get_collection(name=doc.permanent_collection_name)

        # 获取所有数据
        all_data = collection.get(include=['documents', 'metadatas'])

        total_chunks = len(all_data['ids'])

        # 按chunk_index排序
        chunks_with_metadata = []
        for i in range(total_chunks):
            chunks_with_metadata.append({
                'chunk_id': all_data['ids'][i],
                'content': all_data['documents'][i],
                'metadata': all_data['metadatas'][i]
            })

        # 按chunk_index排序
        chunks_with_metadata.sort(key=lambda x: x['metadata']['chunk_index'])

        # 分页
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        paginated_chunks = chunks_with_metadata[start_idx:end_idx]

        # 构建响应
        chunk_items = []
        for item in paginated_chunks:
            metadata = item['metadata']
            chunk_items.append(PermanentChunkItem(
                chunk_id=item['chunk_id'],
                chunk_index=metadata['chunk_index'],
                content=item['content'],
                char_count=metadata['char_count'],
                start_pos=metadata['start_pos'],
                end_pos=metadata['end_pos'],
                metadata=metadata
            ))

        return PermanentDocumentResponse(
            document_id=document_id,
            permanent_collection_name=doc.permanent_collection_name,
            total_chunks=total_chunks,
            chunks=chunk_items,
            page=page,
            page_size=page_size,
            has_more=end_idx < total_chunks
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"查询正式库失败: {str(e)}"
        )