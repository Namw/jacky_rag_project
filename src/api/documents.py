from fastapi import UploadFile, File, HTTPException, APIRouter, Depends
from fastapi.responses import JSONResponse
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from typing import List, Optional
from sentence_transformers import CrossEncoder
import uuid
import os
from pathlib import Path
from datetime import datetime
import fitz  # PyMuPDF
import time
import chromadb
from models.model_paths import get_models_cache_dir
from src.api.auth import get_current_user, User
from src.services.retrieval_service import retrieve_with_rerank, embedding_model, CHROMA_PERMANENT_DIR
from src.services.vector_store_cache import vectorstore_cache

# 修改router的prefix和tags
router = APIRouter(
    prefix="/api/documents",
    tags=["Documents"],
)

# 配置
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Chroma存储路径
CHROMA_TEMP_DIR = Path("data/vectorstore/temp")  # 临时库
CHROMA_TEMP_DIR.mkdir(exist_ok=True)
CHROMA_PERMANENT_DIR.mkdir(exist_ok=True)

# 简单的内存存储（生产环境应该用数据库）
documents_db = {}


class Document:
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
        self.chunks = None
        self.chroma_collection_name = None
        self.permanent_collection_name = None  # 正式库collection名称
        self.confirmed_at = None  # 确认时间

try:
    reranker_model = CrossEncoder(
        model_name_or_path = get_models_cache_dir() + '/BAAI-bge-reranker-large',
        max_length=512,
        device='cpu'
    )
    print("✅ Reranker模型加载成功")
except Exception as e:
    print(f"⚠️ Reranker模型加载失败: {e}")
    reranker_model = None

def cleanup_temp_collection(collection_name: str):
    """删除Chroma临时collection"""
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_TEMP_DIR))
        client.delete_collection(name=collection_name)
        print(f"🗑️ 删除临时collection: {collection_name}")
    except Exception as e:
        print(f"⚠️ 删除collection失败 {collection_name}: {e}")


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


class VectorizeResponse(BaseModel):
    """向量化响应"""
    document_id: str
    status: str
    total_chunks: int
    embedding_dim: int
    message: str


class SearchRequest(BaseModel):
    """搜索请求参数"""
    query: str = Field(..., min_length=1, description="搜索问题")
    top_k: int = Field(default=5, ge=1, le=20, description="返回top-k个最相关的chunks")
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="相似度阈值（可选）")
    use_rerank: bool = Field(default=False, description="是否启用rerank二次精排")  # 新增 ⭐️


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

def split_text_with_overlap(text: str, chunk_size: int, overlap: int, separator: str) -> List[dict]:
    """文本分块函数"""
    chunks = []
    text_length = len(text)
    start = 0
    index = 0

    while start < text_length:
        end = start + chunk_size

        if end < text_length:
            search_start = max(start, end - 100)
            search_end = min(text_length, end + 100)
            search_text = text[search_start:search_end]

            sep_pos = search_text.rfind(separator)
            if sep_pos != -1:
                end = search_start + sep_pos + len(separator)
        else:
            end = text_length

        chunk_content = text[start:end].strip()

        if chunk_content:
            chunk_id = f"chunk_{index}"
            chunks.append({
                "chunk_id": chunk_id,
                "content": chunk_content,
                "start_pos": start,
                "end_pos": end,
                "char_count": len(chunk_content),
                "index": index
            })
            index += 1

        start = end - overlap

        if start >= text_length or (end >= text_length and start == end - overlap):
            break

    return chunks

async def rerank_results(query: str, results: List[dict], top_k: int) -> List[dict]:
    """使用BGE reranker模型进行二次精排"""

    # 检查reranker是否可用
    if reranker_model is None:
        print("⚠️ Reranker不可用，返回原始结果")
        return results[:top_k]

    try:
        # 准备query-document对
        pairs = [[query, item['content']] for item in results]

        # 计算rerank分数
        rerank_scores = reranker_model.predict(pairs)

        # 更新相似度分数
        for i, item in enumerate(results):
            item['similarity_score'] = round(float(rerank_scores[i]), 4)

        # 按rerank分数降序排序
        results.sort(key=lambda x: x['similarity_score'], reverse=True)

        return results[:top_k]

    except Exception as e:
        print(f"⚠️ Rerank失败: {e}")
        return results[:top_k]


def extract_category_from_chunks(chunks: List[dict], max_chunks: int = 3) -> str:
    """
    从前几个chunk提取文档分类（关键词匹配）
    """
    # 合并前几个chunk的内容
    sample_text = " ".join([chunk["content"] for chunk in chunks[:max_chunks]])

    # 关键词匹配规则（可以根据实际情况调整）
    rules = {
        "简历": ["简历", "工作经验", "教育背景", "求职意向", "个人信息", "技能特长"],
        "劳动合同": ["劳动合同", "甲方", "乙方", "合同编号", "签订日期", "合同期限"],
        "公司管理制度": ["管理制度", "规章制度", "第一章", "总则", "第一条", "员工守则"],
        "财务报表": ["资产负债表", "利润表", "现金流量", "财务报表", "会计期间"],
        "会议纪要": ["会议纪要", "参会人员", "会议时间", "会议议题", "决议事项"]
    }

    # 计算每个分类的匹配分数
    category_scores = {}
    for category, keywords in rules.items():
        score = sum(1 for keyword in keywords if keyword in sample_text)
        category_scores[category] = score

    # 找到得分最高的分类
    best_category = max(category_scores.items(), key=lambda x: x[1])

    # 如果得分为0，说明都没匹配到
    if best_category[1] == 0:
        return "其他文档"

    return best_category[0]

# ==================== API Endpoints ====================

@router.post("/upload")
async def upload_document(
        file: UploadFile = File(...),
        current_user: User = Depends(get_current_user)
):
    """上传PDF文件"""

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

        doc_pdf = fitz.open(filepath)
        page_count = doc_pdf.page_count

        text_content = ""
        for page in doc_pdf:
            text_content += page.get_text() + "\n"

        doc_pdf.close()

        if not text_content.strip():
            os.remove(filepath)
            raise HTTPException(status_code=400, detail="PDF文件无法提取文本内容")

    except Exception as e:
        if filepath.exists():
            os.remove(filepath)
        raise HTTPException(status_code=400, detail=f"PDF文件处理失败: {str(e)}")

    doc = Document(
        document_id=document_id,
        filename=file.filename,
        filepath=str(filepath),
        page_count=page_count,
        file_size=file_size,
        user_id=current_user.username  # 使用 username
    )
    doc.text_content = text_content

    documents_db[document_id] = doc

    return JSONResponse(
        status_code=200,
        content={
            "document_id": document_id,
            "filename": file.filename,
            "file_size": file_size,
            "page_count": page_count,
            "status": "uploaded",
            "created_at": doc.created_at.isoformat()
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

    if request.overlap >= request.chunk_size:
        raise HTTPException(status_code=400, detail="overlap不能大于或等于chunk_size")

    try:
        chunks = split_text_with_overlap(
            text=doc.text_content,
            chunk_size=request.chunk_size,
            overlap=request.overlap,
            separator=request.separator
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分块失败: {str(e)}")

    # 👇 新增：自动提取分类
    category = extract_category_from_chunks(chunks, max_chunks=3)

    doc.category = category  # 👈 保存分类
    doc.chunks = chunks
    doc.status = "chunked"

    return ChunkResponse(
        document_id=document_id,
        chunks=[ChunkItem(**chunk) for chunk in chunks],
        total_chunks=len(chunks),
        total_chars=len(doc.text_content),
        chunk_size=request.chunk_size,
        overlap=request.overlap,
        category=category  # 👈 返回给前端
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

    chunk_texts = [chunk["content"] for chunk in doc.chunks]
    chunk_ids = [chunk["chunk_id"] for chunk in doc.chunks]

    metadatas = [
        {
            "chunk_index": chunk["index"],
            "char_count": chunk["char_count"],
            "start_pos": chunk["start_pos"],
            "end_pos": chunk["end_pos"],
            "document_id": document_id
        }
        for chunk in doc.chunks
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

        # 删除已存在的同名collection（如果有）
        try:
            permanent_client.delete_collection(name=permanent_collection_name)
        except:
            pass

        # 创建新的正式collection
        permanent_collection = permanent_client.create_collection(
            name=permanent_collection_name,
            metadata={"hnsw:space": "cosine"}
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
    # 验证文档所有权
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