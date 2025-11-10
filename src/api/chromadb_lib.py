import chromadb
import time
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path
from datetime import datetime
from src.api.auth import get_current_user, User
from src.services.chroma_cleanup import delete_collection_completely
from src.services.retrieval_service import retrieve_with_rerank, embedding_model, get_semantic_cache
from langchain_chroma import Chroma

# --- 1. 全局配置 ---
CHROMA_PERSIST_DIR = str(Path("data/vectorstore/permanent"))


# --- 2. Pydantic 模型 ---

class CollectionInfo(BaseModel):
    """集合信息模型"""
    collection_name: str = Field(..., description="集合名称")
    document_id: str = Field(..., description="文档ID")
    category: str = Field(..., description="文档分类")
    chunk_count: int = Field(..., description="文本块数量")
    created_at: Optional[str] = Field(default=None, description="创建时间（ISO格式）")
    confirmed_at: Optional[str] = Field(default=None, description="确认入库时间（ISO格式）")

    class Config:
        json_schema_extra = {
            "example": {
                "collection_name": "doc_f1fafd68_947c_49a8_9a27_3eba58d43e72",
                "document_id": "f1fafd68-947c-49a8-9a27-3eba58d43e72",
                "category": "劳动合同",
                "chunk_count": 16,
                "created_at": "2025-11-10T10:30:45.123456",
                "confirmed_at": "2025-11-10T10:35:20.654321"
            }
        }

class CollectionsResponse(BaseModel):
    """集合列表响应模型"""
    total: int = Field(..., description="集合总数")
    collections: List[CollectionInfo] = Field(..., description="集合列表")
    timestamp: str = Field(..., description="响应时间戳")

class DeleteCollectionRequest(BaseModel):
    """删除集合请求模型"""
    document_id: str = Field(..., description="要删除的文档ID")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "f1fafd68-947c-49a8-9a27-3eba58d43e72"
            }
        }

class ChunkDetail(BaseModel):
    """单个文本块详情"""
    chunk_id: str = Field(..., description="文本块ID")
    chunk_index: int = Field(..., description="文本块索引")
    content: str = Field(..., description="文本内容")
    char_count: int = Field(..., description="字符数")
    start_pos: int = Field(..., description="起始位置")
    end_pos: int = Field(..., description="结束位置")
    category: str = Field(..., description="文档分类")
    document_id: str = Field(..., description="文档ID")

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "chunk_0",
                "chunk_index": 0,
                "content": "这是文本内容...",
                "char_count": 500,
                "start_pos": 0,
                "end_pos": 500,
                "category": "劳动合同",
                "document_id": "f1fafd68-947c-49a8-9a27-3eba58d43e72"
            }
        }

class CollectionDetailResponse(BaseModel):
    """集合详情响应模型"""
    collection_name: str = Field(..., description="集合名称")
    document_id: str = Field(..., description="文档ID")
    category: str = Field(..., description="文档分类")
    total_chunks: int = Field(..., description="总文本块数")
    chunks: List[ChunkDetail] = Field(..., description="文本块列表")
    page: int = Field(..., description="当前页码")
    page_size: int = Field(..., description="每页数量")
    total_pages: int = Field(..., description="总页数")
    has_more: bool = Field(..., description="是否有下一页")
    created_at: Optional[str] = Field(default=None, description="创建时间（ISO格式）")
    confirmed_at: Optional[str] = Field(default=None, description="确认入库时间（ISO格式）")
    timestamp: str = Field(..., description="响应时间戳")

    class Config:
        json_schema_extra = {
            "example": {
                "collection_name": "doc_f1fafd68_947c_49a8_9a27_3eba58d43e72",
                "document_id": "f1fafd68-947c-49a8-9a27-3eba58d43e72",
                "category": "劳动合同",
                "total_chunks": 16,
                "chunks": [],
                "page": 1,
                "page_size": 10,
                "total_pages": 2,
                "has_more": True,
                "created_at": "2025-11-10T10:30:45.123456",
                "confirmed_at": "2025-11-10T10:35:20.654321",
                "timestamp": "2025-11-07T16:00:00.123456"
            }
        }

class CollectionSearchRequest(BaseModel):
    """集合检索请求参数"""
    query: str = Field(..., min_length=1, description="搜索问题")
    top_k: int = Field(default=5, ge=1, le=20, description="返回top-k个最相关的chunks")
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="相似度阈值（可选）")
    use_rerank: bool = Field(default=False, description="是否启用rerank二次精排")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "劳动合同的期限是多久？",
                "top_k": 5,
                "threshold": 0.5,
                "use_rerank": False
            }
        }

class CollectionSearchResultItem(BaseModel):
    """单个检索结果"""
    chunk_id: str = Field(..., description="文本块ID")
    chunk_index: int = Field(..., description="文本块索引")
    content: str = Field(..., description="文本内容")
    similarity_score: float = Field(..., description="相似度分数")
    char_count: int = Field(..., description="字符数")
    start_pos: int = Field(..., description="起始位置")
    end_pos: int = Field(..., description="结束位置")
    category: str = Field(..., description="文档分类")
    document_id: str = Field(..., description="文档ID")

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "chunk_0",
                "chunk_index": 0,
                "content": "劳动合同期限为三年...",
                "similarity_score": 0.8567,
                "char_count": 485,
                "start_pos": 0,
                "end_pos": 485,
                "category": "劳动合同",
                "document_id": "f1fafd68-947c-49a8-9a27-3eba58d43e72"
            }
        }

class CollectionSearchResponse(BaseModel):
    """集合检索响应"""
    document_id: str = Field(..., description="文档ID")
    collection_name: str = Field(..., description="集合名称")
    query: str = Field(..., description="查询问题")
    results: List[CollectionSearchResultItem] = Field(..., description="检索结果列表")
    total_results: int = Field(..., description="结果总数")
    search_time_ms: float = Field(..., description="检索耗时（毫秒）")
    use_rerank: bool = Field(..., description="是否使用了rerank")
    timestamp: str = Field(..., description="响应时间戳")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "f1fafd68-947c-49a8-9a27-3eba58d43e72",
                "collection_name": "doc_f1fafd68_947c_49a8_9a27_3eba58d43e72",
                "query": "劳动合同的期限是多久？",
                "results": [],
                "total_results": 5,
                "search_time_ms": 123.45,
                "use_rerank": False,
                "timestamp": "2025-11-07T16:30:00.123456"
            }
        }

# --- 3. 核心功能函数 ---

def get_chroma_client():
    """获取ChromaDB客户端"""
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        return client
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to connect to ChromaDB: {str(e)}"
        )

def document_id_to_collection_name(document_id: str) -> str:
    """将文档ID转换为集合名称"""
    return f"doc_{document_id.replace('-', '_')}"

def get_collection_info(client, collection_name: str) -> CollectionInfo:
    """获取单个集合的详细信息"""
    try:
        collection = client.get_collection(name=collection_name)

        # 获取所有数据的 metadata（只需要第一个即可，因为同一文档的分类相同）
        data = collection.get(
            include=['metadatas'],
            limit=1  # 只获取第一个 chunk 的 metadata
        )

        # 提取信息
        if data['metadatas'] and len(data['metadatas']) > 0:
            first_metadata = data['metadatas'][0]
            category = first_metadata.get('category', '未知分类')
            document_id = first_metadata.get('document_id', '未知ID')
        else:
            category = '未知分类'
            document_id = '未知ID'

        # 获取总文档数量
        total_chunks = collection.count()

        # 从 collection 的 metadata 中获取时间戳信息
        col_metadata = collection.metadata or {}
        created_at = col_metadata.get('created_at')
        confirmed_at = col_metadata.get('confirmed_at')

        return CollectionInfo(
            collection_name=collection_name,
            document_id=document_id,
            category=category,
            chunk_count=total_chunks,
            created_at=created_at,
            confirmed_at=confirmed_at
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get collection info for {collection_name}: {str(e)}"
        )

# --- 4. 路由 ---

router = APIRouter(
    prefix="/api/collections",
    tags=["Collections"],
)

@router.get("/list", response_model=CollectionsResponse)
async def list_collections(
        current_user: User = Depends(get_current_user)
):
    """
    获取所有ChromaDB集合列表

    返回所有集合的基本信息，包括:
    - 集合名称
    - 文档ID
    - 文档分类
    - 文本块数量
    """
    try:
        client = get_chroma_client()

        # 获取所有集合
        all_collections = client.list_collections()

        if not all_collections:
            return CollectionsResponse(
                total=0,
                collections=[],
                timestamp=datetime.now().isoformat()
            )

        # 收集每个集合的详细信息
        collections_info = []
        for col in all_collections:
            try:
                info = get_collection_info(client, col.name)
                collections_info.append(info)
            except Exception as e:
                print(f"Warning: Failed to get info for collection {col.name}: {str(e)}")
                continue

        return CollectionsResponse(
            total=len(collections_info),
            collections=collections_info,
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list collections: {str(e)}"
        )

@router.delete("/delete")
async def delete_collection(
        request: DeleteCollectionRequest,
        current_user: User = Depends(get_current_user)
):
    """
    删除指定的文档集合（包括磁盘文件）

    需要管理员权限
    """
    # 检查权限
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin can delete collections"
        )

    try:
        client = get_chroma_client()

        # 将 document_id 转换为 collection_name
        collection_name = document_id_to_collection_name(request.document_id)

        # 检查集合是否存在，并获取信息
        try:
            collection = client.get_collection(name=collection_name)
            chunk_count = collection.count()

            # 获取分类信息
            data = collection.get(include=['metadatas'], limit=1)
            category = "未知分类"
            if data['metadatas'] and len(data['metadatas']) > 0:
                category = data['metadatas'][0].get('category', '未知分类')

        except Exception:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{request.document_id}' not found"
            )

        delete_result = delete_collection_completely(
            collection_name=collection_name,
            persist_dir=CHROMA_PERSIST_DIR,
            verbose=True
        )
        folder_uuid = delete_result["folder_uuid"]
        deleted_folder = delete_result["folder_deleted"]

        cache = get_semantic_cache()
        if cache:
            try:
                cache.clear_cache(collection_name=collection_name)
                print(f"✅ 已清理文档 {request.document_id} 的缓存")
            except Exception as e:
                print(f"⚠️ 清理缓存失败: {e}")

        return {
            "message": "Document deleted successfully",
            "document_id": request.document_id,
            "collection_name": collection_name,
            "folder_uuid": folder_uuid,
            "category": category,
            "deleted_chunks": chunk_count,
            "deleted_folder": deleted_folder,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )

@router.get("/detail/{document_id}", response_model=CollectionDetailResponse)
async def get_collection_detail(
        document_id: str,
        page: int = 1,
        page_size: int = 10,
        current_user: User = Depends(get_current_user)
):
    """
    获取集合详情（分页查看文本块）

    参数:
    - document_id: 文档ID
    - page: 页码（从1开始）
    - page_size: 每页数量（1-50）
    """
    # 参数验证
    if page < 1:
        raise HTTPException(status_code=400, detail="页码必须大于0")

    if page_size < 1 or page_size > 50:
        raise HTTPException(status_code=400, detail="每页数量必须在1-50之间")

    try:
        client = get_chroma_client()

        # 将 document_id 转换为 collection_name
        collection_name = document_id_to_collection_name(document_id)

        # 检查 collection 是否存在
        try:
            collection = client.get_collection(name=collection_name)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{document_id}' not found"
            )

        # 获取所有数据
        all_data = collection.get(include=['documents', 'metadatas'])

        total_chunks = len(all_data['ids'])

        if total_chunks == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Collection is empty"
            )

        # 提取基本信息（从第一个chunk的metadata）
        first_metadata = all_data['metadatas'][0]
        category = first_metadata.get('category', '未知分类')

        # 按 chunk_index 排序
        chunks_with_metadata = []
        for i in range(total_chunks):
            metadata = all_data['metadatas'][i]
            chunks_with_metadata.append({
                'chunk_id': all_data['ids'][i],
                'content': all_data['documents'][i],
                'metadata': metadata
            })

        # 按 chunk_index 排序
        chunks_with_metadata.sort(key=lambda x: x['metadata'].get('chunk_index', 0))

        # 分页
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        paginated_chunks = chunks_with_metadata[start_idx:end_idx]

        # 构建响应
        chunk_items = []
        for item in paginated_chunks:
            metadata = item['metadata']
            chunk_items.append(ChunkDetail(
                chunk_id=item['chunk_id'],
                chunk_index=metadata.get('chunk_index', 0),
                content=item['content'],
                char_count=metadata.get('char_count', len(item['content'])),
                start_pos=metadata.get('start_pos', 0),
                end_pos=metadata.get('end_pos', 0),
                category=metadata.get('category', '未知分类'),
                document_id=metadata.get('document_id', document_id)
            ))

        # 计算总页数
        total_pages = (total_chunks + page_size - 1) // page_size

        # 从 collection 的 metadata 中获取时间戳信息
        col_metadata = collection.metadata or {}
        created_at = col_metadata.get('created_at')
        confirmed_at = col_metadata.get('confirmed_at')

        return CollectionDetailResponse(
            collection_name=collection_name,
            document_id=document_id,
            category=category,
            total_chunks=total_chunks,
            chunks=chunk_items,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_more=end_idx < total_chunks,
            created_at=created_at,
            confirmed_at=confirmed_at,
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get collection detail: {str(e)}"
        )

@router.post("/search/{document_id}", response_model=CollectionSearchResponse)
async def search_collection(
        document_id: str,
        request: CollectionSearchRequest,
        current_user: User = Depends(get_current_user)
):
    """
    在指定的集合中进行检索

    参数:
    - document_id: 文档ID
    - query: 检索问题
    - top_k: 返回结果数量
    - threshold: 相似度阈值（可选）
    - use_rerank: 是否启用二次精排
    """
    start_time = time.time()

    try:
        client = get_chroma_client()

        # 将 document_id 转换为 collection_name
        collection_name = document_id_to_collection_name(document_id)

        # 检查 collection 是否存在
        try:
            collection = client.get_collection(name=collection_name)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{document_id}' not found"
            )

        # 检查 collection 是否为空
        if collection.count() == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Collection is empty"
            )

        # 加载向量库
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=str(CHROMA_PERSIST_DIR)
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
            search_results.append(CollectionSearchResultItem(
                chunk_id=f"chunk_{metadata.get('chunk_index', 0)}",
                chunk_index=metadata.get('chunk_index', 0),
                content=doc_result.page_content,
                similarity_score=round(similarity, 4),
                char_count=metadata.get('char_count', len(doc_result.page_content)),
                start_pos=metadata.get('start_pos', 0),
                end_pos=metadata.get('end_pos', 0),
                category=metadata.get('category', '未知分类'),
                document_id=metadata.get('document_id', document_id)
            ))

        search_time = (time.time() - start_time) * 1000

        return CollectionSearchResponse(
            document_id=document_id,
            collection_name=collection_name,
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            search_time_ms=round(search_time, 2),
            use_rerank=request.use_rerank,
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )