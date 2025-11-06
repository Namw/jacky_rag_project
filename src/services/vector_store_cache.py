from datetime import datetime, timedelta
from typing import Dict,Optional, Tuple
from langchain_chroma import Chroma
import chromadb

from src.services.retrieval_service import embedding_model, CHROMA_PERMANENT_DIR

class VectorStoreCache:
    """VectorStore 缓存管理器（带过期时间）"""

    """向量存储缓存管理器"""
    _instance: Optional['VectorStoreCache'] = None
    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, ttl_minutes: int = 30, max_size: int = 100):
        """
        :param ttl_minutes: 缓存过期时间（分钟）
        :param max_size: 最大缓存数量
        """
        if not hasattr(self, '_initialized'):
            self._cache: Dict[str, Tuple[Chroma, datetime]] = {}
            self._ttl = timedelta(minutes=ttl_minutes)
            self._max_size = max_size
            self._client: Optional[chromadb.PersistentClient] = None

    def get_client(self) -> chromadb.PersistentClient:
        """获取 ChromaDB 客户端"""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(CHROMA_PERMANENT_DIR)
            )
        return self._client

    def clear_client(self):
        self._client = None

    def get(self, collection_name: str) -> Chroma:
        """
        获取 vectorstore（自动处理缓存）
        """
        now = datetime.now()

        # ⭐️ 检查缓存
        if collection_name in self._cache:
            vectorstore, timestamp = self._cache[collection_name]

            # 检查是否过期
            if now - timestamp < self._ttl:
                return vectorstore
            else:
                # 过期，删除缓存
                del self._cache[collection_name]

        # ⭐️ 缓存未命中或已过期，创建新实例
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=str(CHROMA_PERMANENT_DIR)
        )

        # ⭐️ LRU 淘汰：缓存满了删除最旧的
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._cache.keys(),
                             key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

        # 存入缓存
        self._cache[collection_name] = (vectorstore, now)
        return vectorstore

# 创建全局单例
vectorstore_cache = VectorStoreCache(ttl_minutes=30, max_size=100)