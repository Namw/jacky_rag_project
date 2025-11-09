# src/services/retrieval_service.py
"""
统一的召回服务
documents.py（测试）和 chat.py（实际使用）都调用这个
"""
import redis
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from pathlib import Path
from src.services.semantic_cache import SemanticQueryCache
from models.model_factory_ebd import ModelFactoryEbd

CHROMA_PERMANENT_DIR = Path("data/vectorstore/permanent")  # 正式库

# ==================== 全局模型实例（懒加载）====================
print(f"🔧 当前模型提供商: {ModelFactoryEbd.get_provider()}")

def get_embedding_model():
    """获取 Embedding 模型"""
    return ModelFactoryEbd.get_embedding_model()

def get_reranker_model():
    """获取 Reranker 模型"""
    return ModelFactoryEbd.get_reranker_model()

_semantic_cache: Optional[SemanticQueryCache] = None

def init_semantic_cache(
        redis_client: redis.Redis,
        similarity_threshold: float = 0.95,
        ttl_hours: int = 24
    ):
    """初始化语义缓存（需要在应用启动时调用）"""
    global _semantic_cache
    embedding_model = get_embedding_model()
    _semantic_cache = SemanticQueryCache(
        redis_client=redis_client,
        embedding_model=embedding_model,
        similarity_threshold=similarity_threshold,  # 可以调整
        ttl_hours=ttl_hours
    )
    print("✅ 语义缓存初始化成功")

def get_semantic_cache() -> Optional[SemanticQueryCache]:
    """获取语义缓存实例"""
    return _semantic_cache

def retrieve_with_cache(
        vectorstore: Chroma,
        query: str,
        collection_name: str,  # 👈 改为 collection_name，必传
        top_k: int = 5,
        use_rerank: bool = False,
        threshold: Optional[float] = None
) -> List[Tuple[Document, float]]:
    """
    带语义缓存的检索函数

    :param vectorstore: Chroma 向量库实例
    :param query: 查询文本
    :param collection_name: Collection 名称（用于缓存隔离）
    :param top_k: 返回 top-k 结果
    :param use_rerank: 是否启用 rerank
    :param threshold: 相似度阈值
    :return: [(Document, score), ...]
    """
    cache = get_semantic_cache()

    # 1. 尝试从缓存获取
    if cache:
        cached_results = cache.get(
            query=query,
            collection_name=collection_name,
            top_k=top_k,
            use_rerank=use_rerank
        )

        if cached_results is not None:
            # 应用 threshold 过滤（如果需要）
            if threshold is not None:
                cached_results = [
                    (doc, score) for doc, score in cached_results
                    if score >= threshold
                ]
            return cached_results

    # 2. 缓存未命中，执行真实检索
    results = retrieve_with_rerank(
        vectorstore=vectorstore,
        query=query,
        top_k=top_k,
        use_rerank=use_rerank,
        threshold=threshold
    )

    # 3. 存入缓存
    if cache and results:
        cache.set(
            query=query,
            collection_name=collection_name,
            top_k=top_k,
            use_rerank=use_rerank,
            results=results
        )

    return results

def retrieve_with_rerank(
        vectorstore: Chroma,
        query: str,
        top_k: int = 5,
        use_rerank: bool = False,
        threshold: Optional[float] = None
) -> List[Tuple[Document, float]]:
    """
    统一的召回函数

    :param vectorstore: Chroma 向量库实例
    :param query: 查询文本
    :param top_k: 返回 top-k 结果
    :param use_rerank: 是否启用 rerank 二次精排
    :param threshold: 相似度阈值（可选）
    :return: [(Document, score), ...] 列表
    """

    # 1. 召回（如果启用 rerank，多召回一些用于精排）
    initial_k = top_k * 3 if use_rerank else top_k

    results_with_scores = vectorstore.similarity_search_with_score(
        query,
        k=initial_k
    )
    # print(f"检索的文本分类：{results_with_scores[0][0].metadata['category']}")
    # 2. 转换为统一格式（distance → similarity）
    results = []
    for doc, distance in results_with_scores:
        similarity = 1 - distance  # Chroma 返回的是 distance

        # 阈值过滤
        if threshold is not None and similarity < threshold:
            continue

        results.append((doc, similarity))

    # 3. Rerank 二次精排（如果启用）
    if use_rerank and len(results) > 0:
        results = _rerank_results(query, results, top_k)

    # 4. 返回 top_k 个结果
    return results[:top_k]

def _rerank_results(
        query: str,
        results: List[Tuple[Document, float]],
        top_k: int
) -> List[Tuple[Document, float]]:
    """使用 Reranker 进行二次精排（兼容本地和阿里云模型）"""

    reranker_model = get_reranker_model()

    if reranker_model is None:
        print("⚠️ Reranker 不可用，返回原始结果")
        return results[:top_k]

    try:
        # 准备 query-document 对
        pairs = [[query, doc.page_content] for doc, _ in results]

        # 计算 rerank 分数（统一接口）
        rerank_scores = reranker_model.predict(pairs)

        # 更新分数并重新排序
        reranked = []
        for i, (doc, _) in enumerate(results):
            reranked.append((doc, float(rerank_scores[i])))

        # 按 rerank 分数降序排序
        reranked.sort(key=lambda x: x[1], reverse=True)

        return reranked[:top_k]

    except Exception as e:
        print(f"⚠️ Rerank 失败: {e}")
        return results[:top_k]

embedding_model = get_embedding_model()
