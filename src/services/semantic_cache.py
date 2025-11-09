# src/services/semantic_cache.py
"""
语义查询缓存 - 基于 query embedding 的相似度匹配
"""

import json
import redis
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from datetime import timedelta
from langchain_core.documents import Document

class SemanticQueryCache:
    """
    基于语义相似度的查询缓存

    缓存策略：
    1. 使用 query embedding 进行语义匹配
    2. Rerank=True 的缓存可以服务所有请求
    3. Rerank=False 的缓存只能服务 Rerank=False 的请求
    4. TTL 24小时
    """

    def __init__(
            self,
            redis_client: redis.Redis,
            embedding_model,
            similarity_threshold: float = 0.95,
            ttl_hours: int = 24
    ):
        """
        :param redis_client: Redis 客户端
        :param embedding_model: Embedding 模型（阿里云 DashScope）
        :param similarity_threshold: 语义相似度阈值
        :param ttl_hours: 缓存过期时间（小时）
        """
        self.redis = redis_client
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.ttl = timedelta(hours=ttl_hours)

        # Redis key 前缀
        self.QUERY_INDEX_KEY = "semantic_cache:queries"  # 存储所有 query 的索引
        self.RESULT_KEY_PREFIX = "semantic_cache:result:"  # 存储结果

    def _compute_query_embedding(self, query: str) -> np.ndarray:
        """计算 query 的 embedding"""
        try:
            # 阿里云 DashScope 的调用方式
            embedding = self.embedding_model.embed_query(query)
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            print(f"❌ Embedding 计算失败: {e}")
            raise

    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _make_cache_key(
            self,
            collection_name: str,  # 👈 改为 collection_name
            top_k: int
    ) -> str:
        """
        生成缓存键的组成部分

        collection_name 和 top_k 会影响结果，需要隔离缓存
        """
        return f"{collection_name}:{top_k}"

    def get(
            self,
            query: str,
            collection_name: str,  # 👈 改为 collection_name
            top_k: int,
            use_rerank: bool
    ) -> Optional[List[Tuple[Document, float]]]:
        """
        从缓存中获取结果

        :return: 如果命中返回 [(Document, score), ...]，否则返回 None
        """
        try:
            # 1. 计算当前 query 的 embedding
            query_embedding = self._compute_query_embedding(query)

            # 2. 生成缓存范围 key
            cache_scope = self._make_cache_key(collection_name, top_k)

            # 3. 获取该范围下的所有已缓存 query
            index_key = f"{self.QUERY_INDEX_KEY}:{cache_scope}"
            cached_queries_data = self.redis.hgetall(index_key)

            if not cached_queries_data:
                # print(f"🔍 缓存未命中: 该范围无缓存 (scope={cache_scope})")
                return None

            # 4. 遍历找到最相似的 query
            best_match = None
            best_similarity = 0.0

            for query_hash, query_info_json in cached_queries_data.items():
                query_info = json.loads(query_info_json)
                cached_embedding = np.array(query_info['embedding'], dtype=np.float32)

                # 计算相似度
                similarity = self._compute_similarity(query_embedding, cached_embedding)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = query_info

            # 5. 判断是否达到阈值
            if best_similarity < self.similarity_threshold:
                # print(f"🔍 缓存未命中: 相似度不足 (max={best_similarity:.3f}, threshold={self.similarity_threshold})")
                return None

            # 6. 检查 Rerank 兼容性
            cached_rerank = best_match['use_rerank']

            if not cached_rerank and use_rerank:
                # 缓存是普通结果，但请求要 rerank 结果 → 不能用
                # print(f"🔍 缓存不可用: 需要 rerank 但缓存为普通结果 (similarity={best_similarity:.3f})")
                return None

            # 7. 从 Redis 获取结果
            result_key = best_match['result_key']
            cached_result_json = self.redis.get(result_key)

            if not cached_result_json:
                # print(f"⚠️ 缓存过期: 结果已失效")
                # 清理索引
                self.redis.hdel(index_key, best_match['query_hash'])
                return None

            # 8. 反序列化结果
            cached_data = json.loads(cached_result_json)
            results = self._deserialize_results(cached_data['results'])

            print(f"✅ 缓存命中! collection={collection_name}, similarity={best_similarity:.3f}")

            return results

        except Exception as e:
            print(f"⚠️ 缓存获取异常: {e}")
            return None

    def set(
            self,
            query: str,
            collection_name: str,  # 👈 改为 collection_name
            top_k: int,
            use_rerank: bool,
            results: List[Tuple[Document, float]]
    ):
        """
        将结果存入缓存
        """
        try:
            # 1. 计算 query embedding
            query_embedding = self._compute_query_embedding(query)

            # 2. 生成唯一标识
            cache_scope = self._make_cache_key(collection_name, top_k)
            query_hash = self._hash_embedding(query_embedding)

            # 3. 序列化结果
            serialized_results = self._serialize_results(results)

            # 4. 存储结果数据（带 TTL）
            result_key = f"{self.RESULT_KEY_PREFIX}{cache_scope}:{query_hash}"
            result_data = {
                'results': serialized_results,
                'query_text': query,
                'use_rerank': use_rerank,
                'timestamp': self._get_timestamp()
            }
            self.redis.setex(
                result_key,
                self.ttl,
                json.dumps(result_data, ensure_ascii=False)
            )

            # 5. 更新索引（Query → Result 的映射）
            index_key = f"{self.QUERY_INDEX_KEY}:{cache_scope}"
            query_info = {
                'embedding': query_embedding.tolist(),
                'query_text': query,
                'use_rerank': use_rerank,
                'query_hash': query_hash,
                'result_key': result_key
            }
            self.redis.hset(
                index_key,
                query_hash,
                json.dumps(query_info, ensure_ascii=False)
            )
            # 索引也设置 TTL（稍长一点，避免提前过期）
            self.redis.expire(index_key, self.ttl + timedelta(hours=1))

            # print(f"💾 缓存已保存: collection={collection_name}, query='{query[:30]}...', results={len(results)}")

        except Exception as e:
            print(f"⚠️ 缓存存储异常: {e}")

    def _serialize_results(
            self,
            results: List[Tuple[Document, float]]
    ) -> List[Dict[str, Any]]:
        """序列化 Document 对象"""
        serialized = []
        for doc, score in results:
            serialized.append({
                'page_content': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score)
            })
        return serialized

    def _deserialize_results(
            self,
            serialized: List[Dict[str, Any]]
    ) -> List[Tuple[Document, float]]:
        """反序列化为 Document 对象"""
        results = []
        for item in serialized:
            doc = Document(
                page_content=item['page_content'],
                metadata=item['metadata']
            )
            results.append((doc, item['score']))
        return results

    def _hash_embedding(self, embedding: np.ndarray) -> str:
        """将 embedding 转换为唯一哈希（用作 Redis key）"""
        import hashlib
        # 取前几个维度生成哈希（减少计算量）
        truncated = embedding[:100].tobytes()
        return hashlib.md5(truncated).hexdigest()

    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()

    def clear_cache(self, collection_name: Optional[str] = None):  # 👈 改为 collection_name
        """清空缓存（文档更新时使用）"""
        try:
            if collection_name:
                # 清空特定 collection 的缓存
                pattern = f"{self.QUERY_INDEX_KEY}:{collection_name}:*"
            else:
                # 清空所有缓存
                pattern = f"{self.QUERY_INDEX_KEY}:*"

            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
                print(f"🗑️ 已清空 {len(keys)} 个缓存索引 (collection={collection_name or 'all'})")
        except Exception as e:
            print(f"⚠️ 清空缓存失败: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        try:
            all_index_keys = self.redis.keys(f"{self.QUERY_INDEX_KEY}:*")

            total_queries = 0
            rerank_count = 0
            collections = set()

            for index_key in all_index_keys:
                # 从 index_key 解析 collection 名称
                # 格式: semantic_cache:queries:collection_name:top_k
                key_parts = index_key.decode('utf-8').split(':')
                if len(key_parts) >= 3:
                    collections.add(key_parts[2])

                queries_data = self.redis.hgetall(index_key)
                total_queries += len(queries_data)

                for query_info_json in queries_data.values():
                    query_info = json.loads(query_info_json)
                    if query_info.get('use_rerank'):
                        rerank_count += 1

            return {
                'total_cached_queries': total_queries,
                'rerank_cached': rerank_count,
                'normal_cached': total_queries - rerank_count,
                'cache_scopes': len(all_index_keys),
                'collections_cached': list(collections)  # 👈 新增：显示哪些 collection 有缓存
            }
        except Exception as e:
            print(f"⚠️ 获取统计信息失败: {e}")
            return {}