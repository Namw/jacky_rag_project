"""
语义分块器 - 基于 Embedding 的智能文本分块
支持动态阈值调整和滑动窗口相似度计算
"""

from langchain_text_splitters import TextSplitter
from typing import List
import numpy as np


class EmbeddingBasedTextSplitter(TextSplitter):
    """
    基于语义相似度的智能文本分块器

    工作流程:
    1. 初步按固定长度分块（支持重叠）
    2. 计算相邻块的语义相似度（使用滑动窗口）
    3. 动态调整合并阈值
    4. 相似度高的块合并，相似度低的块分离
    """

    def __init__(
            self,
            embedding_model,
            chunk_size: int = 300,
            chunk_overlap: float = 0.1,
            base_threshold: float = 0.8,
            dynamic_threshold: bool = True,
            window_size: int = 2,
            merge_separator: str = ""  # 中文默认无分隔符，英文可用 " "
    ):
        """
        初始化语义分块器

        :param embedding_model: embedding 模型对象（需实现 embed_documents 方法）
        :param chunk_size: 初始分块大小（字符数）
        :param chunk_overlap: 相邻块重叠比例，例如 0.1 表示 10%
        :param base_threshold: 基础相似度阈值（0-1之间）
        :param dynamic_threshold: 是否根据全局相似度分布动态调整阈值
        :param window_size: 滑动窗口大小，计算相邻块时向后看多少个块
        :param merge_separator: 合并块时的分隔符（中文用""，英文用" "）
        """
        super().__init__()
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.base_threshold = base_threshold
        self.dynamic_threshold = dynamic_threshold
        self.window_size = max(1, window_size)  # 至少为1
        self.merge_separator = merge_separator

        # 参数验证
        if not 0 <= chunk_overlap < 1:
            raise ValueError(f"chunk_overlap 必须在 [0, 1) 范围内，当前值: {chunk_overlap}")
        if not 0 <= base_threshold <= 1:
            raise ValueError(f"base_threshold 必须在 [0, 1] 范围内，当前值: {base_threshold}")

    def split_text(self, text: str) -> List[str]:
        """
        主入口：执行语义分块

        :param text: 待分块的文本
        :return: 分块后的文本列表
        """
        text = text.strip()
        if not text:
            return []

        # 1. 初步固定分块
        chunks = self._initial_split(text)

        # 如果只有一块或没有块，直接返回
        if len(chunks) <= 1:
            return chunks

        # 2. 计算相邻块 embedding
        try:
            embeddings = self.embedding_model.embed_documents(chunks)
        except Exception as e:
            print(f"⚠️  Embedding 计算失败，返回原始分块: {e}")
            return chunks

        # 3. 计算滑动窗口语义相似度
        similarities = self._pairwise_similarities(embeddings)

        # 4. 确定合并阈值
        threshold = self._get_threshold(similarities)

        # 5. 按相似度合并块
        merged_chunks = self._merge_chunks(chunks, similarities, threshold)

        return merged_chunks

    def _initial_split(self, text: str) -> List[str]:
        """
        初步按固定长度进行分块，支持重叠率控制

        :param text: 待分块文本
        :return: 初步分块结果
        """
        text_length = len(text)

        # 计算重叠长度和步长
        overlap_len = int(self.chunk_size * self.chunk_overlap)
        step = self.chunk_size - overlap_len

        # 防御性检查
        if step <= 0:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) 过大，导致步长 <= 0。"
                f"请确保 chunk_overlap < 1"
            )

        chunks = []
        start = 0

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk = text[start:end].strip()

            if chunk:  # 只添加非空块
                chunks.append(chunk)

            # 已到末尾，退出
            if end >= text_length:
                break

            # 向前推进，保留重叠部分
            start += step

        return chunks

    def _pairwise_similarities(self, embeddings: List[List[float]]) -> List[float]:
        """
        计算滑动窗口语义相似度

        每个块不仅与下一个块比较，还与后续 window_size 个块比较，取平均值
        这样可以捕捉更长范围的语义连贯性

        :param embeddings: 每个分块的 embedding 向量
        :return: 相似度列表（长度为 len(embeddings) - 1）
        """
        sims = []
        num_chunks = len(embeddings)

        for i in range(num_chunks - 1):
            current_emb = np.array(embeddings[i])
            window_sims = []

            # 向后查看 window_size 个块
            for offset in range(1, self.window_size + 1):
                next_idx = i + offset
                if next_idx >= num_chunks:
                    break

                next_emb = np.array(embeddings[next_idx])

                # 计算余弦相似度
                sim = self._cosine_similarity(current_emb, next_emb)
                window_sims.append(sim)

            # 当前块与后续 window_size 个块的平均相似度
            avg_sim = np.mean(window_sims) if window_sims else 0.0
            sims.append(avg_sim)

        return sims

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度

        :param vec_a: 向量A
        :param vec_b: 向量B
        :return: 余弦相似度（0-1之间）
        """
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        # 防止除零
        if norm_a == 0 or norm_b == 0:
            return 0.0

        return np.dot(vec_a, vec_b) / (norm_a * norm_b)

    def _get_threshold(self, sims: List[float]) -> float:
        """
        获取合并阈值（动态或固定）

        :param sims: 相似度列表
        :return: 阈值
        """
        if not self.dynamic_threshold:
            return self.base_threshold

        return self._dynamic_threshold(sims)

    def _dynamic_threshold(self, sims: List[float]) -> float:
        """
        根据整体相似度分布动态调整阈值

        策略：
        - 相似度高且集中 → 提高阈值（避免过度合并）
        - 相似度低且分散 → 降低阈值（允许更多合并）

        :param sims: 相似度列表
        :return: 动态阈值
        """
        if not sims:
            return self.base_threshold

        mean = np.mean(sims)
        std = np.std(sims)

        # 动态阈值 = 均值 + 0.5倍标准差
        # 限制在 [0.6, 0.95] 范围内
        dynamic = mean + 0.5 * std
        dynamic = np.clip(dynamic, 0.6, 0.95)

        return float(dynamic)

    def _merge_chunks(
            self,
            chunks: List[str],
            sims: List[float],
            threshold: float
    ) -> List[str]:
        """
        根据相似度阈值合并相邻块

        :param chunks: 原始分块列表
        :param sims: 相似度列表
        :param threshold: 合并阈值
        :return: 合并后的分块列表
        """
        if not chunks:
            return []

        merged = [chunks[0]]

        for i in range(1, len(chunks)):
            # 检查当前块与前一块的相似度
            if i - 1 < len(sims) and sims[i - 1] >= threshold:
                # 相似度高，合并到上一块
                merged[-1] += self.merge_separator + chunks[i]
            else:
                # 相似度低，作为新块
                merged.append(chunks[i])

        return merged

    def get_stats(self, text: str) -> dict:
        """
        获取分块统计信息（用于调试和优化）

        :param text: 待分块文本
        :return: 统计信息字典
        """
        chunks = self.split_text(text)

        chunk_lengths = [len(c) for c in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_chunk_length": np.mean(chunk_lengths) if chunks else 0,
            "min_chunk_length": min(chunk_lengths) if chunks else 0,
            "max_chunk_length": max(chunk_lengths) if chunks else 0,
            "total_chars": sum(chunk_lengths),
            "config": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "base_threshold": self.base_threshold,
                "window_size": self.window_size,
                "dynamic_threshold": self.dynamic_threshold
            }
        }