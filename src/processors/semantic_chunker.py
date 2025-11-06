"""
语义分块处理器
职责：将纯文本转换为带元数据的 LangChain Document 对象
"""

from typing import List, Dict, Optional, Any
from langchain_core.documents import Document
from src.tools.embedding_text_splitter import EmbeddingBasedTextSplitter


class SemanticChunker:
    """
    语义分块处理器

    处理流程：
    1. 接收纯文本 + 元数据
    2. 使用 EmbeddingBasedTextSplitter 进行语义分块
    3. 为每个分块添加元数据
    4. 返回 LangChain Document 列表
    """

    def __init__(self, splitter: EmbeddingBasedTextSplitter):
        """
        初始化分块处理器

        :param splitter: EmbeddingBasedTextSplitter 实例
        """
        self.splitter = splitter

    def process_text(
            self,
            text: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        处理单个文本，返回分块后的 Document 列表

        :param text: 待处理的纯文本
        :param metadata: 元数据字典（可选）
        :return: Document 列表
        """
        if not text or not text.strip():
            return []

        # 使用语义分块器分块
        chunks = self.splitter.split_text(text)

        # 封装为 Document 对象
        documents = []
        base_metadata = metadata or {}

        for i, chunk in enumerate(chunks):
            # 合并基础元数据 + chunk_id
            doc_metadata = {
                **base_metadata,
                "chunk_id": i,
                "chunk_index": i,  # 兼容不同命名习惯
                "total_chunks": len(chunks)
            }

            doc = Document(
                page_content=chunk,
                metadata=doc_metadata
            )
            documents.append(doc)

        return documents

    def process_batch(
            self,
            texts: List[str],
            metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[Document]:
        """
        批量处理多个文本

        :param texts: 文本列表
        :param metadatas: 对应的元数据列表（可选，长度需与texts相同）
        :return: 所有文本的 Document 列表
        """
        if metadatas is not None and len(texts) != len(metadatas):
            raise ValueError(
                f"texts 和 metadatas 长度不一致: {len(texts)} vs {len(metadatas)}"
            )

        all_documents = []

        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else None
            docs = self.process_text(text, metadata)
            all_documents.extend(docs)

        return all_documents

    def process_with_source(
            self,
            text: str,
            source: str,
            additional_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        处理文本并自动添加来源信息

        :param text: 待处理文本
        :param source: 来源标识（如文件名、URL等）
        :param additional_metadata: 额外的元数据
        :return: Document 列表
        """
        metadata = {"source": source}

        if additional_metadata:
            metadata.update(additional_metadata)

        return self.process_text(text, metadata)

    def get_chunk_stats(self, text: str) -> Dict[str, Any]:
        """
        获取文本分块的统计信息（不生成 Document）

        :param text: 待分析文本
        :return: 统计信息字典
        """
        return self.splitter.get_stats(text)


# ============ 工厂函数 ============
def create_semantic_chunker(
        embedding_model,
        chunk_size: int = 300,
        chunk_overlap: float = 0.1,
        base_threshold: float = 0.8,
        dynamic_threshold: bool = True,
        window_size: int = 2,
        merge_separator: str = ""
) -> SemanticChunker:
    """
    快速创建 SemanticChunker 实例

    :param embedding_model: HuggingFace Embedding 模型
    :param chunk_size: 分块大小
    :param chunk_overlap: 重叠比例
    :param base_threshold: 基础相似度阈值
    :param dynamic_threshold: 是否动态调整阈值
    :param window_size: 滑动窗口大小
    :param merge_separator: 合并分隔符
    :return: SemanticChunker 实例
    """
    splitter = EmbeddingBasedTextSplitter(
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        base_threshold=base_threshold,
        dynamic_threshold=dynamic_threshold,
        window_size=window_size,
        merge_separator=merge_separator
    )

    return SemanticChunker(splitter)