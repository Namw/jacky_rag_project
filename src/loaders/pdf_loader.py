"""
PDF 文档加载器
职责：从PDF文件中提取文本，并使用语义分块器处理
"""

from pathlib import Path
from typing import List, Optional, Union
import fitz  # PyMuPDF
from langchain_core.documents import Document
from src.loaders.base_loader import BaseDocumentLoader
from src.processors.semantic_chunker import SemanticChunker
from src.processors.semantic_chunker import create_semantic_chunker

class PDFLoader(BaseDocumentLoader):
    """
    PDF 文档加载器

    处理流程：
    1. 打开PDF文件
    2. 逐页提取文本
    3. 使用 SemanticChunker 对每页文本进行语义分块
    4. 为每个分块添加详细元数据（文件名、页码、路径等）
    5. 返回 LangChain Document 列表
    """

    def __init__(self, chunker: SemanticChunker, verbose: bool = True):
        """
        初始化 PDF 加载器

        :param chunker: SemanticChunker 实例
        :param verbose: 是否打印处理信息
        """
        super().__init__(verbose=verbose)
        self.chunker = chunker

    def load(self, file_path: Union[str, Path]) -> List[Document]:
        """
        加载单个 PDF 文件

        :param file_path: PDF 文件路径
        :return: Document 列表
        """
        file_path = Path(file_path)

        # 文件检查
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        if file_path.suffix.lower() != '.pdf':
            raise ValueError(f"不是PDF文件: {file_path}")

        if self.verbose:
            print(f"📄 正在加载: {file_path.name}")

        # 提取文本
        pages_data = self._extract_text(file_path)

        if not pages_data:
            if self.verbose:
                print(f"⚠️  {file_path.name} 没有提取到文本内容")
            return []

        # 分块处理
        documents = self._process_pages(pages_data, file_path)

        if self.verbose:
            print(f"✅ {file_path.name}: 共 {len(pages_data)} 页，生成 {len(documents)} 个分块")

        return documents

    def _get_file_pattern(self) -> str:
        """
        返回 PDF 加载器支持的文件模式

        :return: 文件模式
        """
        return "*.pdf"

    def _extract_text(self, file_path: Path) -> List[dict]:
        """
        从 PDF 中提取文本（按页）

        :param file_path: PDF 文件路径
        :return: 页面数据列表，格式: [{"page": 1, "text": "..."}, ...]
        """
        pages_data = []

        try:
            doc = fitz.open(str(file_path))

            for page_num, page in enumerate(doc):
                text = page.get_text("text").strip()

                # 只保留非空页
                if text:
                    pages_data.append({
                        "page": page_num + 1,  # 页码从1开始
                        "text": text
                    })

            doc.close()

        except Exception as e:
            raise RuntimeError(f"PDF解析失败: {e}")

        return pages_data

    def _process_pages(
            self,
            pages_data: List[dict],
            file_path: Path
    ) -> List[Document]:
        """
        处理所有页面，生成 Document 列表

        :param pages_data: 页面数据列表
        :param file_path: PDF 文件路径
        :return: Document 列表
        """
        all_documents = []

        for page_info in pages_data:
            page_num = page_info["page"]
            text = page_info["text"]

            # 使用 SemanticChunker 对每一页进行分块
            metadata = {
                "source": file_path.name,
                "file_path": str(file_path.absolute()),
                "page": page_num,
                "doc_type": "pdf"
            }

            page_documents = self.chunker.process_text(text, metadata)

            # 为每个分块添加页内编号
            for i, doc in enumerate(page_documents):
                doc.metadata["page_chunk_id"] = i
                doc.metadata["page_chunk_total"] = len(page_documents)

            all_documents.extend(page_documents)

        return all_documents

    def get_pdf_info(self, file_path: Union[str, Path]) -> dict:
        """
        获取 PDF 文件的基本信息（不进行分块）

        :param file_path: PDF 文件路径
        :return: PDF 信息字典
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        doc = fitz.open(str(file_path))

        # 统计文本页数和字符数
        text_pages = 0
        total_chars = 0

        for page in doc:
            text = page.get_text("text").strip()
            if text:
                text_pages += 1
                total_chars += len(text)

        info = {
            "filename": file_path.name,
            "file_path": str(file_path.absolute()),
            "file_size_mb": file_path.stat().st_size / (1024 * 1024),
            "total_pages": len(doc),
            "text_pages": text_pages,
            "empty_pages": len(doc) - text_pages,
            "total_chars": total_chars,
            "avg_chars_per_page": total_chars / text_pages if text_pages > 0 else 0,
            "metadata": doc.metadata
        }

        doc.close()

        return info


# ============ 工厂函数 ============
def create_pdf_loader(
        embedding_model,
        chunk_size: int = 300,
        chunk_overlap: float = 0.1,
        base_threshold: float = 0.8,
        dynamic_threshold: bool = True,
        window_size: int = 2,
        verbose: bool = True
) -> PDFLoader:
    """
    快速创建 PDFLoader 实例

    :param embedding_model: HuggingFace Embedding 模型
    :param chunk_size: 分块大小
    :param chunk_overlap: 重叠比例
    :param base_threshold: 基础相似度阈值
    :param dynamic_threshold: 是否动态调整阈值
    :param window_size: 滑动窗口大小
    :param verbose: 是否打印处理信息
    :return: PDFLoader 实例
    """

    chunker = create_semantic_chunker(
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        base_threshold=base_threshold,
        dynamic_threshold=dynamic_threshold,
        window_size=window_size,
        merge_separator=""  # 中文默认无分隔符
    )

    return PDFLoader(chunker, verbose=verbose)