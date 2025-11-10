"""
文档加载器基类
提供统一的接口，支持多种文档格式扩展
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union, Optional
from langchain_core.documents import Document


class BaseDocumentLoader(ABC):
    """
    文档加载器基类

    所有格式的加载器（PDF、DOC、EXCEL等）都应继承此类
    统一接口：load() -> List[Document]
    """

    def __init__(self, verbose: bool = True):
        """
        初始化文档加载器

        :param verbose: 是否打印处理信息
        """
        self.verbose = verbose

    @abstractmethod
    def load(self, file_path: Union[str, Path]) -> List[Document]:
        """
        加载单个文件

        :param file_path: 文件路径
        :return: Document 列表
        """
        pass

    def load_batch(self, file_paths: List[Union[str, Path]]) -> List[Document]:
        """
        批量加载多个文件

        :param file_paths: 文件路径列表
        :return: 所有文件的 Document 列表
        """
        all_documents = []

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"开始批量加载 {len(file_paths)} 个文件")
            print(f"{'=' * 60}\n")

        for i, file_path in enumerate(file_paths, 1):
            if self.verbose:
                print(f"[{i}/{len(file_paths)}] ", end="")

            try:
                docs = self.load(file_path)
                all_documents.extend(docs)
            except Exception as e:
                if self.verbose:
                    print(f"❌ 加载失败 {Path(file_path).name}: {e}")
                continue

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"批量加载完成: 共 {len(all_documents)} 个文档块")
            print(f"{'=' * 60}\n")

        return all_documents

    def load_directory(
            self,
            dir_path: Union[str, Path],
            recursive: bool = False,
            pattern: Optional[str] = None
    ) -> List[Document]:
        """
        加载目录下的所有文件

        :param dir_path: 目录路径
        :param recursive: 是否递归子目录
        :param pattern: 文件扩展名模式（如"*.pdf"），若为None则使用默认
        :return: 所有文件的 Document 列表
        """
        dir_path = Path(dir_path)

        if not dir_path.exists():
            raise FileNotFoundError(f"目录不存在: {dir_path}")

        if not dir_path.is_dir():
            raise ValueError(f"不是目录: {dir_path}")

        # 使用子类指定的文件扩展名，或使用传入的pattern
        file_pattern = pattern or self._get_file_pattern()

        # 查找所有匹配的文件
        if recursive:
            matching_files = list(dir_path.rglob(file_pattern))
        else:
            matching_files = list(dir_path.glob(file_pattern))

        if not matching_files:
            if self.verbose:
                print(f"⚠️  目录中没有找到匹配文件 ({file_pattern}): {dir_path}")
            return []

        if self.verbose:
            print(f"📁 在 {dir_path} 中找到 {len(matching_files)} 个文件 ({file_pattern})")

        return self.load_batch(matching_files)

    @abstractmethod
    def _get_file_pattern(self) -> str:
        """
        返回此加载器支持的文件扩展名模式

        :return: 文件模式（如"*.pdf"）
        """
        pass
