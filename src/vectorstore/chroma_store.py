"""
Chroma 向量数据库管理模块
负责文档的存储、检索、删除等操作
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from models.model_paths import get_models_cache_dir


class VectorStoreManager:
    """
    向量数据库管理器

    功能：
    - 添加文档到向量数据库
    - 相似度检索
    - 文档管理（查看、删除）
    - 数据库统计
    """

    def __init__(
            self,
            embedding_model: HuggingFaceEmbeddings,
            persist_directory: str = "./data/vectorstore",
            collection_name: str = "rag_documents"
    ):
        """
        初始化向量数据库管理器

        :param embedding_model: HuggingFaceEmbeddings 实例
        :param persist_directory: 数据库持久化目录
        :param collection_name: 集合名称
        """
        self.embedding_model = embedding_model
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name

        # 确保目录存在
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # 初始化或加载 Chroma
        self.vectorstore = self._init_vectorstore()

    def _init_vectorstore(self) -> Chroma:
        """
        初始化或加载 Chroma 向量数据库

        :return: Chroma 实例
        """
        # Chroma 会自动检测目录是否存在数据
        # 如果存在则加载，不存在则创建新的
        vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=str(self.persist_directory)
        )

        return vectorstore

    def add_documents(
            self,
            documents: List[Document],
            verbose: bool = True
    ) -> List[str]:
        """
        添加文档到向量数据库

        :param documents: Document 列表
        :param verbose: 是否打印详细信息
        :return: 文档ID列表
        """
        if not documents:
            if verbose:
                print("⚠️  没有文档需要添加")
            return []

        if verbose:
            print(f"📥 正在添加 {len(documents)} 个文档块到向量数据库...")

        try:
            # 添加文档并获取ID
            ids = self.vectorstore.add_documents(documents)

            if verbose:
                print(f"✅ 成功添加 {len(ids)} 个文档块")

                # 统计来源
                sources = set(doc.metadata.get('source', 'Unknown') for doc in documents)
                print(f"   来源文件: {', '.join(sources)}")

            return ids

        except Exception as e:
            if verbose:
                print(f"❌ 添加文档失败: {e}")
            raise

    def search(
            self,
            query: str,
            k: int = 5,
            filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        相似度检索

        :param query: 查询文本
        :param k: 返回top-k个结果
        :param filter_dict: 元数据过滤条件，如 {"source": "file.pdf"}
        :return: Document 列表（按相似度排序）
        """
        try:
            if filter_dict:
                results = self.vectorstore.similarity_search(
                    query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vectorstore.similarity_search(query, k=k)

            return results

        except Exception as e:
            print(f"❌ 检索失败: {e}")
            return []

    def search_with_score(
            self,
            query: str,
            k: int = 5,
            filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """
        相似度检索（带分数）

        :param query: 查询文本
        :param k: 返回top-k个结果
        :param filter_dict: 元数据过滤条件
        :return: (Document, score) 元组列表，score越小越相似
        """
        try:
            if filter_dict:
                results = self.vectorstore.similarity_search_with_score(
                    query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vectorstore.similarity_search_with_score(query, k=k)

            return results

        except Exception as e:
            print(f"❌ 检索失败: {e}")
            return []

    def delete_by_source(self, source: str, verbose: bool = True) -> bool:
        """
        删除指定来源的所有文档

        :param source: 文件名，如 "劳动合同.pdf"
        :param verbose: 是否打印详细信息
        :return: 是否成功
        """
        try:
            # 先查询该来源的所有文档ID
            collection = self.vectorstore._collection
            results = collection.get(where={"source": source})

            if not results['ids']:
                if verbose:
                    print(f"⚠️  未找到来源为 '{source}' 的文档")
                return False

            # 删除
            collection.delete(ids=results['ids'])

            if verbose:
                print(f"✅ 已删除 {len(results['ids'])} 个来自 '{source}' 的文档块")

            return True

        except Exception as e:
            if verbose:
                print(f"❌ 删除失败: {e}")
            return False

    def delete_all(self, confirm: bool = False, verbose: bool = True) -> bool:
        """
        删除所有文档（危险操作）

        :param confirm: 必须设为True才执行
        :param verbose: 是否打印详细信息
        :return: 是否成功
        """
        if not confirm:
            if verbose:
                print("⚠️  删除所有文档需要设置 confirm=True")
            return False

        try:
            # 删除整个集合
            self.vectorstore._client.delete_collection(self.collection_name)

            # 重新初始化
            self.vectorstore = self._init_vectorstore()

            if verbose:
                print("✅ 已删除所有文档")

            return True

        except Exception as e:
            if verbose:
                print(f"❌ 删除失败: {e}")
            return False

    def list_sources(self) -> List[str]:
        """
        列出所有已存储文档的来源

        :return: 文件名列表
        """
        try:
            collection = self.vectorstore._collection
            results = collection.get()

            if not results['metadatas']:
                return []

            # 提取所有 source 字段
            sources = set()
            for metadata in results['metadatas']:
                if 'source' in metadata:
                    sources.add(metadata['source'])

            return sorted(list(sources))

        except Exception as e:
            print(f"❌ 获取来源列表失败: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """
        获取数据库统计信息

        :return: 统计信息字典
        """
        try:
            collection = self.vectorstore._collection
            results = collection.get()

            total_docs = len(results['ids'])
            sources = self.list_sources()

            # 统计每个来源的文档数
            source_counts = {}
            if results['metadatas']:
                for metadata in results['metadatas']:
                    source = metadata.get('source', 'Unknown')
                    source_counts[source] = source_counts.get(source, 0) + 1

            return {
                "total_documents": total_docs,
                "total_sources": len(sources),
                "sources": sources,
                "source_counts": source_counts,
                "collection_name": self.collection_name,
                "persist_directory": str(self.persist_directory)
            }

        except Exception as e:
            print(f"❌ 获取统计信息失败: {e}")
            return {}

    def print_stats(self):
        """打印统计信息（格式化）"""
        stats = self.get_stats()

        print("\n" + "=" * 70)
        print("向量数据库统计信息")
        print("=" * 70)
        print(f"总文档块数: {stats.get('total_documents', 0)}")
        print(f"总文件数: {stats.get('total_sources', 0)}")
        print(f"集合名称: {stats.get('collection_name', 'N/A')}")
        print(f"存储路径: {stats.get('persist_directory', 'N/A')}")

        if stats.get('source_counts'):
            print("\n各文件文档块数:")
            for source, count in sorted(stats['source_counts'].items()):
                print(f"  📄 {source}: {count} 个块")

        print("=" * 70 + "\n")


# ============ 工厂函数 ============
def create_vectorstore_manager(
        embedding_model: HuggingFaceEmbeddings,
        persist_directory: str = "./data/vectorstore",
        collection_name: str = "rag_documents"
) -> VectorStoreManager:
    """
    快速创建向量数据库管理器

    :param embedding_model: HuggingFaceEmbeddings 实例
    :param persist_directory: 数据库持久化目录
    :param collection_name: 集合名称
    :return: VectorStoreManager 实例
    """
    return VectorStoreManager(
        embedding_model=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )


# ============ 使用示例 ============
if __name__ == "__main__":
    from langchain_huggingface import HuggingFaceEmbeddings

    # 1. 初始化 embedding 模型
    print("初始化 Embedding 模型...")
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        cache_folder=get_models_cache_dir(),
        model_kwargs={
            "device": "cpu",
            "local_files_only": True  # ← 加上这个，不联网
        }
    )

    # 2. 创建向量数据库管理器
    print("初始化向量数据库...")
    manager = create_vectorstore_manager(
        embedding_model=embedding,
        persist_directory="./data/vectorstore_test"
    )

    # 3. 创建测试文档
    test_docs = [
        Document(
            page_content="深度学习是机器学习的一个分支。",
            metadata={"source": "test1.pdf", "page": 1}
        ),
        Document(
            page_content="神经网络是深度学习的基础。",
            metadata={"source": "test1.pdf", "page": 1}
        ),
        Document(
            page_content="大语言模型可以理解和生成文本。",
            metadata={"source": "test2.pdf", "page": 1}
        )
    ]

    # 4. 添加文档
    print("\n测试：添加文档")
    manager.add_documents(test_docs)

    # 5. 查看统计
    print("\n测试：查看统计信息")
    manager.print_stats()

    # 6. 检索测试
    print("\n测试：相似度检索")
    query = "什么是深度学习？"
    results = manager.search_with_score(query, k=2)

    print(f"查询: {query}")
    print(f"找到 {len(results)} 个结果:\n")
    for doc, score in results:
        print(f"相似度: {score:.4f}")
        print(f"内容: {doc.page_content}")
        print(f"来源: {doc.metadata.get('source', 'N/A')}")
        print()

    # 7. 删除测试
    print("\n测试：删除文档")
    manager.delete_by_source("test1.pdf")
    manager.print_stats()

    # 8. 清空数据库
    print("\n测试：清空数据库")
    manager.delete_all(confirm=True)
    manager.print_stats()

    # 获取集合
    collection = manager.vectorstore._collection
    # 检查文档数量
    count = collection.count()
    print(f"集合中的文档数量: {count}")
    assert count == 0, "数据库未完全清空!"