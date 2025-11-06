"""
向量数据库完整流程测试
PDF加载 → 分块 → 向量化存储 → 检索
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings

from models.model_paths import get_models_cache_dir
from src.loaders.pdf_loader import create_pdf_loader
from src.vectorstore.chroma_store import create_vectorstore_manager

def test_full_pipeline(
    embedding: HuggingFaceEmbeddings,
    manager: 'VectorStoreManager'
):
    """测试完整流程：PDF → 分块 → 存储 → 检索"""

    print("\n" + "🚀 " + "="*66 + " 🚀")
    print("     RAG 完整流程测试")
    print("🚀 " + "="*66 + " 🚀\n")

    # ========== 步骤1: 加载并分块PDF ==========
    print("="*70)
    print("步骤1: 加载并分块 PDF 文件")
    print("="*70 + "\n")

    loader = create_pdf_loader(
        embedding_model=embedding,
        chunk_size=300,
        chunk_overlap=0.1,
        base_threshold=0.8,
        dynamic_threshold=True,
        window_size=2,
        verbose=True
    )

    # 加载多个PDF
    pdf_files = [
        "../data/documents/劳动合同.pdf",
        "../data/documents/汪春养简历.pdf"
    ]

    # 过滤存在的文件
    existing_files = [f for f in pdf_files if os.path.exists(f)]

    if not existing_files:
        print("❌ 没有找到PDF文件")
        return

    documents = loader.load_batch(existing_files)

    print(f"\n✅ 成功加载并分块 {len(documents)} 个文档块\n")

    # ========== 步骤2: 存储文档 ==========
    print("="*70)
    print("步骤2: 将文档存储到向量数据库")
    print("="*70 + "\n")

    # 先查看数据库状态
    print("当前数据库状态:")
    manager.print_stats()

    manager.add_documents(documents, verbose=True)

    # 查看存储后的状态
    print("\n存储后的数据库状态:")
    manager.print_stats()

    # ========== 步骤3: 检索测试 ==========
    print("="*70)
    print("步骤3: 相似度检索测试")
    print("="*70 + "\n")

    test_queries = [
        "劳动合同的期限是多久？",
        "工作地点在哪里？",
        "汪春养的工作经历有哪些？"
    ]

    for query in test_queries:
        print(f"\n📝 查询: {query}")
        print("-" * 70)

        results = manager.search_with_score(query, k=3)

        if results:
            for i, (doc, score) in enumerate(results, 1):
                print(f"\n结果 {i} (相似度: {score:.4f})")
                print(f"  来源: {doc.metadata.get('source', 'N/A')}")
                print(f"  页码: {doc.metadata.get('page', 'N/A')}")
                print(f"  内容: {doc.page_content[:100]}...")
        else:
            print("  未找到相关结果")

    # ========== 步骤4: 查看所有来源 ==========
    print("\n" + "="*70)
    print("步骤4: 查看所有已存储的文件")
    print("="*70 + "\n")

    sources = manager.list_sources()
    print(f"共有 {len(sources)} 个文件:")
    for source in sources:
        print(f"  📄 {source}")

    print("\n" + "✅ " + "="*66 + " ✅")
    print("     完整流程测试完成！")
    print("✅ " + "="*66 + " ✅\n")


def test_incremental_add(
    embedding: HuggingFaceEmbeddings,
    manager: 'VectorStoreManager'
):
    """测试增量添加文档"""
    print("\n" + "="*70)
    print("测试：增量添加新文档")
    print("="*70 + "\n")

    print("当前状态:")
    manager.print_stats()

    # 2. 加载新文档
    loader = create_pdf_loader(embedding_model=embedding, verbose=False)

    new_pdf = "../data/documents/MTBG产研团队对接人.xlsx"  # 注意：这是Excel，需要其他loader
    # 这里只是演示，实际需要用Excel loader

    # 如果有新的PDF
    new_pdf = "../data/documents/新文档.pdf"  # 替换为实际文件
    if os.path.exists(new_pdf):
        new_docs = loader.load(new_pdf)
        manager.add_documents(new_docs)

        print("\n添加新文档后:")
        manager.print_stats()


def test_delete_document(
    manager: 'VectorStoreManager'
):
    """测试删除文档"""
    print("\n" + "="*70)
    print("测试：删除指定文档")
    print("="*70 + "\n")

    print("删除前:")
    manager.print_stats()

    # 2. 删除某个文件
    source_to_delete = "劳动合同.pdf"
    manager.delete_by_source(source_to_delete)

    print("\n删除后:")
    manager.print_stats()


def test_search_with_filter(
    manager: 'VectorStoreManager'
):
    """测试带过滤条件的检索"""
    print("\n" + "="*70)
    print("测试：元数据过滤检索")
    print("="*70 + "\n")

    # 2. 只在"劳动合同.pdf"中检索
    query = "工作地点在哪里？"
    print(f"查询: {query}")
    print("过滤条件: source = '劳动合同.pdf'\n")

    results = manager.search_with_score(
        query,
        k=3,
        filter_dict={"source": "劳动合同.pdf"}
    )

    for i, (doc, score) in enumerate(results, 1):
        print(f"结果 {i} (相似度: {score:.4f})")
        print(f"  来源: {doc.metadata.get('source')}")
        print(f"  内容: {doc.page_content[:100]}...\n")


def main():
    """主函数"""
    try:
        # ========== 只加载一次资源 ==========
        print("\n" + "="*70)
        print("初始化资源（只加载一次）")
        print("="*70 + "\n")

        # 1. 加载 Embedding 模型
        print("⏳ 加载 Embedding 模型...")
        embedding = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            cache_folder=get_models_cache_dir(),
            model_kwargs={
                "device": "cpu",
                "local_files_only": True
            }
        )
        print("✅ Embedding 模型加载完成")

        # 2. 创建向量数据库管理器
        print("⏳ 初始化向量数据库...")
        manager = create_vectorstore_manager(
            embedding_model=embedding,
            persist_directory="../data/vectorstore",
            collection_name="rag_documents"
        )
        print("✅ 向量数据库初始化完成\n")

        # ========== 运行各个测试，传递资源 ==========
        # 完整流程测试
        test_full_pipeline(embedding, manager)

        # 其他测试（可选）
        # test_incremental_add(embedding, manager)
        # test_delete_document(manager)
        # test_search_with_filter(manager)

    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()