"""
完整 RAG 流程测试
PDF加载 → 向量存储 → 检索 → LLM生成答案
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

from models.model_paths import get_models_cache_dir
from src.loaders.pdf_loader import create_pdf_loader
from src.vectorstore.chroma_store import create_vectorstore_manager
from src.rag_pipeline import create_rag_pipeline
from config.model_config import ModelProvider

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def test_rag_query():
    """测试 RAG 完整查询流程"""

    # 加载环境变量
    load_dotenv()

    print("\n" + "🚀 " + "=" * 66 + " 🚀")
    print("     完整 RAG 流程测试")
    print("🚀 " + "=" * 66 + " 🚀\n")

    # ========== 步骤1: 初始化资源 ==========
    print("=" * 70)
    print("步骤1: 初始化资源")
    print("=" * 70 + "\n")

    # 1.1 加载 Embedding 模型
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

    # 1.2 初始化向量数据库
    print("⏳ 初始化向量数据库...")
    vectorstore = create_vectorstore_manager(
        embedding_model=embedding,
        persist_directory="../data/vectorstore"
    )
    print("✅ 向量数据库初始化完成")

    # 查看数据库状态
    vectorstore.print_stats()

    # ========== 步骤2: 如果数据库为空，先加载文档 ==========
    stats = vectorstore.get_stats()
    if stats.get('total_documents', 0) == 0:
        print("\n" + "=" * 70)
        print("步骤2: 数据库为空，加载文档...")
        print("=" * 70 + "\n")

        loader = create_pdf_loader(
            embedding_model=embedding,
            chunk_size=300,
            chunk_overlap=0.1,
            verbose=False
        )

        pdf_files = [
            "../data/documents/劳动合同.pdf",
            "../data/documents/汪春养简历.pdf"
        ]

        existing_files = [f for f in pdf_files if os.path.exists(f)]

        if existing_files:
            documents = loader.load_batch(existing_files)
            vectorstore.add_documents(documents, verbose=True)
            print("\n数据加载完成！")
            vectorstore.print_stats()
        else:
            print("⚠️  未找到PDF文件，无法进行测试")
            return

    # ========== 步骤3: 创建 RAG Pipeline ==========
    print("\n" + "=" * 70)
    print("步骤3: 创建 RAG Pipeline")
    print("=" * 70 + "\n")

    rag = create_rag_pipeline(
        vectorstore_manager=vectorstore,
        model_provider=ModelProvider.DEEPSEEK,  # 使用 DeepSeek
        top_k=3,
        temperature=0.7,
        verbose=True
    )

    # ========== 步骤4: 测试查询 ==========
    print("\n" + "=" * 70)
    print("步骤4: 测试 RAG 查询")
    print("=" * 70 + "\n")

    test_questions = [
        "工作地点在哪里？",
        "劳动合同的期限是多久？",
        "工资待遇是怎样的？"
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'─' * 70}")
        print(f"查询 {i}/{len(test_questions)}")
        print(f"{'─' * 70}\n")

        result = rag.query(question)
        rag.print_answer(result)

        if i < len(test_questions):
            input("按 Enter 继续下一个查询...")

    # ========== 步骤5: 测试带过滤的查询 ==========
    print("\n" + "=" * 70)
    print("步骤5: 测试元数据过滤查询")
    print("=" * 70 + "\n")

    # 只在"劳动合同.pdf"中检索
    question = "这份合同的主要内容是什么？"
    result = rag.query(
        question,
        filter_dict={"source": "劳动合同.pdf"}
    )
    rag.print_answer(result)

    print("\n" + "✅ " + "=" * 66 + " ✅")
    print("     所有测试完成！")
    print("✅ " + "=" * 66 + " ✅\n")


def test_model_switching():
    """测试模型切换功能"""

    load_dotenv()

    print("\n" + "=" * 70)
    print("测试：模型切换")
    print("=" * 70 + "\n")

    # 初始化资源
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        cache_folder=get_models_cache_dir(),
        model_kwargs={"device": "cpu", "local_files_only": True}
    )

    vectorstore = create_vectorstore_manager(
        embedding_model=embedding,
        persist_directory="../data/vectorstore"
    )

    # 创建 RAG Pipeline（默认 DeepSeek）
    rag = create_rag_pipeline(
        vectorstore_manager=vectorstore,
        model_provider=ModelProvider.DEEPSEEK,
        verbose=True
    )

    question = "总结一下劳动合同的主要内容"

    # 测试 DeepSeek
    print("\n【使用 DeepSeek】")
    result = rag.query(question, top_k=3)
    rag.print_answer(result)

    # 切换到 Qwen
    print("\n" + "=" * 70)
    print("切换模型到 Qwen")
    print("=" * 70 + "\n")

    rag.switch_model(ModelProvider.QWEN)

    print("\n【使用 Qwen】")
    result = rag.query(question, top_k=3)
    rag.print_answer(result)


def test_custom_prompt():
    """测试自定义 Prompt"""

    load_dotenv()

    print("\n" + "=" * 70)
    print("测试：自定义 System Prompt")
    print("=" * 70 + "\n")

    # 初始化
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        cache_folder=get_models_cache_dir(),
        model_kwargs={"device": "cpu", "local_files_only": True}
    )

    vectorstore = create_vectorstore_manager(
        embedding_model=embedding,
        persist_directory="../data/vectorstore"
    )

    rag = create_rag_pipeline(
        vectorstore_manager=vectorstore,
        model_provider=ModelProvider.DEEPSEEK,
        verbose=True
    )

    # 自定义 System Prompt
    custom_prompt = """你是一个专业的法律顾问助手。

在回答问题时：
1. 使用专业但易懂的语言
2. 重点关注法律条款和权益
3. 如果涉及重要权益，给出提醒
4. 基于资料客观回答，不做过度推断"""

    question = "劳动合同中对劳动者的权益保护有哪些？"

    # 检索文档
    documents = rag.retrieve_documents(question, k=3)

    # 使用自定义 prompt 生成答案
    answer = rag.generate_answer(question, documents, system_prompt=custom_prompt)

    print(f"\n{'=' * 70}")
    print(f"问题: {question}")
    print(f"{'=' * 70}\n")
    print(f"📝 答案：")
    print(answer)
    print(f"\n{'=' * 70}\n")


def main():
    """主函数"""
    try:
        # 主测试：完整 RAG 查询流程
        test_rag_query()

        # 其他测试（可选）
        # test_model_switching()
        # test_custom_prompt()

    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()