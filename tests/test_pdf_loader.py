"""
PDF 分块测试脚本
测试 PDF 加载和语义分块功能
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings

from models.model_paths import get_models_cache_dir
from src.loaders.pdf_loader import create_pdf_loader


def test_single_pdf():
    """测试单个 PDF 文件加载"""
    print("\n" + "=" * 70)
    print("测试1: 加载单个 PDF 文件")
    print("=" * 70 + "\n")

    # 1. 初始化 embedding 模型
    print("⏳ 正在初始化 Embedding 模型（首次加载可能较慢）...")
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        cache_folder=get_models_cache_dir(),
        model_kwargs={
            "device": "cpu",
            "local_files_only": True  # ← 加上这个，不联网
        }
    )
    print("✅ Embedding 模型加载完成\n")

    # 2. 创建 PDF Loader
    loader = create_pdf_loader(
        embedding_model=embedding,
        chunk_size=300,
        chunk_overlap=0.1,
        base_threshold=0.8,
        dynamic_threshold=True,
        window_size=2,
        verbose=True
    )

    # 3. 加载 PDF
    pdf_path = "../data/documents/劳动合同.pdf"

    if not os.path.exists(pdf_path):
        print(f"❌ 文件不存在: {pdf_path}")
        return None

    documents = loader.load(pdf_path)

    # 4. 查看结果
    print(f"\n{'=' * 70}")
    print(f"分块结果统计")
    print(f"{'=' * 70}")
    print(f"总分块数: {len(documents)}")

    if documents:
        # 分析分块长度
        chunk_lengths = [len(doc.page_content) for doc in documents]
        print(f"平均分块长度: {sum(chunk_lengths) / len(chunk_lengths):.0f} 字符")
        print(f"最短分块: {min(chunk_lengths)} 字符")
        print(f"最长分块: {max(chunk_lengths)} 字符")

        # 显示前3个分块
        print(f"\n{'=' * 70}")
        print("前3个分块示例:")
        print(f"{'=' * 70}\n")

        for i, doc in enumerate(documents[:3]):
            print(f"--- 分块 {i + 1} ---")
            print(f"页码: {doc.metadata.get('page', 'N/A')}")
            print(f"长度: {len(doc.page_content)} 字符")
            print(f"内容预览: {doc.page_content[:150]}...")
            print(f"完整元数据: {doc.metadata}")
            print()

    return documents

def main():
    """主函数：运行所有测试"""

    print("\n" + "🚀 " + "=" * 66 + " 🚀")
    print("     PDF 语义分块测试")
    print("🚀 " + "=" * 66 + " 🚀")

    try:
        # 测试1: 单个文件
        docs1 = test_single_pdf()
        print("\n" + "✅ " + "=" * 66 + " ✅")
        print("     所有测试完成！")
        print("✅ " + "=" * 66 + " ✅\n")

    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    # print(get_model_path("bge-large-zh-v1.5"))