import chromadb
from chromadb.config import Settings
import pandas as pd


def read_chroma_collection(collection_identifier, persist_dir="./data/vectorstore/permanent", use_uuid=False):
    """
    读取指定ChromaDB集合的内容

    参数:
    collection_identifier: 集合名称或UUID
    persist_dir: ChromaDB持久化目录路径
    use_uuid: 是否使用UUID查找（默认False，使用name）

    返回:
    DataFrame: 包含ID、元数据、原始文本和向量的数据
    """
    # 初始化ChromaDB客户端
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(allow_reset=True)
    )

    # 获取指定集合
    try:
        if use_uuid:
            # 如果使用UUID，需要遍历所有集合找到匹配的
            all_collections = client.list_collections()
            collection = None
            for coll in all_collections:
                if str(coll.id) == collection_identifier:
                    collection = client.get_collection(name=coll.name)
                    print(f"找到集合: {coll.name}")
                    break
            if collection is None:
                raise ValueError(f"未找到UUID为 {collection_identifier} 的集合")
        else:
            # 直接使用name获取
            collection = client.get_collection(name=collection_identifier)
    except Exception as e:
        print(f"错误: 无法获取集合 {collection_identifier}")
        print(e)
        return None

    # 获取所有数据
    all_data = collection.get(include=['embeddings', 'documents', 'metadatas'])

    # 检查是否有数据
    if not all_data['ids']:
        print("集合为空")
        return pd.DataFrame()

    # 构建DataFrame - 将embedding转换为列表
    df = pd.DataFrame({
        'id': all_data['ids'],
        'text': all_data['documents'],
        'metadata': all_data['metadatas'],
        'embedding': [list(emb) for emb in all_data['embeddings']]  # 关键修改：转换为列表
    })

    return df


def preview_collection(df, num_samples=5):
    """
    预览指定集合的数据

    参数:
    df: 包含集合数据的DataFrame
    num_samples: 预览的样本数量
    """
    if df is None or df.empty:
        print("没有数据可预览")
        return

    print(f"\n{'=' * 80}")
    print(f"集合包含 {len(df)} 条记录")
    print(f"{'=' * 80}")

    # 预览前几条记录
    print(f"\n显示前 {min(num_samples, len(df))} 条记录:")
    for i in range(min(num_samples, len(df))):
        print(f"\n{'-' * 80}")
        print(f"📄 记录 {i + 1}/{len(df)}")
        print(f"{'-' * 80}")
        print(f"🆔 ID: {df.iloc[i]['id']}")
        print(f"\n📝 文本内容:")
        text = df.iloc[i]['text']
        if len(text) > 300:
            print(f"{text[:300]}...")
            print(f"   (总长度: {len(text)} 字符)")
        else:
            print(text)

        print(f"\n📋 元数据: {df.iloc[i]['metadata']}")

        if df.iloc[i]['embedding'] is not None:
            emb = df.iloc[i]['embedding']
            print(f"\n🔢 向量信息:")
            print(f"   - 维度: {len(emb)}")
            print(f"   - 前5个值: {emb[:5]}")
            print(f"   - 数据类型: {type(emb)}")


def analyze_collection(df):
    """
    分析集合的统计信息
    """
    if df is None or df.empty:
        print("没有数据可分析")
        return

    print(f"\n{'=' * 80}")
    print("📊 集合统计分析")
    print(f"{'=' * 80}")

    print(f"\n基本信息:")
    print(f"  - 总记录数: {len(df)}")
    print(f"  - 列数: {len(df.columns)}")
    print(f"  - 列名: {list(df.columns)}")

    if 'text' in df.columns:
        text_lengths = df['text'].apply(len)
        print(f"\n文本长度统计:")
        print(f"  - 平均长度: {text_lengths.mean():.2f} 字符")
        print(f"  - 最短文本: {text_lengths.min()} 字符")
        print(f"  - 最长文本: {text_lengths.max()} 字符")
        print(f"  - 中位数: {text_lengths.median():.2f} 字符")

    if 'embedding' in df.columns and len(df) > 0:
        emb_dim = len(df.iloc[0]['embedding'])
        print(f"\n向量信息:")
        print(f"  - 向量维度: {emb_dim}")
        print(f"  - 向量总数: {len(df)}")

    if 'metadata' in df.columns:
        print(f"\n元数据示例:")
        unique_keys = set()
        for meta in df['metadata']:
            if meta:
                unique_keys.update(meta.keys())
        print(f"  - 元数据字段: {list(unique_keys)}")


# 使用示例
if __name__ == "__main__":
    # 获取所有集合
    client = chromadb.PersistentClient(
        path="../data/vectorstore/permanent",
    )

    collections = client.list_collections()
    print("=" * 80)
    print("📚 可用的 ChromaDB 集合")
    print("=" * 80)
    for idx, collection in enumerate(collections, 1):
        print(f"\n{idx}. 集合名称: {collection.name}")
        print(f"   UUID: {collection.id}")

    # 读取指定集合的数据
    print("\n" + "=" * 80)
    choice = input("请选择输入方式 (1: 使用name, 2: 使用UUID, 3: 输入编号): ").strip()

    df = None
    if choice == "1":
        collection_identifier = input("请输入集合名称: ").strip()
        df = read_chroma_collection(collection_identifier, persist_dir="../data/vectorstore/permanent", use_uuid=False)
    elif choice == "2":
        collection_identifier = input("请输入集合UUID: ").strip()
        df = read_chroma_collection(collection_identifier, persist_dir="../data/vectorstore/permanent", use_uuid=True)
    elif choice == "3":
        idx_input = input("请输入集合编号: ").strip()
        try:
            idx = int(idx_input) - 1
            if 0 <= idx < len(collections):
                df = read_chroma_collection(collections[idx].name, persist_dir="../data/vectorstore/permanent",
                                            use_uuid=False)
            else:
                print("❌ 无效的编号")
        except ValueError:
            print("❌ 请输入有效的数字")
    else:
        print("❌ 无效的选择")

    # 预览和分析数据
    if df is not None and not df.empty:
        # 显示统计分析
        analyze_collection(df)

        # 预览详细数据
        print("\n" + "=" * 80)
        show_preview = input("\n是否查看详细记录？(y/n): ").strip().lower()
        if show_preview == 'y':
            num = input("要查看几条记录？(默认5): ").strip()
            num_samples = int(num) if num.isdigit() else 5
            preview_collection(df, num_samples)

        # 可选：保存为CSV（不包含embedding）
        print("\n" + "=" * 80)
        save = input("是否保存为CSV文件（不含向量）？(y/n): ").strip().lower()
        if save == 'y':
            output_file = input("请输入文件名（默认：collection_data.csv）: ").strip() or "collection_data.csv"
            df[['id', 'text', 'metadata']].to_csv(output_file, index=False, encoding='utf-8')
            print(f"✅ 数据已保存到 {output_file}")
    else:
        print("\n❌ 没有可用的数据")