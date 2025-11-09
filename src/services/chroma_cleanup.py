"""
ChromaDB 清理工具模块
提供统一的 collection 删除和物理文件清理功能
"""
from pathlib import Path
import sqlite3
import shutil
from typing import Optional
import chromadb


def get_folder_uuid_from_db(
        collection_name: str,
        persist_dir: str
) -> Optional[str]:
    """
    从 ChromaDB SQLite 数据库获取 collection 对应的文件夹 UUID

    :param collection_name: collection 名称
    :param persist_dir: ChromaDB 持久化目录路径
    :return: 文件夹 UUID，如果未找到返回 None
    """
    db_path = Path(persist_dir) / "chroma.sqlite3"

    if not db_path.exists():
        print(f"⚠️ 数据库文件不存在: {db_path}")
        return None

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # 1. 从 collections 表获取 collection_id
        cursor.execute(
            "SELECT id FROM collections WHERE name = ?",
            (collection_name,)
        )
        result = cursor.fetchone()

        if not result:
            conn.close()
            return None

        collection_db_id = result[0]

        # 2. 从 segments 表获取向量存储的 segment_id（文件夹名）
        cursor.execute(
            "SELECT id FROM segments WHERE collection = ? AND type = ?",
            (collection_db_id, 'urn:chroma:segment/vector/hnsw-local-persisted')
        )
        segment_result = cursor.fetchone()

        conn.close()

        if segment_result:
            return segment_result[0]
        return None

    except Exception as e:
        print(f"⚠️ 查询数据库失败: {str(e)}")
        return None


def delete_collection_completely(
        collection_name: str,
        persist_dir: str,
        verbose: bool = True
) -> dict:
    """
    完全删除 ChromaDB collection（包括数据库记录和物理文件）

    :param collection_name: collection 名称
    :param persist_dir: ChromaDB 持久化目录路径
    :param verbose: 是否打印详细日志
    :return: 删除结果字典
    """
    result = {
        "collection_deleted": False,
        "folder_deleted": False,
        "folder_uuid": None,
        "error": None
    }

    try:
        # 1. 获取文件夹 UUID（在删除 collection 之前获取）
        folder_uuid = get_folder_uuid_from_db(collection_name, persist_dir)
        result["folder_uuid"] = folder_uuid

        if verbose and folder_uuid:
            print(f"📍 找到文件夹 UUID: {folder_uuid}")

        # 2. 删除 ChromaDB collection
        try:
            client = chromadb.PersistentClient(path=persist_dir)
            client.delete_collection(name=collection_name)
            result["collection_deleted"] = True

            if verbose:
                print(f"✅ 已删除 collection: {collection_name}")
        except Exception as e:
            if verbose:
                print(f"⚠️ 删除 collection 失败: {str(e)}")
            result["error"] = f"Failed to delete collection: {str(e)}"

        # 3. 删除物理文件夹
        if folder_uuid:
            uuid_folder_path = Path(persist_dir) / folder_uuid

            if uuid_folder_path.exists():
                try:
                    shutil.rmtree(uuid_folder_path)
                    result["folder_deleted"] = True

                    if verbose:
                        print(f"✅ 已删除文件夹: {uuid_folder_path}")
                except Exception as e:
                    if verbose:
                        print(f"⚠️ 删除文件夹失败: {str(e)}")
                    if not result["error"]:
                        result["error"] = f"Failed to delete folder: {str(e)}"
            else:
                if verbose:
                    print(f"⚠️ 文件夹不存在: {uuid_folder_path}")
        else:
            if verbose:
                print(f"⚠️ 未找到 collection 对应的文件夹 UUID")

    except Exception as e:
        result["error"] = str(e)
        if verbose:
            print(f"❌ 删除过程出错: {str(e)}")

    return result