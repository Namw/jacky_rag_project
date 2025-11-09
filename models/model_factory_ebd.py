# src/services/model_factory.py
"""
模型工厂 - 统一管理 Embedding 和 Reranker 模型的加载
"""

import os
import dashscope
from dashscope import TextReRank
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from models.model_paths import get_models_cache_dir
from langchain_community.embeddings import DashScopeEmbeddings

class ModelFactoryEbd:
    """模型工厂类，根据配置加载对应的模型"""

    _embedding_model = None
    _reranker_model = None

    @staticmethod
    def get_provider() -> str:
        """获取当前模型提供商"""
        return os.getenv("MODEL_PROVIDER", "aliyun").lower()

    @staticmethod
    def get_embedding_model():
        """获取 Embedding 模型（单例模式）"""
        if ModelFactoryEbd._embedding_model is None:
            provider = ModelFactoryEbd.get_provider()

            if provider == "local":
                ModelFactoryEbd._embedding_model = ModelFactoryEbd._load_local_embedding()
            elif provider == "aliyun":
                ModelFactoryEbd._embedding_model = ModelFactoryEbd._load_aliyun_embedding()
            else:
                raise ValueError(f"不支持的模型提供商: {provider}")

        return ModelFactoryEbd._embedding_model

    @staticmethod
    def get_reranker_model():
        """获取 Reranker 模型（单例模式）"""
        if ModelFactoryEbd._reranker_model is None:
            provider = ModelFactoryEbd.get_provider()
            if provider == "local":
                ModelFactoryEbd._reranker_model = ModelFactoryEbd._load_local_reranker()
            elif provider == "aliyun":
                ModelFactoryEbd._reranker_model = ModelFactoryEbd._load_aliyun_reranker()
            else:
                raise ValueError(f"不支持的模型提供商: {provider}")

        return ModelFactoryEbd._reranker_model

    # ==================== 本地模型加载 ====================

    @staticmethod
    def _load_local_embedding():
        """加载本地 Embedding 模型"""
        try:
            model_name = "BAAI/bge-large-zh-v1.5"
            print(f"📦 加载本地 Embedding 模型: {model_name}")
            embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                cache_folder=get_models_cache_dir(),
                model_kwargs={
                    "device": "cpu",
                    "local_files_only": True
                }
            )
            print("✅ 本地 Embedding 模型加载成功")
            return embedding_model

        except Exception as e:
            print(f"❌ 本地 Embedding 模型加载失败: {e}")
            raise

    @staticmethod
    def _load_local_reranker():
        """加载本地 Reranker 模型"""
        try:
            model_name = "/BAAI-bge-reranker-large"
            model_path = get_models_cache_dir() + f'/{model_name}'

            print(f"📦 加载本地 Reranker 模型: {model_name}")
            reranker_model = CrossEncoder(
                model_name_or_path=model_path,
                max_length=512,
                device='cpu'
            )
            print("✅ 本地 Reranker 模型加载成功")
            return reranker_model

        except Exception as e:
            print(f"⚠️ 本地 Reranker 模型加载失败: {e}")
            return None

    # ==================== 阿里云模型加载 ====================

    @staticmethod
    def _load_aliyun_embedding():
        """加载阿里云 Embedding 模型"""
        try:
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                raise ValueError("请在 .env 中配置 DASHSCOPE_API_KEY")
            model_name = "text-embedding-v4"
            print(f"📦 加载阿里云 Embedding 模型: {model_name}")
            embedding_model = DashScopeEmbeddings(
                model=model_name,
                dashscope_api_key=api_key
            )
            print("✅ 阿里云 Embedding 模型加载成功")
            return embedding_model

        except Exception as e:
            print(f"❌ 阿里云 Embedding 模型加载失败: {e}")
            raise

    @staticmethod
    def _load_aliyun_reranker():
        """加载阿里云 Reranker 模型（返回包装器）"""
        try:
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                raise ValueError("请在 .env 中配置 DASHSCOPE_API_KEY")
            dashscope.api_key = api_key
            model_name = "qwen3-rerank"
            print(f"📦 加载阿里云 Reranker 模型: {model_name}")
            print("✅ 阿里云 Reranker 模型配置成功")
            # 返回一个包装器，使其接口与本地模型一致
            return AliyunRerankerWrapper(model_name, api_key)
        except Exception as e:
            print(f"⚠️ 阿里云 Reranker 模型配置失败: {e}")
            return None


class AliyunRerankerWrapper:
    """阿里云 Reranker 的包装器，统一接口"""

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        dashscope.api_key = api_key

    def predict(self, pairs):
        """
        与 CrossEncoder 接口保持一致
        pairs: [[query, doc], [query, doc], ...]
        返回: [score1, score2, ...]
        """
        if not pairs:
            return []

        query = pairs[0][0]  # 所有 pair 的 query 相同
        documents = [pair[1] for pair in pairs]

        try:
            response = TextReRank.call(
                model=self.model_name,
                query=query,
                documents=documents
            )

            if response.status_code == 200:
                # 提取分数
                scores = [result['relevance_score'] for result in response.output.results]
                return scores
            else:
                print(f"⚠️ 阿里云 Rerank 调用失败: {response.message}")
                return [0.5] * len(pairs)  # 返回默认分数

        except Exception as e:
            print(f"⚠️ 阿里云 Rerank 异常: {e}")
            return [0.5] * len(pairs)