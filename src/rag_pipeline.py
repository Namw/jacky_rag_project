"""
RAG Pipeline 主流程
检索 → Prompt组装 → LLM生成 → 答案返回
"""

from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.vectorstore.chroma_store import VectorStoreManager
from models.model_factory import ModelFactory
from models.model_paths import get_models_cache_dir
from config.model_config import ModelProvider


class RAGPipeline:
    """
    RAG 主流程管理器

    功能：
    1. 向量检索相关文档
    2. 组装 Prompt
    3. 调用 LLM 生成答案
    4. 返回答案 + 来源
    """

    def __init__(
            self,
            vectorstore_manager: VectorStoreManager,
            model_provider: ModelProvider = ModelProvider.DEEPSEEK,
            top_k: int = 5,
            temperature: float = 0.7,
            verbose: bool = True
    ):
        """
        初始化 RAG Pipeline

        :param vectorstore_manager: 向量数据库管理器
        :param model_provider: LLM 提供商
        :param top_k: 检索文档数量
        :param temperature: LLM 温度
        :param verbose: 是否打印详细信息
        """
        self.vectorstore = vectorstore_manager
        self.model_provider = model_provider
        self.top_k = top_k
        self.temperature = temperature
        self.verbose = verbose

        # 初始化 LLM
        self.llm = self._init_llm()

    def _init_llm(self) -> ChatOpenAI:
        """初始化 LLM 模型"""
        if self.verbose:
            print(f"\n⏳ 初始化 LLM: {self.model_provider.value}")

        llm = ModelFactory.create_model(
            provider=self.model_provider,
            temperature=self.temperature
        )

        if self.verbose:
            print(f"✅ LLM 初始化完成\n")

        return llm

    def retrieve_documents(
            self,
            query: str,
            k: Optional[int] = None,
            filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """
        检索相关文档

        :param query: 查询文本
        :param k: 检索数量（如果不指定则使用初始化时的 top_k）
        :param filter_dict: 元数据过滤条件
        :return: (Document, score) 列表
        """
        k = k or self.top_k

        if self.verbose:
            print(f"🔍 检索相关文档 (top_k={k})...")

        results = self.vectorstore.search_with_score(
            query=query,
            k=k,
            filter_dict=filter_dict
        )

        if self.verbose:
            print(f"   找到 {len(results)} 个相关文档\n")

        return results

    def build_prompt(
            self,
            query: str,
            documents: List[tuple[Document, float]],
            system_prompt: Optional[str] = None
    ) -> tuple[str, str]:
        """
        构建 Prompt

        :param query: 用户问题
        :param documents: 检索到的文档列表
        :param system_prompt: 自定义系统提示词
        :return: (system_message, user_message)
        """
        # 默认系统提示词
        if system_prompt is None:
            system_prompt = """你是一个专业的知识助手。

你的任务是根据提供的参考资料回答用户的问题。

要求：
1. 仔细阅读参考资料，基于资料内容回答
2. 如果资料中没有相关信息，诚实地告诉用户"根据提供的资料，我无法回答这个问题"
3. 回答要准确、简洁、有条理
4. 可以适当引用资料中的关键信息
5. 不要编造资料中没有的内容"""

        # 组装参考资料
        context_parts = []
        for i, (doc, score) in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            content = doc.page_content

            context_parts.append(
                f"【参考资料 {i}】\n"
                f"来源: {source} (第{page}页)\n"
                f"内容: {content}\n"
            )

        context = "\n".join(context_parts)

        # 用户消息
        user_message = f"""参考资料：
{context}

---

用户问题：{query}

请基于上述参考资料回答问题。"""

        return system_prompt, user_message

    def generate_answer(
            self,
            query: str,
            documents: List[tuple[Document, float]],
            system_prompt: Optional[str] = None
    ) -> str:
        """
        生成答案

        :param query: 用户问题
        :param documents: 检索到的文档
        :param system_prompt: 自定义系统提示词
        :return: LLM 生成的答案
        """
        # 构建 Prompt
        sys_msg, user_msg = self.build_prompt(query, documents, system_prompt)

        if self.verbose:
            print(f"🤖 调用 LLM 生成答案...")

        # 调用 LLM
        messages = [
            SystemMessage(content=sys_msg),
            HumanMessage(content=user_msg)
        ]

        try:
            response = self.llm.invoke(messages)
            answer = response.content

            if self.verbose:
                print(f"✅ 答案生成完成\n")

            return answer

        except Exception as e:
            error_msg = f"LLM 调用失败: {str(e)}"
            if self.verbose:
                print(f"❌ {error_msg}\n")
            return f"抱歉，生成答案时出现错误：{error_msg}"

    def query(
            self,
            question: str,
            top_k: Optional[int] = None,
            filter_dict: Optional[Dict[str, Any]] = None,
            return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        完整的 RAG 查询流程

        :param question: 用户问题
        :param top_k: 检索文档数量
        :param filter_dict: 元数据过滤
        :param return_sources: 是否返回来源信息
        :return: 包含答案和来源的字典
        """
        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"RAG 查询")
            print(f"{'=' * 70}")
            print(f"问题: {question}\n")

        # 1. 检索文档
        documents = self.retrieve_documents(question, k=top_k, filter_dict=filter_dict)

        if not documents:
            return {
                "answer": "抱歉，没有找到相关的参考资料。",
                "sources": [],
                "question": question
            }

        # 2. 生成答案
        answer = self.generate_answer(question, documents)

        # 3. 整理返回结果
        result = {
            "answer": answer,
            "question": question
        }

        if return_sources:
            sources = []
            for doc, score in documents:
                sources.append({
                    "source": doc.metadata.get('source', 'Unknown'),
                    "page": doc.metadata.get('page', 'N/A'),
                    "content": doc.page_content[:200] + "...",
                    "score": float(score)
                })
            result["sources"] = sources

        if self.verbose:
            print(f"{'=' * 70}\n")

        return result

    def switch_model(self, provider: ModelProvider):
        """
        切换 LLM 提供商

        :param provider: 新的模型提供商
        """
        self.model_provider = provider
        self.llm = self._init_llm()

    def print_answer(self, result: Dict[str, Any]):
        """
        格式化打印答案

        :param result: query() 返回的结果
        """
        print(f"\n{'=' * 70}")
        print(f"问题: {result['question']}")
        print(f"{'=' * 70}\n")

        print(f"📝 答案：")
        print(result['answer'])

        if 'sources' in result and result['sources']:
            print(f"\n{'─' * 70}")
            print(f"📚 参考来源:")
            for i, source in enumerate(result['sources'], 1):
                print(f"\n  [{i}] {source['source']} (第{source['page']}页)")
                print(f"      相似度: {source['score']:.4f}")
                print(f"      内容: {source['content']}")

        print(f"\n{'=' * 70}\n")


# ============ 工厂函数 ============
def create_rag_pipeline(
        vectorstore_manager: VectorStoreManager,
        model_provider: ModelProvider = ModelProvider.DEEPSEEK,
        top_k: int = 5,
        temperature: float = 0.7,
        verbose: bool = True
) -> RAGPipeline:
    """
    快速创建 RAG Pipeline

    :param vectorstore_manager: 向量数据库管理器
    :param model_provider: LLM 提供商
    :param top_k: 检索文档数量
    :param temperature: LLM 温度
    :param verbose: 是否打印详细信息
    :return: RAGPipeline 实例
    """
    return RAGPipeline(
        vectorstore_manager=vectorstore_manager,
        model_provider=model_provider,
        top_k=top_k,
        temperature=temperature,
        verbose=verbose
    )


# ============ 使用示例 ============
if __name__ == "__main__":
    from langchain_huggingface import HuggingFaceEmbeddings
    from src.vectorstore.chroma_store import create_vectorstore_manager
    # 加载环境变量
    from dotenv import load_dotenv
    load_dotenv()

    # 1. 初始化 embedding
    print("初始化 Embedding...")
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        cache_folder=get_models_cache_dir(),
        model_kwargs={
            "device": "cpu",
            "local_files_only": True
        }
    )

    # 2. 初始化向量数据库
    print("初始化向量数据库...")
    vectorstore = create_vectorstore_manager(
        embedding_model=embedding,
        persist_directory="../data/vectorstore"
    )

    # 3. 创建 RAG Pipeline
    print("创建 RAG Pipeline...")
    rag = create_rag_pipeline(
        vectorstore_manager=vectorstore,
        model_provider=ModelProvider.DEEPSEEK,
        top_k=3,
        verbose=True
    )

    # 4. 测试查询
    test_questions = [
        "劳动合同的期限是多久？",
        "工作地点在哪里？",
        "汪春养的工作经历有哪些？"
    ]

    for question in test_questions:
        result = rag.query(question)
        rag.print_answer(result)
        input("\n按 Enter 继续下一个问题...")

    # 5. 测试切换模型
    print("\n切换到 Qwen 模型...")
    rag.switch_model(ModelProvider.QWEN)

    result = rag.query("总结一下劳动合同的主要内容")
    rag.print_answer(result)