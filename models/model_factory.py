"""
模型工厂类

提供统一的接口来创建不同提供商的聊天模型
遵循工厂模式和策略模式
"""

import os
from typing import Optional
from langchain_openai import ChatOpenAI

from config.model_config import ModelProvider, get_model_config


class ModelFactory:
    """
    模型工厂类

    用于创建不同提供商的聊天模型实例
    支持动态选择和创建模型
    """

    @staticmethod
    def create_model(
        provider: ModelProvider,
        temperature: Optional[float] = None,
        **kwargs
    ) -> ChatOpenAI:
        """
        创建指定提供商的聊天模型

        Args:
            provider: 模型提供商
            temperature: 温度参数（可选，如果不指定使用配置默认值）
            **kwargs: 其他参数，传递给模型

        Returns:
            ChatOpenAI 实例

        Raises:
            ValueError: 如果提供商不支持或 API Key 未配置
            Exception: 如果创建模型失败
        """
        # 获取模型配置
        config = get_model_config(provider)

        # 获取 API Key
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(
                f"❌ 错误：未找到 {config.api_key_env} 环境变量\n"
                f"   请在 .env 文件中设置：{config.api_key_env}=your_api_key\n"
                f"   或通过环境变量设置：export {config.api_key_env}=your_api_key"
            )

        # 使用传入的 temperature 或使用配置中的默认值
        temp = temperature if temperature is not None else config.temperature

        # 合并额外参数
        model_params = {
            "model": config.model_name,
            "api_key": api_key,
            "base_url": config.base_url,
            "temperature": temp,
            **config.extra_params,
            **kwargs,  # 用户传入的参数优先级最高
        }

        try:
            print(f"🚀 初始化 {provider.value} 模型...")
            print(f"   模型: {config.model_name}")
            print(f"   温度: {temp}")

            model = ChatOpenAI(**model_params)
            print(f"✅ {provider.value} 模型初始化成功")
            return model

        except Exception as e:
            raise Exception(
                f"❌ 创建 {provider.value} 模型失败：{str(e)}"
            )

    @staticmethod
    def get_available_providers() -> list:
        """获取所有可用的模型提供商"""
        return list(ModelProvider)

    @staticmethod
    def get_provider_info() -> dict:
        """
        获取所有提供商的详细信息

        Returns:
            提供商信息字典
        """
        info = {}
        for provider in ModelFactory.get_available_providers():
            config = get_model_config(provider)
            info[provider.value] = {
                "model": config.model_name,
                "api_key_env": config.api_key_env,
                "base_url": config.base_url,
                "temperature": config.temperature,
            }
        return info


class ModelManager:
    """
    模型管理器

    用于管理应用中的模型实例
    支持模型的创建、切换和管理
    """

    def __init__(self):
        """初始化模型管理器"""
        self._models = {}  # 缓存已创建的模型
        self._current_provider = None
        self._current_model = None

    def create_or_get_model(
        self,
        provider: ModelProvider,
        **kwargs
    ) -> ChatOpenAI:
        """
        创建或获取指定提供商的模型

        如果模型已经创建过，则返回缓存的实例
        否则创建新实例并缓存

        Args:
            provider: 模型提供商
            **kwargs: 其他参数

        Returns:
            ChatOpenAI 实例
        """
        if provider not in self._models:
            self._models[provider] = ModelFactory.create_model(provider, **kwargs)
        return self._models[provider]

    def switch_model(self, provider: ModelProvider) -> ChatOpenAI:
        """
        切换到指定的模型提供商

        Args:
            provider: 目标模型提供商

        Returns:
            新的模型实例
        """
        model = self.create_or_get_model(provider)
        self._current_provider = provider
        self._current_model = model
        print(f"\n➜ 已切换到 {provider.value} 模型\n")
        return model

    def get_current_model(self) -> Optional[ChatOpenAI]:
        """获取当前使用的模型"""
        return self._current_model

    def get_current_provider(self) -> Optional[ModelProvider]:
        """获取当前使用的模型提供商"""
        return self._current_provider

    def clear_cache(self):
        """清除缓存的模型实例"""
        self._models.clear()
        self._current_model = None
        self._current_provider = None


# 全局模型管理器实例
_global_manager = None


def get_model_manager() -> ModelManager:
    """获取全局模型管理器实例"""
    global _global_manager
    if _global_manager is None:
        _global_manager = ModelManager()
    return _global_manager
