"""
模型配置定义

支持的模型：
- qwen: 阿里巴巴通义千问
- deepseek: Deepseek 聊天模型
- openai: OpenAI GPT 模型
- claude: Anthropic Claude 模型
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum


class ModelProvider(str, Enum):
    """模型提供商枚举"""
    QWEN = "qwen"
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    CLAUDE = "claude"


@dataclass
class ModelConfig:
    """模型配置基类"""
    provider: ModelProvider
    model_name: str
    api_key_env: str  # 环境变量名称
    base_url: str  # API 端点
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    extra_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


# 模型配置注册表
MODEL_CONFIGS = {
    ModelProvider.QWEN: ModelConfig(
        provider=ModelProvider.QWEN,
        model_name="qwen3-coder-plus",
        api_key_env="DASHSCOPE_API_KEY",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.7,
    ),
    ModelProvider.DEEPSEEK: ModelConfig(
        provider=ModelProvider.DEEPSEEK,
        model_name="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
        temperature=0.7,
    ),
    ModelProvider.OPENAI: ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        temperature=0.7,
    ),
    ModelProvider.CLAUDE: ModelConfig(
        provider=ModelProvider.CLAUDE,
        model_name="claude-3-sonnet-20240229",
        api_key_env="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com",
        temperature=0.7,
    ),
}


def get_model_config(provider: ModelProvider) -> ModelConfig:
    """
    获取指定提供商的模型配置

    Args:
        provider: 模型提供商

    Returns:
        模型配置对象

    Raises:
        ValueError: 如果提供商不支持
    """
    if provider not in MODEL_CONFIGS:
        supported = ", ".join([p.value for p in ModelProvider])
        raise ValueError(
            f"不支持的模型提供商: {provider.value}\n"
            f"支持的提供商: {supported}"
        )
    return MODEL_CONFIGS[provider]


def get_all_providers() -> list:
    """获取所有支持的提供商"""
    return list(ModelProvider)


def get_provider_by_name(name: str) -> Optional[ModelProvider]:
    """
    根据提供商名称获取枚举

    Args:
        name: 提供商名称（小写）

    Returns:
        ModelProvider 枚举，如果不存在返回 None
    """
    try:
        return ModelProvider(name.lower())
    except ValueError:
        return None
