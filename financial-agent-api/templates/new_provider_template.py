"""新 Provider 注册模板

本文件是 Agent 平台新 LLM Provider 的注册模板，包含完整的注册流程说明和示例代码。
复制此文件到 ``app/rag/`` 或 ``app/core/`` 目录下并重命名后，按以下 5 步完成注册。

═══════════════════════════════════════════════════════════════
  Provider 注册 5 步流程
═══════════════════════════════════════════════════════════════

Step 1 — 复制模板
    将本文件复制到 ``app/rag/`` 或 ``app/core/`` 目录，
    并重命名为有意义的名称，例如 ``provider_volcengine.py``。

Step 2 — 实现 Provider 客户端类
    创建继承或符合项目 LLM 调用约定的客户端类。
    - 类名使用 PascalCase（如 ``VolcEngineClient``）
    - 实现 ``chat()`` 和 ``embed()`` 方法（或按需实现其中之一）
    - 构造函数接收 base_url、api_key、model_name 等参数
    - 使用 httpx.AsyncClient 进行 HTTP 调用

Step 3 — 在 app/core/enums.py 中注册枚举
    打开 ``app/core/enums.py``，在 ``ModelProvider`` 和 ``ModelName`` 中
    添加新 Provider 的枚举值::

        class ModelProvider(StrEnum):
            ...
            VOLCENGINE = "volcengine"


        class ModelName(StrEnum):
            ...
            VOLCENGINE_CHAT = "volcengine-doubao-pro-32k"

Step 4 — 在 app/core/config.py 中添加配置项
    打开 ``app/core/config.py``，添加新 Provider 的环境变量配置::

        VOLCENGINE_BASE_URL: str = ""
        VOLCENGINE_API_KEY: str = ""
        VOLCENGINE_MODEL_NAME: str = "doubao-pro-32k"

Step 5 — 在路由或引擎中集成
    在 ``app/rag/engine_router.py`` 或 ``app/agent/chain.py`` 中，
    根据配置动态选择 Provider::

        from app.rag.provider_volcengine import VolcEngineClient

        if provider == ModelProvider.VOLCENGINE:
            client = VolcEngineClient(
                base_url=settings.VOLCENGINE_BASE_URL,
                api_key=settings.VOLCENGINE_API_KEY,
                model_name=settings.VOLCENGINE_MODEL_NAME,
            )

═══════════════════════════════════════════════════════════════
"""

from typing import Any

import httpx


class MyCustomProviderClient:
    """自定义 LLM Provider 客户端示例

    实现 chat 和 embed 两个核心方法，遵循项目统一的 LLM 调用接口。
    实际使用时请将 ``MyCustomProvider`` 替换为真实 Provider 名称。

    Attributes:
        _base_url: Provider API 基础 URL
        _api_key: Provider API Key
        _model_name: 默认模型名称
        _client: 异步 HTTP 客户端
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout: float = 60.0,
    ) -> None:
        """初始化 Provider 客户端

        Args:
            base_url: Provider API 基础 URL
            api_key: Provider API Key
            model_name: 默认模型名称
            timeout: 请求超时时间（秒），默认 60.0
        """
        self._base_url: str = base_url.rstrip("/")
        self._api_key: str = api_key
        self._model_name: str = model_name
        self._client: httpx.AsyncClient = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """调用 Chat Completion API

        Args:
            messages: 对话消息列表，格式为 ``[{"role": "user", "content": "..."}]``
            model: 模型名称，为 None 时使用默认模型
            temperature: 生成温度，默认 0.7
            max_tokens: 最大生成 Token 数，默认 2048

        Returns:
            Provider 返回的原始响应字典

        Raises:
            httpx.HTTPStatusError: HTTP 状态码非 2xx 时抛出
            httpx.TimeoutException: 请求超时时抛出
        """
        payload: dict[str, Any] = {
            "model": model or self._model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = await self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()

    async def embed(
        self,
        input_texts: list[str],
        model: str | None = None,
    ) -> list[list[float]]:
        """调用 Embedding API

        Args:
            input_texts: 待嵌入的文本列表
            model: 嵌入模型名称，为 None 时使用默认模型

        Returns:
            嵌入向量列表，每个元素为 float 列表

        Raises:
            httpx.HTTPStatusError: HTTP 状态码非 2xx 时抛出
            httpx.TimeoutException: 请求超时时抛出
        """
        payload: dict[str, Any] = {
            "model": model or self._model_name,
            "input": input_texts,
        }

        response = await self._client.post("/embeddings", json=payload)
        response.raise_for_status()
        result = response.json()

        # 适配 OpenAI 兼容格式的返回
        return [item["embedding"] for item in result.get("data", [])]

    async def close(self) -> None:
        """关闭异步 HTTP 客户端，释放资源"""
        await self._client.aclose()

    async def __aenter__(self) -> "MyCustomProviderClient":
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """异步上下文管理器出口"""
        await self.close()
