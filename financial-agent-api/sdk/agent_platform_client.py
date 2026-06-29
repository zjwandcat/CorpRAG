"""Agent Platform Client — 同步与异步 SDK 客户端

提供 AgentPlatformClient（同步）和 AsyncAgentPlatformClient（异步）两个客户端类，
封装了企业 GenAI Agent 平台的所有 API 接口，包括：

- 对话（chat / chat_stream）
- 文档上传（upload_document）
- 代码审查（review_code）
- 健康检查（get_health）

SSE 流式解析遵循 ``event: <type>`` / ``data: <json>`` 协议，
每次 yield ``{"event": etype, "data": parsed_data}`` 字典。

使用示例::

    # 同步
    with AgentPlatformClient("http://localhost:8000", "sk-xxx") as c:
        for chunk in c.chat_stream("你好"):
            print(chunk)

    # 异步
    async with AsyncAgentPlatformClient("http://localhost:8000", "sk-xxx") as c:
        result = await c.chat("你好")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, AsyncGenerator, Generator, Optional

import httpx


class AgentPlatformClient:
    """同步客户端，封装 Agent Platform 的所有 REST API。

    Args:
        base_url: 平台服务地址，例如 ``http://localhost:8000``
        api_key: API 密钥，用于 ``Authorization: Bearer {api_key}`` 认证
        timeout: 请求超时时间（秒），默认 120
    """

    def __init__(self, base_url: str, api_key: str, timeout: int = 120) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # 对话
    # ------------------------------------------------------------------

    def chat(self, query: str, session_id: Optional[str] = None) -> dict[str, Any]:
        """发送对话请求（非流式）。

        Args:
            query: 用户输入文本
            session_id: 可选的会话 ID，用于多轮对话上下文延续

        Returns:
            平台返回的完整对话响应字典，通常包含 ``answer``、``session_id`` 等字段
        """
        payload: dict[str, Any] = {"query": query}
        if session_id is not None:
            payload["session_id"] = session_id
        response = self._client.post("/api/v1/chat", json=payload)
        response.raise_for_status()
        return response.json()

    def chat_stream(
        self, query: str, session_id: Optional[str] = None
    ) -> Generator[dict[str, Any], None, None]:
        """发送流式对话请求，通过 SSE 逐步返回结果。

        解析规则：
        - ``event: <type>`` 行 → 提取事件类型
        - ``data: <json>`` 行 → 解析 JSON 数据
        - 每次迭代 yield ``{"event": etype, "data": parsed_data}``

        Args:
            query: 用户输入文本
            session_id: 可选的会话 ID

        Yields:
            包含 ``event`` 和 ``data`` 两个键的字典
        """
        payload: dict[str, Any] = {"query": query}
        if session_id is not None:
            payload["session_id"] = session_id

        with self._client.stream("POST", "/api/v1/chat/stream", json=payload) as response:
            response.raise_for_status()
            current_event: str = "message"
            for line in response.iter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("event:"):
                    current_event = line[len("event:") :].strip()
                elif line.startswith("data:"):
                    raw_data = line[len("data:") :].strip()
                    try:
                        parsed_data: Any = json.loads(raw_data)
                    except json.JSONDecodeError:
                        parsed_data = raw_data
                    yield {"event": current_event, "data": parsed_data}

    # ------------------------------------------------------------------
    # 文档上传
    # ------------------------------------------------------------------

    def upload_document(self, file_path: str, department: str = "General") -> dict[str, Any]:
        """上传文档到平台知识库。

        Args:
            file_path: 待上传文件的本地路径
            department: 所属部门标签，默认 ``"General"``

        Returns:
            上传结果字典，通常包含 ``document_id``、``status`` 等字段
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        with path.open("rb") as f:
            files = {"file": (path.name, f)}
            data = {"department": department}
            response = self._client.post("/api/v1/docs/upload", files=files, data=data)
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # 代码审查
    # ------------------------------------------------------------------

    def review_code(
        self,
        code_content: Optional[str] = None,
        code_url: Optional[str] = None,
        review_type: str = "full",
        stream: bool = False,
    ) -> dict[str, Any]:
        """提交代码审查请求。

        至少需要提供 ``code_content`` 或 ``code_url`` 之一。

        Args:
            code_content: 待审查的代码文本内容
            code_url: 待审查的代码 URL 地址
            review_type: 审查类型，可选 ``"full"``、``"security"``、
                ``"architecture"``、``"performance"``、``"style"``，默认 ``"full"``
            stream: 是否启用流式返回（当前版本未实现流式，预留参数）

        Returns:
            审查结果字典，包含 ``findings``、``summary`` 等字段
        """
        payload: dict[str, Any] = {
            "review_type": review_type,
            "stream": stream,
        }
        if code_content is not None:
            payload["code_content"] = code_content
        if code_url is not None:
            payload["code_url"] = code_url

        response = self._client.post("/review/code", json=payload)
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # 健康检查
    # ------------------------------------------------------------------

    def get_health(self) -> dict[str, Any]:
        """获取平台健康状态。

        Returns:
            健康检查结果字典，通常包含 ``status``、``version`` 等字段
        """
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # 生命周期管理
    # ------------------------------------------------------------------

    def close(self) -> None:
        """关闭底层 HTTP 客户端，释放连接资源。"""
        self._client.close()

    def __enter__(self) -> AgentPlatformClient:
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        self.close()


class AsyncAgentPlatformClient:
    """异步客户端，封装 Agent Platform 的所有 REST API。

    所有网络 I/O 均为异步，适用于 FastAPI / asyncio 等异步运行时。

    Args:
        base_url: 平台服务地址，例如 ``http://localhost:8000``
        api_key: API 密钥，用于 ``Authorization: Bearer {api_key}`` 认证
        timeout: 请求超时时间（秒），默认 120
    """

    def __init__(self, base_url: str, api_key: str, timeout: int = 120) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # 对话
    # ------------------------------------------------------------------

    async def chat(self, query: str, session_id: Optional[str] = None) -> dict[str, Any]:
        """发送异步对话请求（非流式）。

        Args:
            query: 用户输入文本
            session_id: 可选的会话 ID，用于多轮对话上下文延续

        Returns:
            平台返回的完整对话响应字典
        """
        payload: dict[str, Any] = {"query": query}
        if session_id is not None:
            payload["session_id"] = session_id
        response = await self._client.post("/api/v1/chat", json=payload)
        response.raise_for_status()
        return response.json()

    async def chat_stream(
        self, query: str, session_id: Optional[str] = None
    ) -> AsyncGenerator[dict[str, Any], None]:
        """发送异步流式对话请求，通过 SSE 逐步返回结果。

        解析规则与同步版本一致：
        - ``event: <type>`` 行 → 提取事件类型
        - ``data: <json>`` 行 → 解析 JSON 数据
        - 每次迭代 yield ``{"event": etype, "data": parsed_data}``

        Args:
            query: 用户输入文本
            session_id: 可选的会话 ID

        Yields:
            包含 ``event`` 和 ``data`` 两个键的字典
        """
        payload: dict[str, Any] = {"query": query}
        if session_id is not None:
            payload["session_id"] = session_id

        async with self._client.stream("POST", "/api/v1/chat/stream", json=payload) as response:
            response.raise_for_status()
            current_event: str = "message"
            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("event:"):
                    current_event = line[len("event:") :].strip()
                elif line.startswith("data:"):
                    raw_data = line[len("data:") :].strip()
                    try:
                        parsed_data: Any = json.loads(raw_data)
                    except json.JSONDecodeError:
                        parsed_data = raw_data
                    yield {"event": current_event, "data": parsed_data}

    # ------------------------------------------------------------------
    # 生命周期管理
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """关闭底层异步 HTTP 客户端，释放连接资源。"""
        await self._client.aclose()

    async def __aenter__(self) -> AsyncAgentPlatformClient:
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        await self.close()


__all__ = [
    "AgentPlatformClient",
    "AsyncAgentPlatformClient",
]
