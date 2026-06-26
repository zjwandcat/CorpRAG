"""NVIDIA RAG Blueprint 适配层模块

通过 OpenAI 兼容 API 与 NVIDIA RAG Blueprint 通信，
负责请求构造、响应解析、格式转换和降级信号。
"""

import logging
from dataclasses import dataclass

import httpx

from app.core.config import Settings
from app.core.enums import RAGEngine
from app.models.schemas import SourceReference

__all__ = ["BlueprintClient", "BlueprintResult"]

logger = logging.getLogger(__name__)

_BLUEPRINT_SYSTEM_PROMPT = (
    "你是一个企业内部办公知识库助手，请基于提供的文档内容准确回答用户问题。\n"
    "如果文档中没有相关信息，请如实告知用户。\n"
    "回答时请注明信息来源。"
)


@dataclass
class BlueprintResult:
    """Blueprint 查询结果"""

    answer: str  # Blueprint 生成的回答
    sources: list[SourceReference]  # 统一格式的溯源数据
    rag_engine: RAGEngine  # 固定为 RAGEngine.BLUEPRINT
    is_fallback: bool  # 固定为 False（降级由 EngineRouter 处理）


class BlueprintClient:
    """NVIDIA RAG Blueprint 适配层

    通过 OpenAI 兼容 API 与 Blueprint 通信，
    负责请求构造、响应解析、格式转换和降级信号。
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._api_url = settings.BLUEPRINT_API_URL
        self._api_key = settings.BLUEPRINT_API_KEY
        self._llm_model = settings.BLUEPRINT_LLM_MODELNAME
        self._embeddings_model = settings.BLUEPRINT_EMBEDDINGS_MODELNAME
        self._embeddings_dimensions = settings.BLUEPRINT_EMBEDDINGS_DIMENSIONS
        self._timeout = settings.BLUEPRINT_TIMEOUT

    def is_configured(self) -> bool:
        """检查 BLUEPRINT_API_URL 是否配置

        前端据此决定是否禁用 "NVIDIA Blueprint" 复选框。
        """
        return bool(self._api_url.strip())

    async def health_check(self) -> bool:
        """检查 Blueprint 是否可用

        向 {BLUEPRINT_API_URL}/v1/models 发送 GET 请求，
        返回 True/False 表示 Blueprint 是否可用。
        """
        if not self.is_configured():
            return False

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                headers: dict[str, str] = {}
                if self._api_key:
                    headers["Authorization"] = f"Bearer {self._api_key}"

                response = await client.get(
                    f"{self._api_url.rstrip('/')}/v1/models",
                    headers=headers,
                )
                return response.status_code == 200
        except Exception as exc:
            logger.warning("Blueprint 健康检查失败：%s", exc)
            return False

    async def query(self, query: str, department: str = "通用") -> BlueprintResult:
        """向 Blueprint 发送查询请求

        Args:
            query: 用户查询文本
            department: 部门名称，用于语义过滤

        Returns:
            BlueprintResult 包含回答和溯源数据

        Raises:
            httpx.TimeoutException: 请求超时
            httpx.HTTPStatusError: API 返回错误状态码
            ConnectionError: 连接失败
        """
        if not self.is_configured():
            raise ConnectionError("Blueprint API URL 未配置")

        # 构造 system prompt，注入部门上下文
        system_content = _BLUEPRINT_SYSTEM_PROMPT
        if department != "通用":
            system_content += f"\n\n请重点关注「{department}」相关的文档内容。"

        # 构造请求体
        request_body: dict = {
            "model": self._llm_model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": query},
            ],
            "temperature": 0.3,
        }

        # 维度适配：通过 extra_body 传递 Embedding 配置
        if self._embeddings_model or self._embeddings_dimensions:
            extra_body: dict[str, object] = {}
            if self._embeddings_model:
                extra_body["APP_EMBEDDINGS_MODELNAME"] = self._embeddings_model
            if self._embeddings_dimensions:
                extra_body["APP_EMBEDDINGS_DIMENSIONS"] = self._embeddings_dimensions
            request_body["extra_body"] = extra_body

        # 发送请求
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._api_url.rstrip('/')}/v1/chat/completions",
                json=request_body,
                headers=headers,
            )
            response.raise_for_status()

        # 解析响应
        data = response.json()
        answer = ""
        sources: list[SourceReference] = []

        # 从 choices[0].message.content 提取 answer
        choices = data.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            answer = str(content) if content else ""

        # 从响应 metadata 或 citations 字段提取溯源信息
        # Blueprint 可能通过不同方式返回溯源数据
        citations = data.get("citations", [])
        if citations:
            for i, citation in enumerate(citations):
                source_name = citation.get("source", citation.get("document", f"来源 {i + 1}"))
                snippet = citation.get("snippet", citation.get("text", ""))
                score = float(citation.get("score", 0.0))
                sources.append(
                    SourceReference(
                        source=str(source_name),
                        department=department,
                        score=score,
                        snippet=str(snippet)[:200],
                        rerank_score=None,
                    )
                )

        logger.info(
            "Blueprint 查询完成，回答长度：%d，溯源数：%d",
            len(answer),
            len(sources),
        )

        return BlueprintResult(
            answer=answer,
            sources=sources,
            rag_engine=RAGEngine.BLUEPRINT,
            is_fallback=False,
        )
