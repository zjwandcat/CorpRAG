"""双 RAG 引擎路由控制器模块

根据请求中的 rag_engine 字段路由到方案 A 或方案 B，
当方案 B 不可用时自动降级到方案 A。
"""

import logging
from typing import Any

import httpx

from app.core.enums import RAGEngine
from app.models.schemas import ChatResponse
from app.rag.nvidia_blueprint_client import BlueprintClient

__all__ = ["EngineRouter"]

logger = logging.getLogger(__name__)


class EngineRouter:
    """双 RAG 引擎路由控制器

    根据请求中的 rag_engine 字段路由到方案 A 或方案 B，
    当方案 B 不可用时自动降级到方案 A。
    """

    def __init__(
        self,
        agent_graph: Any,
        blueprint_client: BlueprintClient,
    ) -> None:
        self._agent_graph = agent_graph
        self._blueprint_client = blueprint_client

    async def route(
        self,
        query: str,
        session_id: str | None,
        rag_engine: RAGEngine,
    ) -> ChatResponse:
        """根据 rag_engine 字段路由到方案 A 或方案 B

        Args:
            query: 用户查询
            session_id: 会话 ID
            rag_engine: RAG 引擎选择

        Returns:
            ChatResponse 包含回答、溯源、引擎信息等
        """
        # 方案 A：自建 RAG + Reranker
        if rag_engine == RAGEngine.BUILTIN or rag_engine is None:
            return await self._route_builtin(query, session_id)

        # 方案 B：NVIDIA Blueprint
        if rag_engine == RAGEngine.BLUEPRINT:
            return await self._route_blueprint(query, session_id)

        # rag_engine 非法值，默认使用方案 A
        logger.warning("未知的 RAG 引擎类型：%s，降级到自建 RAG", rag_engine)
        return await self._route_builtin(query, session_id)

    async def _route_builtin(self, query: str, session_id: str | None) -> ChatResponse:
        """方案 A：自建 RAG + Reranker"""
        import asyncio

        response = await asyncio.to_thread(
            self._agent_graph.run, query=query, session_id=session_id
        )
        # 补充 RAG 引擎信息
        response.rag_engine = RAGEngine.BUILTIN
        response.is_fallback = False
        response.fallback_message = ""
        return response

    async def _route_blueprint(self, query: str, session_id: str | None) -> ChatResponse:
        """方案 B：NVIDIA Blueprint（含降级逻辑）"""
        # 检查 Blueprint 是否已配置
        if not self._blueprint_client.is_configured():
            logger.warning("Blueprint 未配置，降级到自建 RAG")
            return await self._fallback_to_builtin(
                query, session_id, "Blueprint 未配置，已降级到自建 RAG"
            )

        try:
            # 调用 Blueprint 查询
            result = await self._blueprint_client.query(query=query)

            # 构造 ChatResponse
            from uuid import uuid4

            if session_id is None:
                session_id = str(uuid4())

            response = ChatResponse(
                answer=result.answer,
                answer_format="markdown",
                tools_used=[],
                intermediate_steps=[],
                total_duration_ms=0,
                session_id=session_id,
                rag_engine=RAGEngine.BLUEPRINT,
                is_fallback=False,
                fallback_message="",
            )

            # 如果 Blueprint 返回了溯源数据，添加到 intermediate_steps
            if result.sources:
                from app.models.schemas import ToolCallStep

                step = ToolCallStep(
                    tool_name="blueprint_rag",
                    tool_args={"query": query},
                    tool_result=result.answer[:500],
                    tool_result_type="search_results",
                    sources=result.sources,
                    duration_ms=0,
                    status="success",
                )
                response.intermediate_steps = [step]

            return response

        except (httpx.TimeoutException, httpx.HTTPStatusError, ConnectionError) as exc:
            # Blueprint 不可用，降级到方案 A
            logger.warning("Blueprint 不可用，降级到自建 RAG：%s", exc)
            error_type = type(exc).__name__
            if isinstance(exc, httpx.TimeoutException):
                fallback_msg = "Blueprint 超时，已降级到自建 RAG"
            elif isinstance(exc, httpx.HTTPStatusError):
                fallback_msg = "Blueprint 错误，已降级到自建 RAG"
            else:
                fallback_msg = "Blueprint 连接失败，已降级到自建 RAG"
            return await self._fallback_to_builtin(query, session_id, fallback_msg)

        except Exception as exc:
            # 其他异常（格式异常等），降级到方案 A
            logger.warning("Blueprint 响应异常，降级到自建 RAG：%s", exc)
            return await self._fallback_to_builtin(
                query, session_id, "Blueprint 响应异常，已降级到自建 RAG"
            )

    async def _fallback_to_builtin(
        self,
        query: str,
        session_id: str | None,
        fallback_message: str,
    ) -> ChatResponse:
        """降级到方案 A

        Args:
            query: 用户查询
            session_id: 会话 ID
            fallback_message: 降级提示信息

        Returns:
            ChatResponse 使用方案 A 的结果，标记为降级
        """
        import asyncio

        response = await asyncio.to_thread(
            self._agent_graph.run, query=query, session_id=session_id
        )
        # 标记为降级
        response.rag_engine = RAGEngine.BUILTIN
        response.is_fallback = True
        response.fallback_message = fallback_message
        return response
