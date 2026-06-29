"""双 RAG 引擎路由控制器模块

根据请求中的 rag_engine 字段路由到方案 A 或方案 B，
当方案 B 不可用时自动降级到方案 A。
支持知识图谱检索融合：向量检索完成后，额外调用 KG 图谱检索，
将图谱结果追加到 intermediate_steps。
"""

import logging
import time
from typing import TYPE_CHECKING, Any

import httpx

from app.core.enums import RAGEngine
from app.core.logging_config import log_function_call
from app.models.schemas import ChatResponse
from app.observability.metrics import RAG_RETRIEVAL_COUNT, RAG_RETRIEVAL_LATENCY
from app.rag.nvidia_blueprint_client import BlueprintClient

if TYPE_CHECKING:
    from app.rag.knowledge_graph import KnowledgeGraphManager

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
        kg_manager: "KnowledgeGraphManager | None" = None,
    ) -> None:
        self._agent_graph = agent_graph
        self._blueprint_client = blueprint_client
        self._kg_manager = kg_manager

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
        route_start = time.monotonic()

        # 方案 A：自建 RAG + Reranker
        if rag_engine == RAGEngine.BUILTIN or rag_engine is None:
            logger.info("路由决策：选择自建 RAG 引擎（BUILTIN）")
            response = await self._route_builtin(query, session_id)
            duration_ms = (time.monotonic() - route_start) * 1000
            log_function_call(
                func_name="EngineRouter.route",
                duration_ms=duration_ms,
                result_summary=f"引擎=BUILTIN，降级={response.is_fallback}",
            )
            RAG_RETRIEVAL_COUNT.labels(engine="builtin", status="success").inc()
            RAG_RETRIEVAL_LATENCY.labels(engine="builtin").observe(duration_ms / 1000.0)
            return response

        # 方案 B：NVIDIA Blueprint
        if rag_engine == RAGEngine.BLUEPRINT:
            logger.info("路由决策：选择 NVIDIA Blueprint 引擎（BLUEPRINT）")
            response = await self._route_blueprint(query, session_id)
            duration_ms = (time.monotonic() - route_start) * 1000
            log_function_call(
                func_name="EngineRouter.route",
                duration_ms=duration_ms,
                result_summary=f"引擎={response.rag_engine}，降级={response.is_fallback}",
            )
            # 根据是否降级记录指标
            metric_status = "fallback" if response.is_fallback else "success"
            RAG_RETRIEVAL_COUNT.labels(engine=str(response.rag_engine), status=metric_status).inc()
            RAG_RETRIEVAL_LATENCY.labels(engine=str(response.rag_engine)).observe(
                duration_ms / 1000.0
            )
            return response

        # rag_engine 非法值，默认使用方案 A
        logger.warning("未知的 RAG 引擎类型：%s，降级到自建 RAG", rag_engine)
        response = await self._route_builtin(query, session_id)
        duration_ms = (time.monotonic() - route_start) * 1000
        log_function_call(
            func_name="EngineRouter.route",
            duration_ms=duration_ms,
            result_summary=f"未知引擎={rag_engine}，降级到BUILTIN",
        )
        RAG_RETRIEVAL_COUNT.labels(engine="builtin", status="fallback").inc()
        RAG_RETRIEVAL_LATENCY.labels(engine="builtin").observe(duration_ms / 1000.0)
        return response

    async def _route_builtin(self, query: str, session_id: str | None) -> ChatResponse:
        """方案 A：自建 RAG + Reranker + 知识图谱融合检索"""
        route_start = time.monotonic()
        response = await self._agent_graph.run(query=query, session_id=session_id)
        route_duration = time.monotonic() - route_start
        # 补充 RAG 引擎信息
        response.rag_engine = RAGEngine.BUILTIN
        response.is_fallback = False
        response.fallback_message = ""

        # 知识图谱检索融合：向量检索完成后，额外调用 KG 图谱检索
        if self._kg_manager is not None:
            try:
                kg_results = self._kg_manager.search(entity=query)
                if kg_results:
                    from app.models.schemas import SourceReference, ToolCallStep

                    kg_sources = [
                        SourceReference(
                            source=kg.source_document or "知识图谱",
                            department="通用",
                            score=kg.confidence,
                            snippet=f"{kg.entity} -[{kg.relation}]-> {kg.target_entity}",
                        )
                        for kg in kg_results
                    ]
                    kg_step = ToolCallStep(
                        tool_name="search_knowledge_graph",
                        tool_args={"entity": query},
                        tool_result=f"图谱检索到 {len(kg_results)} 条关联关系",
                        tool_result_type="search_results",
                        sources=kg_sources,
                        duration_ms=0,
                        status="success",
                    )
                    response.intermediate_steps.append(kg_step)
                    logger.info(
                        "知识图谱融合检索完成，查询=%s，图谱结果数=%d",
                        query,
                        len(kg_results),
                    )
            except Exception as exc:
                logger.warning(
                    "知识图谱检索融合失败，降级为仅向量检索：%s（异常类型=%s）",
                    exc,
                    type(exc).__name__,
                )

        RAG_RETRIEVAL_COUNT.labels(engine="builtin", status="success").inc()
        RAG_RETRIEVAL_LATENCY.labels(engine="builtin").observe(route_duration)
        return response

    async def _route_blueprint(self, query: str, session_id: str | None) -> ChatResponse:
        """方案 B：NVIDIA Blueprint（含降级逻辑）"""
        # 检查 Blueprint 是否已配置
        if not self._blueprint_client.is_configured():
            logger.warning("Blueprint 未配置，降级到自建 RAG，原因：BlueprintClient 未完成配置")
            log_function_call(
                func_name="EngineRouter._route_blueprint",
                result_summary="降级：Blueprint未配置",
            )
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

            RAG_RETRIEVAL_COUNT.labels(engine="blueprint", status="success").inc()
            RAG_RETRIEVAL_LATENCY.labels(engine="blueprint").observe(0.0)

            return response

        except (httpx.TimeoutException, httpx.HTTPStatusError, ConnectionError) as exc:
            # Blueprint 不可用，降级到方案 A
            error_type = type(exc).__name__
            logger.warning(
                "Blueprint 不可用，降级到自建 RAG：%s（异常类型=%s）",
                exc,
                error_type,
            )
            log_function_call(
                func_name="EngineRouter._route_blueprint",
                result_summary=f"降级：Blueprint不可用（{error_type}）",
            )
            if isinstance(exc, httpx.TimeoutException):
                fallback_msg = "Blueprint 超时，已降级到自建 RAG"
            elif isinstance(exc, httpx.HTTPStatusError):
                fallback_msg = "Blueprint 错误，已降级到自建 RAG"
            else:
                fallback_msg = "Blueprint 连接失败，已降级到自建 RAG"
            return await self._fallback_to_builtin(query, session_id, fallback_msg)

        except Exception as exc:
            # 其他异常（格式异常等），降级到方案 A
            error_type = type(exc).__name__
            logger.warning(
                "Blueprint 响应异常，降级到自建 RAG：%s（异常类型=%s）",
                exc,
                error_type,
            )
            log_function_call(
                func_name="EngineRouter._route_blueprint",
                result_summary=f"降级：Blueprint响应异常（{error_type}）",
            )
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
        fallback_start = time.monotonic()

        logger.warning(
            "执行降级：Blueprint -> 自建 RAG，降级原因：%s",
            fallback_message,
        )

        response = await self._agent_graph.run(query=query, session_id=session_id)
        # 标记为降级
        response.rag_engine = RAGEngine.BUILTIN
        response.is_fallback = True
        response.fallback_message = fallback_message

        duration_ms = (time.monotonic() - fallback_start) * 1000
        log_function_call(
            func_name="EngineRouter._fallback_to_builtin",
            duration_ms=duration_ms,
            result_summary=f"降级原因={fallback_message}，降级引擎={RAGEngine.BUILTIN}",
        )

        RAG_RETRIEVAL_COUNT.labels(engine="builtin", status="fallback").inc()
        RAG_RETRIEVAL_LATENCY.labels(engine="builtin").observe(duration_ms / 1000.0)

        return response
