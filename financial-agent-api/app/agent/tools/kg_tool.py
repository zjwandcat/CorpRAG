"""知识图谱检索工具

提供 LangChain Tool 接口，供 Agent 在推理过程中调用知识图谱检索功能。
"""

import time

from langchain_core.tools import BaseTool, tool

from app.core.logging_config import log_agent_step, log_function_call, get_logger
from app.observability.metrics import TOOL_CALL_COUNT, TOOL_CALL_LATENCY
from app.rag.knowledge_graph import KnowledgeGraphManager

logger = get_logger(__name__)

__all__ = ["make_search_knowledge_graph_tool"]


def make_search_knowledge_graph_tool(
    kg_manager: KnowledgeGraphManager,
) -> BaseTool:
    """创建知识图谱检索工具

    Args:
        kg_manager: KnowledgeGraphManager 实例

    Returns:
        LangChain Tool 实例
    """

    @tool
    def search_knowledge_graph(
        entity: str, relation: str | None = None
    ) -> str:
        """基于知识图谱检索实体关系信息。用于查询实体间的关联关系，如组织架构、人员隶属、流程依赖等。

        Args:
            entity: 实体名称，如"张三"、"研发中心"、"报销流程"
            relation: 可选的关系类型过滤，如"隶属"、"负责"、"依赖"

        Returns:
            检索到的实体关系信息文本
        """
        step_start = time.monotonic()
        logger.info(
            "执行工具 search_knowledge_graph，entity=%s, relation=%s",
            entity,
            relation or "*",
        )

        try:
            results = kg_manager.search(entity=entity, relation=relation)

            if not results:
                result = f"知识图谱中未找到与「{entity}」相关的实体关系信息。"
                duration_ms = (time.monotonic() - step_start) * 1000
                logger.info("工具返回结果：%s", result)

                log_agent_step(
                    step_name="search_knowledge_graph",
                    tool_name="search_knowledge_graph",
                    duration_ms=duration_ms,
                    status="success",
                )
                log_function_call(
                    func_name="search_knowledge_graph",
                    kwargs={"entity": entity, "relation": relation},
                    duration_ms=duration_ms,
                    result_summary="no_results",
                )

                TOOL_CALL_COUNT.labels(
                    tool_name="search_knowledge_graph", status="success"
                ).inc()
                TOOL_CALL_LATENCY.labels(
                    tool_name="search_knowledge_graph"
                ).observe(duration_ms / 1000.0)

                return result

            # 格式化结果为可读文本
            formatted_parts: list[str] = []
            for r in results:
                source_info = f"（来源：{r.source_document}）" if r.source_document else ""
                formatted_parts.append(
                    f"- {r.entity} —[{r.relation}]→ {r.target_entity}{source_info}"
                )

            result = "\n".join(formatted_parts)

            duration_ms = (time.monotonic() - step_start) * 1000
            logger.info(
                "知识图谱检索到 %d 条关系，耗时=%.1fms",
                len(results),
                duration_ms,
            )

            log_agent_step(
                step_name="search_knowledge_graph",
                tool_name="search_knowledge_graph",
                duration_ms=duration_ms,
                status="success",
            )
            log_function_call(
                func_name="search_knowledge_graph",
                kwargs={"entity": entity, "relation": relation},
                duration_ms=duration_ms,
                result_summary=f"result_count={len(results)}",
            )

            TOOL_CALL_COUNT.labels(
                tool_name="search_knowledge_graph", status="success"
            ).inc()
            TOOL_CALL_LATENCY.labels(
                tool_name="search_knowledge_graph"
            ).observe(duration_ms / 1000.0)

            return result

        except Exception as exc:
            result = f"知识图谱检索暂时不可用（{type(exc).__name__}），请稍后重试。"
            duration_ms = (time.monotonic() - step_start) * 1000
            logger.warning(
                "知识图谱检索工具执行失败：%s，耗时=%.1fms", exc, duration_ms
            )

            log_agent_step(
                step_name="search_knowledge_graph",
                tool_name="search_knowledge_graph",
                duration_ms=duration_ms,
                status="error",
            )
            log_function_call(
                func_name="search_knowledge_graph",
                kwargs={"entity": entity, "relation": relation},
                duration_ms=duration_ms,
                result_summary=f"error: {type(exc).__name__}",
            )

            TOOL_CALL_COUNT.labels(
                tool_name="search_knowledge_graph", status="error"
            ).inc()
            TOOL_CALL_LATENCY.labels(
                tool_name="search_knowledge_graph"
            ).observe(duration_ms / 1000.0)

            return result

    return search_knowledge_graph