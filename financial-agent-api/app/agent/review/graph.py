"""多Agent审查状态图模块（已废弃）

.. deprecated::
    本模块的 LangGraph StateGraph 调度逻辑与 ``supervisor.py`` 的
    ``SupervisorAgent.dispatch()`` 完全重复。当前审查流程统一使用
    ``SupervisorAgent``（ThreadPoolExecutor 实现），本模块不再维护。

    保留 ``should_dispatch`` 辅助函数供外部查询审查维度映射，
    其余节点函数和图构建逻辑已移除。

迁移指引：
    - 调度审查任务 → 使用 ``SupervisorAgent.dispatch()``
    - 查询审查维度 → 使用 ``should_dispatch()`` 或直接引用 ``REVIEW_TYPE_WORKERS``
"""

import logging
import warnings
from typing import Any

from langchain_core.language_models import BaseChatModel

from app.agent.review.constants import REVIEW_TYPE_WORKERS
from app.agent.review.state import ReviewState

logger = logging.getLogger(__name__)

__all__ = ["build_review_graph", "should_dispatch"]


# =====================================================================
# 条件路由（保留：供外部查询审查维度映射）
# =====================================================================


def should_dispatch(state: ReviewState) -> list[str]:
    """根据 review_type 决定分发到哪些 Worker

    返回需要执行的 Worker 节点名称列表。
    也可用于外部模块查询某审查类型对应的维度列表。

    Args:
        state: 当前审查状态

    Returns:
        Worker 节点名称列表
    """
    review_type = state.get("review_type", "full")
    dimensions = REVIEW_TYPE_WORKERS.get(review_type, REVIEW_TYPE_WORKERS["full"])

    # 转换为节点名称
    worker_nodes: list[str] = [f"worker_{dim}" for dim in dimensions]
    logger.info(
        "条件路由：review_type=%s，分发到 %s",
        review_type,
        worker_nodes,
    )

    return worker_nodes


# =====================================================================
# 图构建（废弃：与 SupervisorAgent 重复）
# =====================================================================


def build_review_graph(
    llm: BaseChatModel,
    worker_timeout_seconds: int = 60,
) -> Any:
    """构建并编译多Agent审查状态图（已废弃）

    .. deprecated::
        本函数与 ``SupervisorAgent.dispatch()`` 实现相同的调度逻辑，
        但当前审查流程统一使用 SupervisorAgent。调用此函数将发出
        ``DeprecationWarning``。

        请迁移至 ``SupervisorAgent``：

        .. code-block:: python

            # 旧用法（已废弃）
            graph = build_review_graph(llm)
            result = graph.invoke(initial_state)

            # 新用法
            supervisor = SupervisorAgent(llm=llm)
            response = supervisor.dispatch(review_request)

    Args:
        llm: LangChain BaseChatModel 实例
        worker_timeout_seconds: Worker 执行超时时间（秒），默认 60

    Returns:
        编译后的 CompiledStateGraph

    Raises:
        DeprecationWarning: 始终发出废弃警告
    """
    warnings.warn(
        "build_review_graph 已废弃，与 SupervisorAgent.dispatch() 功能重复。"
        "请迁移至 SupervisorAgent，详见文档。",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.warning(
        "build_review_graph 已废弃，与 SupervisorAgent.dispatch() 功能重复，"
        "请迁移至 SupervisorAgent"
    )

    # 延迟导入，避免在模块加载时引入不必要的依赖
    from langgraph.graph import END, StateGraph

    from app.agent.review.workers.architecture_agent import ArchitectureAgent
    from app.agent.review.workers.performance_agent import PerformanceAgent
    from app.agent.review.workers.security_agent import SecurityAgent
    from app.agent.review.workers.style_agent import StyleAgent
    from app.agent.review.workers.summary_agent import SummaryAgent
    from app.core.enums import ReviewStatus, ReviewType
    from app.models.schemas import WorkerResult

    # ---- Worker 实例缓存 ----
    _worker_instances: dict[str, Any] = {}
    _summary_instance: SummaryAgent | None = None

    def _get_worker(dimension: str) -> Any:
        """获取或创建 Worker Agent 实例"""
        if dimension not in _worker_instances:
            worker_classes: dict[str, type] = {
                "security": SecurityAgent,
                "architecture": ArchitectureAgent,
                "performance": PerformanceAgent,
                "style": StyleAgent,
            }
            worker_cls = worker_classes.get(dimension)
            if worker_cls is None:
                raise ValueError(f"未知的审查维度：{dimension}")
            _worker_instances[dimension] = worker_cls(
                llm=llm,
                worker_timeout_seconds=worker_timeout_seconds,
            )
        return _worker_instances[dimension]

    def _get_summary_agent() -> SummaryAgent:
        """获取或创建 SummaryAgent 实例"""
        nonlocal _summary_instance
        if _summary_instance is None:
            _summary_instance = SummaryAgent(
                llm=llm,
                worker_timeout_seconds=worker_timeout_seconds,
            )
        return _summary_instance

    # ---- 节点函数 ----

    def supervisor_node(state: ReviewState) -> dict:
        """Supervisor 节点：解析审查类型，初始化状态"""
        review_type = state.get("review_type", "full")
        logger.info("Supervisor 节点：解析审查类型 [%s]", review_type)
        return {
            "current_status": "supervisor_dispatching",
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    def _make_worker_node(dimension: str) -> Any:
        """创建 Worker 节点函数"""

        def worker_node(state: ReviewState) -> dict:
            """Worker 节点：执行指定维度的审查任务"""
            code_content = state.get("code_content", "")
            trace_id = state.get("trace_id", "")
            worker_results = dict(state.get("worker_results", {}))

            logger.info(
                "Worker [%s] 节点开始执行，trace_id=%s",
                dimension,
                trace_id,
            )

            worker = _worker_instances.get(dimension)
            if worker is None:
                logger.error("Worker [%s] 实例未找到", dimension)
                worker_results[dimension] = WorkerResult(
                    dimension=ReviewType(dimension),
                    status=ReviewStatus.FAILED,
                    findings=[],
                    duration_ms=0,
                    error_message=f"Worker [{dimension}] 实例未初始化",
                )
                return {
                    "worker_results": worker_results,
                    "current_status": "workers_executing",
                }

            context: dict[str, str] = {"trace_id": trace_id}
            result = worker.safe_execute(code_content=code_content, context=context)
            worker_results[dimension] = result

            logger.info(
                "Worker [%s] 节点执行完成，状态：%s",
                dimension,
                result.status,
            )

            return {
                "worker_results": worker_results,
                "current_status": "workers_executing",
            }

        return worker_node

    def summary_node(state: ReviewState) -> dict:
        """Summary 节点：汇总所有 Worker 结果"""
        worker_results_dict = state.get("worker_results", {})
        worker_results = list(worker_results_dict.values())
        code_content = state.get("code_content", "")
        trace_id = state.get("trace_id", "")

        logger.info(
            "Summary 节点开始汇总，维度数：%d，trace_id=%s",
            len(worker_results),
            trace_id,
        )

        summary_agent = _summary_instance
        if summary_agent is None:
            logger.error("SummaryAgent 实例未初始化")
            return {
                "current_status": "error",
                "is_fallback": True,
                "fallback_message": "SummaryAgent 实例未初始化",
            }

        try:
            summary_report = summary_agent.generate_summary(
                code_content=code_content,
                worker_results=worker_results,
            )
        except Exception as exc:
            logger.error("Summary 执行失败：%s", exc, exc_info=True)
            summary_report = f"汇总报告生成失败：{exc!s}"

        logger.info("Summary 节点汇总完成")

        return {
            "current_status": "completed",
            "summary_report": summary_report,
        }

    # ---- 图构建 ----

    # 初始化 Worker 实例
    for dimension in ("security", "architecture", "performance", "style"):
        _get_worker(dimension)
    _get_summary_agent()

    # 创建状态图
    graph = StateGraph(ReviewState)

    # 添加节点
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("worker_security", _make_worker_node("security"))
    graph.add_node("worker_architecture", _make_worker_node("architecture"))
    graph.add_node("worker_performance", _make_worker_node("performance"))
    graph.add_node("worker_style", _make_worker_node("style"))
    graph.add_node("summary", summary_node)

    # 设置入口
    graph.set_entry_point("supervisor")

    # 添加条件边
    graph.add_conditional_edges(
        "supervisor",
        should_dispatch,
        {
            "worker_security": "worker_security",
            "worker_architecture": "worker_architecture",
            "worker_performance": "worker_performance",
            "worker_style": "worker_style",
        },
    )

    # 添加汇聚边
    graph.add_edge("worker_security", "summary")
    graph.add_edge("worker_architecture", "summary")
    graph.add_edge("worker_performance", "summary")
    graph.add_edge("worker_style", "summary")

    # 终止边
    graph.add_edge("summary", END)

    # 编译图
    compiled = graph.compile()

    logger.info(
        "审查状态图构建完成（已废弃），节点：supervisor, worker_security, "
        "worker_architecture, worker_performance, worker_style, summary"
    )
    return compiled
