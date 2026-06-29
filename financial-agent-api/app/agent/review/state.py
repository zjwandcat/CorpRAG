"""审查状态定义模块

定义多Agent协作审查流程的全局状态对象 ReviewState，
复用现有 AgentState 的 Annotated + add_messages 模式。
"""

import logging
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

logger = logging.getLogger(__name__)

__all__ = ["ReviewState"]


class ReviewState(TypedDict):
    """多Agent协作审查状态对象

    管理审查会话的全局状态，贯穿 Supervisor 调度、Worker 执行、
    Summary 汇总等整个审查流程。

    Attributes:
        messages: 对话消息列表，使用 add_messages reducer 自动追加
        session_id: 审查会话唯一标识
        review_type: 审查类型（full/security/architecture/performance/style）
        code_content: 待审查的代码内容
        code_url: 代码仓库 PR 链接
        worker_results: 各维度 Worker 执行结果（维度名 → WorkerResult）
        current_status: 当前审查状态
        iteration_count: 迭代计数器
        summary_report: Summary Agent 生成的汇总报告
        is_fallback: 是否降级为单 Agent 模式
        fallback_message: 降级提示信息
        trace_id: 全链路追踪标识
    """

    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    review_type: str  # full/security/architecture/performance/style
    code_content: str
    code_url: str
    worker_results: dict[str, Any]  # dimension -> WorkerResult
    current_status: Literal[
        "idle",
        "supervisor_dispatching",
        "workers_executing",
        "summary_generating",
        "completed",
        "error",
        "fallback",
    ]
    iteration_count: int
    summary_report: str
    is_fallback: bool
    fallback_message: str
    trace_id: str
