"""多Agent协作审查模块

基于 SupervisorAgent（ThreadPoolExecutor）实现多Agent协作代码审查系统。
LangGraph StateGraph 路径（build_review_graph）已废弃，请使用 SupervisorAgent。
"""

from app.agent.review.base_worker import BaseWorkerAgent
from app.agent.review.graph import build_review_graph  # noqa: F401 — 保留向后兼容，已废弃
from app.agent.review.state import ReviewState
from app.agent.review.supervisor import SupervisorAgent

__all__ = [
    "BaseWorkerAgent",
    "ReviewState",
    "SupervisorAgent",
    "build_review_graph",  # 已废弃，保留向后兼容
]
