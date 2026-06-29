"""
Agent 模块包：工具定义、核心逻辑和 LangGraph 状态图。

注意：build_review_graph 已废弃，请使用 SupervisorAgent。
"""

# __all__ 声明：明确导出的公共接口，便于外部模块导入
__all__ = [
    "AgentChain",
    "AgentGraph",
    "AgentState",
    "BaseWorkerAgent",
    "ReviewState",
    "SupervisorAgent",
    "build_agent_graph",
    "build_review_graph",  # 已废弃，保留向后兼容
]
