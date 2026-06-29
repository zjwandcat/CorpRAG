"""多Agent协作审查常量定义模块

集中管理审查类型到 Worker 维度的映射关系，
避免在 supervisor.py 和 graph.py 中重复定义。
"""

__all__ = ["REVIEW_TYPE_WORKERS"]

# 审查类型到 Worker 维度的映射
REVIEW_TYPE_WORKERS: dict[str, list[str]] = {
    "full": ["security", "architecture", "performance", "style"],
    "security": ["security"],
    "architecture": ["architecture"],
    "performance": ["performance"],
    "style": ["style"],
}
