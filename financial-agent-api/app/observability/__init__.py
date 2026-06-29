"""可观测性模块

提供 Prometheus 指标、HTTP 中间件和 K8s 环境结构化日志配置。
涵盖请求计数、延迟分布、LLM 调用追踪、工具调用追踪等核心指标。
"""

from app.observability.metrics import (
    ACTIVE_SESSIONS,
    GPU_MEMORY_USED,
    GPU_UTILIZATION,
    LLM_CALL_COUNT,
    LLM_CALL_LATENCY,
    LLM_TOKEN_USAGE,
    PLATFORM_INFO,
    RAG_RETRIEVAL_COUNT,
    RAG_RETRIEVAL_LATENCY,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    TOOL_CALL_COUNT,
    TOOL_CALL_LATENCY,
    track_tool_call,
)

# metrics.py 内部变量名为 _PROMETHEUS_AVAILABLE（带下划线前缀），
# 此处重新导出为 PROMETHEUS_AVAILABLE 以保持公开 API 一致性
from app.observability.metrics import _PROMETHEUS_AVAILABLE as PROMETHEUS_AVAILABLE
from app.observability.middleware import CorrelationIdMiddleware, MetricsMiddleware
from app.observability.logging_config import JsonFormatter as K8sJsonFormatter
from app.observability.logging_config import setup_json_logging

__all__ = [
    # 指标
    "ACTIVE_SESSIONS",
    "GPU_MEMORY_USED",
    "GPU_UTILIZATION",
    "LLM_CALL_COUNT",
    "LLM_CALL_LATENCY",
    "LLM_TOKEN_USAGE",
    "PLATFORM_INFO",
    "PROMETHEUS_AVAILABLE",
    "RAG_RETRIEVAL_COUNT",
    "RAG_RETRIEVAL_LATENCY",
    "REQUEST_COUNT",
    "REQUEST_LATENCY",
    "TOOL_CALL_COUNT",
    "TOOL_CALL_LATENCY",
    "track_tool_call",
    # 中间件
    "CorrelationIdMiddleware",
    "MetricsMiddleware",
    # K8s 日志配置
    "K8sJsonFormatter",
    "setup_json_logging",
]
