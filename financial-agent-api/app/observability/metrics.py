"""Prometheus 指标定义模块

为 financial-agent-api 提供完整的 Prometheus 可观测性指标，涵盖：
- HTTP 请求指标（请求计数、延迟分布）
- LLM 调用指标（调用计数、延迟、Token 用量）
- 工具调用指标（调用计数、延迟）
- RAG 检索指标（检索计数、延迟）
- 运行时指标（活跃会话数、GPU 利用率/显存）
- 平台信息指标

当 prometheus_client 未安装时，所有指标和装饰器自动降级为空操作（no-op），
不影响核心业务功能。

使用方式：
    from app.observability.metrics import REQUEST_COUNT, track_tool_call

    # 记录 HTTP 请求
    REQUEST_COUNT.labels(method="GET", endpoint="/health", status_code=200).inc()

    # 使用装饰器追踪工具调用
    @track_tool_call("search_documents")
    async def search_documents(query: str) -> list[dict]:
        ...
"""

import functools
import time
from typing import Any, Callable, Coroutine, TypeVar, Union

# ============================================================================
# Prometheus 指标导入（优雅降级）
# ============================================================================

_PROMETHEUS_AVAILABLE: bool

try:
    from prometheus_client import Counter, Gauge, Histogram, Info

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False


# ============================================================================
# No-op 桩类：当 prometheus_client 未安装时使用
# ============================================================================


class _NoopMetric:
    """空操作指标桩类

    当 prometheus_client 未安装时，所有指标操作（inc, dec, observe, labels 等）
    均为空操作，确保核心业务逻辑不受影响。
    """

    def labels(self, *args: Any, **kwargs: Any) -> "_NoopMetric":
        """空操作：返回自身以支持链式调用

        Args:
            *args: 位置参数（忽略）
            **kwargs: 关键字参数（忽略）

        Returns:
            自身实例，支持链式调用
        """
        return self

    def inc(self, amount: float = 1.0) -> None:
        """空操作：递增指标

        Args:
            amount: 递增量（忽略）
        """

    def dec(self, amount: float = 1.0) -> None:
        """空操作：递减指标

        Args:
            amount: 递减量（忽略）
        """

    def observe(self, amount: float) -> None:
        """空操作：观测值

        Args:
            amount: 观测值（忽略）
        """

    def set(self, value: float) -> None:
        """空操作：设置值

        Args:
            value: 设置值（忽略）
        """

    def info(self, val: dict[str, str]) -> None:
        """空操作：设置信息

        Args:
            val: 信息字典（忽略）
        """


# ============================================================================
# 指标实例化
# ============================================================================

if _PROMETHEUS_AVAILABLE:
    # ---- HTTP 请求指标 ----
    REQUEST_COUNT: Union[Counter, _NoopMetric] = Counter(
        "agent_http_requests_total",
        "Total count of HTTP requests by method, endpoint and status code",
        labelnames=["method", "endpoint", "status_code"],
    )

    REQUEST_LATENCY: Union[Histogram, _NoopMetric] = Histogram(
        "agent_http_request_duration_seconds",
        "HTTP request latency in seconds by method and endpoint",
        labelnames=["method", "endpoint"],
        buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
    )

    # ---- LLM 调用指标 ----
    LLM_CALL_COUNT: Union[Counter, _NoopMetric] = Counter(
        "agent_llm_calls_total",
        "Total count of LLM API calls by provider, model and status",
        labelnames=["provider", "model", "status"],
    )

    LLM_CALL_LATENCY: Union[Histogram, _NoopMetric] = Histogram(
        "agent_llm_call_duration_seconds",
        "LLM API call latency in seconds by provider and model",
        labelnames=["provider", "model"],
        buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
    )

    LLM_TOKEN_USAGE: Union[Counter, _NoopMetric] = Counter(
        "agent_llm_tokens_total",
        "Total LLM token usage by provider, model and type (prompt/completion)",
        labelnames=["provider", "model", "type"],
    )

    # ---- 工具调用指标 ----
    TOOL_CALL_COUNT: Union[Counter, _NoopMetric] = Counter(
        "agent_tool_calls_total",
        "Total count of tool calls by tool name and status",
        labelnames=["tool_name", "status"],
    )

    TOOL_CALL_LATENCY: Union[Histogram, _NoopMetric] = Histogram(
        "agent_tool_call_duration_seconds",
        "Tool call latency in seconds by tool name",
        labelnames=["tool_name"],
        buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )

    # ---- RAG 检索指标 ----
    RAG_RETRIEVAL_COUNT: Union[Counter, _NoopMetric] = Counter(
        "agent_rag_retrievals_total",
        "Total count of RAG retrievals by engine and status",
        labelnames=["engine", "status"],
    )

    RAG_RETRIEVAL_LATENCY: Union[Histogram, _NoopMetric] = Histogram(
        "agent_rag_retrieval_duration_seconds",
        "RAG retrieval latency in seconds by engine",
        labelnames=["engine"],
        buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )

    # ---- 运行时指标 ----
    ACTIVE_SESSIONS: Union[Gauge, _NoopMetric] = Gauge(
        "agent_active_sessions",
        "Number of currently active sessions",
    )

    GPU_UTILIZATION: Union[Gauge, _NoopMetric] = Gauge(
        "agent_gpu_utilization_ratio",
        "GPU utilization ratio (0.0 - 1.0) by device",
        labelnames=["device"],
    )

    GPU_MEMORY_USED: Union[Gauge, _NoopMetric] = Gauge(
        "agent_gpu_memory_used_bytes",
        "GPU memory used in bytes by device",
        labelnames=["device"],
    )

    # ---- 平台信息指标 ----
    PLATFORM_INFO: Union[Info, _NoopMetric] = Info(
        "agent_platform",
        "Platform and runtime information",
    )

    # ---- Knowledge Graph 指标 ----
    KG_TRIPLET_COUNT: Union[Counter, _NoopMetric] = Counter(
        "agent_kg_triplet_total",
        "Total count of knowledge graph triplets by operation",
        labelnames=["operation"],
    )

    KG_SEARCH_LATENCY: Union[Histogram, _NoopMetric] = Histogram(
        "agent_kg_search_duration_seconds",
        "Knowledge graph search latency in seconds by entity",
        labelnames=["entity"],
        buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    )

    # ---- HITL 审批指标 ----
    HITL_APPROVAL_COUNT: Union[Counter, _NoopMetric] = Counter(
        "agent_hitl_approval_total",
        "Total count of HITL approvals by tool name and action",
        labelnames=["tool_name", "action"],
    )

    HITL_APPROVAL_PENDING: Union[Gauge, _NoopMetric] = Gauge(
        "agent_hitl_approval_pending",
        "Number of currently pending HITL approvals",
    )

    # ---- Guardrails 指标 ----
    GUARDRAIL_INTERVENTION_COUNT: Union[Counter, _NoopMetric] = Counter(
        "agent_guardrail_intervention_total",
        "Total count of guardrail interventions by tool name and action",
        labelnames=["tool_name", "action"],
    )

    # ---- Cache 指标 ----
    CACHE_HIT_TOTAL: Union[Counter, _NoopMetric] = Counter(
        "agent_cache_hit_total",
        "Total cache hits",
    )

    CACHE_MISS_TOTAL: Union[Counter, _NoopMetric] = Counter(
        "agent_cache_miss_total",
        "Total cache misses",
    )

    # ---- Drift 指标 ----
    DRIFT_ALERT_TOTAL: Union[Counter, _NoopMetric] = Counter(
        "agent_drift_alerts_total",
        "Total drift alerts",
        labelnames=["method"],
    )

    DRIFT_SCORE: Union[Gauge, _NoopMetric] = Gauge(
        "agent_drift_score",
        "Current drift score",
        labelnames=["method"],
    )
else:
    # prometheus_client 未安装，使用 no-op 桩
    REQUEST_COUNT = _NoopMetric()
    REQUEST_LATENCY = _NoopMetric()
    LLM_CALL_COUNT = _NoopMetric()
    LLM_CALL_LATENCY = _NoopMetric()
    LLM_TOKEN_USAGE = _NoopMetric()
    TOOL_CALL_COUNT = _NoopMetric()
    TOOL_CALL_LATENCY = _NoopMetric()
    RAG_RETRIEVAL_COUNT = _NoopMetric()
    RAG_RETRIEVAL_LATENCY = _NoopMetric()
    ACTIVE_SESSIONS = _NoopMetric()
    GPU_UTILIZATION = _NoopMetric()
    GPU_MEMORY_USED = _NoopMetric()
    PLATFORM_INFO = _NoopMetric()
    KG_TRIPLET_COUNT = _NoopMetric()
    KG_SEARCH_LATENCY = _NoopMetric()
    HITL_APPROVAL_COUNT = _NoopMetric()
    HITL_APPROVAL_PENDING = _NoopMetric()
    GUARDRAIL_INTERVENTION_COUNT = _NoopMetric()
    CACHE_HIT_TOTAL = _NoopMetric()
    CACHE_MISS_TOTAL = _NoopMetric()
    DRIFT_ALERT_TOTAL = _NoopMetric()
    DRIFT_SCORE = _NoopMetric()


# ============================================================================
# 装饰器
# ============================================================================

_F = TypeVar("_F", bound=Callable[..., Coroutine[Any, Any, Any]])


def track_tool_call(tool_name: str) -> Callable[[_F], _F]:
    """异步工具调用追踪装饰器

    自动记录 TOOL_CALL_COUNT 和 TOOL_CALL_LATENCY，无需手动埋点。
    成功时 status="success"，异常时 status="error" 并重新抛出异常。

    Args:
        tool_name: 工具名称，用于指标标签

    Returns:
        装饰后的异步函数

    Example:
        @track_tool_call("search_documents")
        async def search_documents(query: str) -> list[dict]:
            ...
    """

    def decorator(func: _F) -> _F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                TOOL_CALL_COUNT.labels(tool_name=tool_name, status="success").inc()
                TOOL_CALL_LATENCY.labels(tool_name=tool_name).observe(elapsed)
                return result
            except Exception:
                elapsed = time.perf_counter() - start_time
                TOOL_CALL_COUNT.labels(tool_name=tool_name, status="error").inc()
                TOOL_CALL_LATENCY.labels(tool_name=tool_name).observe(elapsed)
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


# 公共别名：去除下划线前缀，便于外部导入
PROMETHEUS_AVAILABLE: bool = _PROMETHEUS_AVAILABLE


# ============================================================================
# 公开 API 声明
# ============================================================================

__all__ = [
    "ACTIVE_SESSIONS",
    "CACHE_HIT_TOTAL",
    "CACHE_MISS_TOTAL",
    "DRIFT_ALERT_TOTAL",
    "DRIFT_SCORE",
    "GPU_MEMORY_USED",
    "GPU_UTILIZATION",
    "GUARDRAIL_INTERVENTION_COUNT",
    "HITL_APPROVAL_COUNT",
    "HITL_APPROVAL_PENDING",
    "KG_SEARCH_LATENCY",
    "KG_TRIPLET_COUNT",
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
]
