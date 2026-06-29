"""Starlette 可观测性中间件模块

为 financial-agent-api 提供两个核心中间件：
- MetricsMiddleware：自动记录 HTTP 请求计数和延迟分布到 Prometheus
- CorrelationIdMiddleware：管理请求关联 ID，贯穿日志和响应链路

使用方式：
    from app.observability.middleware import MetricsMiddleware, CorrelationIdMiddleware

    app.add_middleware(MetricsMiddleware)
    app.add_middleware(CorrelationIdMiddleware)
"""

import uuid
from typing import Any, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.observability.metrics import REQUEST_COUNT, REQUEST_LATENCY


# ============================================================================
# 指标中间件
# ============================================================================


class MetricsMiddleware(BaseHTTPMiddleware):
    """HTTP 请求指标采集中间件

    自动为每个 HTTP 请求记录：
    - REQUEST_COUNT：按 method / endpoint / status_code 分类的请求计数
    - REQUEST_LATENCY：按 method / endpoint 分类的请求延迟分布

    排除 /metrics 端点本身，避免 Prometheus 抓取产生自引用指标。
    """

    async def dispatch(self, request: Request, call_next: Callable[..., Any]) -> Response:
        """处理请求并记录指标

        Args:
            request: Starlette 请求对象
            call_next: 下一个中间件/路由处理函数

        Returns:
            响应对象
        """
        # 排除 Prometheus 自身抓取端点，避免自引用
        if request.url.path == "/metrics":
            return await call_next(request)

        method = request.method
        endpoint = request.url.path

        # 记录请求开始时间
        import time

        start_time = time.perf_counter()

        # 执行后续处理
        response = await call_next(request)

        # 计算延迟
        elapsed = time.perf_counter() - start_time

        # 记录指标
        status_code = str(response.status_code)
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code,
        ).inc()
        REQUEST_LATENCY.labels(
            method=method,
            endpoint=endpoint,
        ).observe(elapsed)

        return response


# ============================================================================
# 关联 ID 中间件
# ============================================================================


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """请求关联 ID 中间件

    从请求头 X-Correlation-ID 读取关联 ID，若不存在则自动生成 UUID v4。
    将关联 ID 注入到：
    - request.state.correlation_id：供后续处理链路使用
    - 响应头 X-Correlation-ID：供调用方追踪

    关联 ID 可与 app.core.logging_config 的 set_request_id 配合使用，
    实现跨模块日志关联追踪。
    """

    async def dispatch(self, request: Request, call_next: Callable[..., Any]) -> Response:
        """处理请求并注入关联 ID

        Args:
            request: Starlette 请求对象
            call_next: 下一个中间件/路由处理函数

        Returns:
            带有 X-Correlation-ID 响应头的响应对象
        """
        # 从请求头读取或生成关联 ID
        correlation_id: str = request.headers.get(
            "X-Correlation-ID",
            str(uuid.uuid4()),
        )

        # 注入到 request.state，供后续处理链路访问
        request.state.correlation_id = correlation_id

        # 执行后续处理
        response = await call_next(request)

        # 将关联 ID 写入响应头
        response.headers["X-Correlation-ID"] = correlation_id

        return response


# ============================================================================
# 公开 API 声明
# ============================================================================

__all__ = [
    "CorrelationIdMiddleware",
    "MetricsMiddleware",
]
