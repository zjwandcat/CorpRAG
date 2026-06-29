"""可观测性模块单元测试

测试 app.observability.metrics 和 app.observability.logging_config 的核心功能：
- REQUEST_COUNT 指标递增
- @track_tool_call 装饰器（成功 / 异常路径）
- JsonFormatter JSON 结构化输出

当 prometheus_client 未安装时，metrics 相关测试优雅跳过。
"""

import asyncio
import json
import logging
from typing import Any

import pytest

from app.observability.logging_config import JsonFormatter
from app.observability.metrics import (
    _PROMETHEUS_AVAILABLE,
    REQUEST_COUNT,
    TOOL_CALL_COUNT,
    TOOL_CALL_LATENCY,
    track_tool_call,
)


# ---------------------------------------------------------------------------
# Prometheus 可用性标记 — 用于跳过需要真实指标的测试
# ---------------------------------------------------------------------------

requires_prometheus = pytest.mark.skipif(
    not _PROMETHEUS_AVAILABLE,
    reason="prometheus_client 未安装，跳过需要真实 Counter/Histogram 的测试",
)


# ===========================================================================
# TestMetrics — 测试 app/observability/metrics.py
# ===========================================================================


class TestMetrics:
    """Prometheus 指标与装饰器测试"""

    @requires_prometheus
    def test_request_count_inc(self) -> None:
        """REQUEST_COUNT 递增

        验证：调用 labels().inc() 后，Counter 值正确递增
        """
        # 先获取当前值
        metric_name: str = "agent_http_requests_total"
        labels: dict[str, str] = {"method": "GET", "endpoint": "/health", "status_code": "200"}

        # 记录递增前的值
        before: float = self._get_counter_value(metric_name, labels)

        # 执行递增
        REQUEST_COUNT.labels(method="GET", endpoint="/health", status_code=200).inc()

        # 验证递增后的值
        after: float = self._get_counter_value(metric_name, labels)
        assert after == before + 1

    @requires_prometheus
    def test_tool_decorator_success(self) -> None:
        """@track_tool_call 装饰器 — 正常函数

        验证：装饰的异步函数正常执行后，
        TOOL_CALL_COUNT (status=success) 递增，TOOL_CALL_LATENCY 记录延迟
        """
        tool_name: str = "test_tool_success"
        count_metric: str = "agent_tool_calls_total"
        latency_metric: str = "agent_tool_call_duration_seconds"

        success_labels: dict[str, str] = {"tool_name": tool_name, "status": "success"}
        latency_labels: dict[str, str] = {"tool_name": tool_name}

        before_count: float = self._get_counter_value(count_metric, success_labels)

        # 定义被装饰的异步函数
        @track_tool_call(tool_name)
        async def _normal_tool() -> str:
            return "ok"

        # 执行
        result: str = asyncio.run(_normal_tool())
        assert result == "ok"

        # 验证计数递增
        after_count: float = self._get_counter_value(count_metric, success_labels)
        assert after_count == before_count + 1

        # 验证延迟被记录（Histogram 的 _sum 值应 > 0）
        latency_sum: float = self._get_histogram_sum(latency_metric, latency_labels)
        assert latency_sum > 0

    @requires_prometheus
    def test_tool_decorator_error(self) -> None:
        """@track_tool_call 装饰器 — 异常函数

        验证：装饰的异步函数抛出异常后，
        TOOL_CALL_COUNT (status=error) 递增，TOOL_CALL_LATENCY 记录延迟，
        异常被正确重新抛出
        """
        tool_name: str = "test_tool_error"
        count_metric: str = "agent_tool_calls_total"
        latency_metric: str = "agent_tool_call_duration_seconds"

        error_labels: dict[str, str] = {"tool_name": tool_name, "status": "error"}
        latency_labels: dict[str, str] = {"tool_name": tool_name}

        before_count: float = self._get_counter_value(count_metric, error_labels)

        # 定义会抛出异常的异步函数
        @track_tool_call(tool_name)
        async def _failing_tool() -> None:
            raise ValueError("模拟工具异常")

        # 执行并捕获异常
        with pytest.raises(ValueError, match="模拟工具异常"):
            asyncio.run(_failing_tool())

        # 验证 error 计数递增
        after_count: float = self._get_counter_value(count_metric, error_labels)
        assert after_count == before_count + 1

        # 验证延迟被记录
        latency_sum: float = self._get_histogram_sum(latency_metric, latency_labels)
        assert latency_sum > 0

    # -----------------------------------------------------------------------
    # No-op 降级测试（prometheus_client 未安装时仍可运行）
    # -----------------------------------------------------------------------

    @pytest.mark.skipif(
        _PROMETHEUS_AVAILABLE,
        reason="仅在 prometheus_client 未安装时运行 no-op 降级测试",
    )
    def test_noop_metric_labels_inc(self) -> None:
        """No-op 桩类 — labels().inc() 不抛异常

        验证：prometheus_client 未安装时，_NoopMetric 的链式调用不报错
        """
        # 不应抛出任何异常
        REQUEST_COUNT.labels(method="GET", endpoint="/health", status_code=200).inc()
        TOOL_CALL_COUNT.labels(tool_name="any", status="success").inc()
        TOOL_CALL_LATENCY.labels(tool_name="any").observe(0.5)

    @pytest.mark.skipif(
        _PROMETHEUS_AVAILABLE,
        reason="仅在 prometheus_client 未安装时运行 no-op 降级测试",
    )
    def test_noop_track_tool_call_success(self) -> None:
        """No-op 降级 — @track_tool_call 装饰器正常函数仍可执行

        验证：prometheus_client 未安装时，装饰器不阻碍业务逻辑
        """

        @track_tool_call("noop_tool")
        async def _noop_tool() -> str:
            return "noop_ok"

        result: str = asyncio.run(_noop_tool())
        assert result == "noop_ok"

    @pytest.mark.skipif(
        _PROMETHEUS_AVAILABLE,
        reason="仅在 prometheus_client 未安装时运行 no-op 降级测试",
    )
    def test_noop_track_tool_call_error(self) -> None:
        """No-op 降级 — @track_tool_call 装饰器异常函数仍正确抛出异常

        验证：prometheus_client 未安装时，装饰器不吞掉异常
        """

        @track_tool_call("noop_failing_tool")
        async def _noop_failing_tool() -> None:
            raise RuntimeError("no-op 异常测试")

        with pytest.raises(RuntimeError, match="no-op 异常测试"):
            asyncio.run(_noop_failing_tool())

    # -----------------------------------------------------------------------
    # 辅助方法：从 Prometheus REGISTRY 读取指标值
    # -----------------------------------------------------------------------

    @staticmethod
    def _get_counter_value(
        metric_name: str,
        labels: dict[str, str],
    ) -> float:
        """从 Prometheus REGISTRY 读取 Counter 当前值

        Args:
            metric_name: 指标名称（如 agent_http_requests_total）
            labels: 标签键值对

        Returns:
            Counter 的当前值
        """
        from prometheus_client import REGISTRY

        for metric in REGISTRY.collect():
            if metric.name == metric_name:
                for sample in metric.samples:
                    if sample.labels == labels:
                        return sample.value
        return 0.0

    @staticmethod
    def _get_histogram_sum(
        metric_name: str,
        labels: dict[str, str],
    ) -> float:
        """从 Prometheus REGISTRY 读取 Histogram _sum 值

        Args:
            metric_name: 指标名称
            labels: 标签键值对

        Returns:
            Histogram 的 _sum 值
        """
        from prometheus_client import REGISTRY

        sum_suffix: str = "_sum"
        for metric in REGISTRY.collect():
            if metric.name == metric_name:
                for sample in metric.samples:
                    if sample.name == f"{metric_name}{sum_suffix}" and sample.labels == labels:
                        return sample.value
        return 0.0


# ===========================================================================
# TestLogging — 测试 app/observability/logging_config.py
# ===========================================================================


class TestLogging:
    """K8s JSON 结构化日志测试"""

    def test_json_output(self) -> None:
        """JsonFormatter 输出包含 message 和 timestamp

        验证：格式化后的日志为合法 JSON，且包含 message 和 timestamp 字段
        """
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="测试日志消息",
            args=None,
            exc_info=None,
        )

        output: str = formatter.format(record)
        parsed: dict[str, Any] = json.loads(output)

        # 验证核心字段存在
        assert "message" in parsed
        assert "timestamp" in parsed

        # 验证字段值
        assert parsed["message"] == "测试日志消息"
        assert isinstance(parsed["timestamp"], str)
        assert len(parsed["timestamp"]) > 0

        # 验证其他必要字段
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert parsed["line"] == 42

    def test_json_output_with_exception(self) -> None:
        """JsonFormatter 输出包含异常信息

        验证：当 LogRecord 包含异常时，输出 JSON 包含 exception 字段
        """
        formatter = JsonFormatter()

        try:
            raise ValueError("模拟异常")
        except ValueError:
            import sys

            exc_info: tuple[type[BaseException], BaseException, Any] | None = sys.exc_info()

        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=99,
            msg="发生异常",
            args=None,
            exc_info=exc_info,
        )

        output: str = formatter.format(record)
        parsed: dict[str, Any] = json.loads(output)

        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]
        assert "模拟异常" in parsed["exception"]

    def test_json_output_all_fields(self) -> None:
        """JsonFormatter 输出包含所有定义字段

        验证：格式化后的 JSON 包含文档中声明的所有字段
        """
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.WARNING,
            pathname="test.py",
            lineno=10,
            msg="完整字段测试",
            args=None,
            exc_info=None,
        )

        output: str = formatter.format(record)
        parsed: dict[str, Any] = json.loads(output)

        expected_fields: list[str] = [
            "timestamp",
            "level",
            "logger",
            "message",
            "module",
            "function",
            "line",
            "thread",
            "process",
        ]

        for field in expected_fields:
            assert field in parsed, f"缺少字段: {field}"

    def test_json_output_is_valid_json(self) -> None:
        """JsonFormatter 输出为合法单行 JSON

        验证：输出可以被 json.loads 解析，且为单行（无换行符）
        """
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.DEBUG,
            pathname="test.py",
            lineno=1,
            msg="JSON 合法性测试",
            args=None,
            exc_info=None,
        )

        output: str = formatter.format(record)

        # 验证可解析
        parsed: dict[str, Any] = json.loads(output)
        assert isinstance(parsed, dict)

        # 验证单行（无换行）
        assert "\n" not in output

    def test_json_output_chinese_message(self) -> None:
        """JsonFormatter 正确处理中文消息

        验证：中文消息在 JSON 输出中不被转义（ensure_ascii=False）
        """
        formatter = JsonFormatter()
        chinese_msg: str = "中文日志消息测试"
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg=chinese_msg,
            args=None,
            exc_info=None,
        )

        output: str = formatter.format(record)

        # ensure_ascii=False 时，中文字符应直接出现在输出中
        assert "中文日志消息测试" in output
