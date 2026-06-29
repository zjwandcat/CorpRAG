"""K8s 环境结构化 JSON 日志配置

为 Kubernetes 容器环境提供轻量级 JSON 日志配置方案。
与 ``app.core.logging_config`` 兼容，是其 K8s 场景下的替代方案。

核心差异：
- ``app.core.logging_config``：完整的日志基础设施，包含文件轮转、
  多文件输出（app.log / error.log / audit.log）、请求追踪 ID 等
- ``app.observability.logging_config``：轻量级方案，仅输出 JSON 到 stdout，
  适合 K8s 中由 Fluentd / Filebeat 采集容器标准输出的场景

使用方式：
    from app.observability.logging_config import setup_json_logging

    # 在应用启动时调用（替代 app.core.logging_config.setup_logging）
    setup_json_logging(level="INFO")
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


# ============================================================================
# 需要抑制日志的第三方库及其级别
# ============================================================================

_NOISY_LIBRARIES: dict[str, int] = {
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "chromadb": logging.WARNING,
    "urllib3": logging.WARNING,
    "multipart": logging.WARNING,
    "watchfiles": logging.WARNING,
    "sse_starlette": logging.WARNING,
    "http11": logging.WARNING,
}


# ============================================================================
# JSON 结构化日志格式化器
# ============================================================================


class JsonFormatter(logging.Formatter):
    """K8s 环境 JSON 结构化日志格式化器

    将日志记录格式化为单行 JSON 字符串，输出到 stdout，
    便于 Fluentd / Filebeat / Grafana Loki 等日志采集器解析。

    输出字段：
    - timestamp: ISO 8601 格式时间戳（UTC）
    - level: 日志级别
    - logger: logger 名称
    - message: 日志消息
    - module: 模块名
    - function: 函数名
    - line: 行号
    - thread: 线程名
    - process: 进程 ID

    与 ``app.core.logging_config.JsonFormatter`` 的区别：
    - 不依赖 contextvars 中的 request_id（K8s 场景由关联 ID 中间件管理）
    - 不处理 extra_fields（保持输出简洁）
    - 额外输出 process 字段（便于 K8s 多容器环境排查）
    """

    def format(self, record: logging.LogRecord) -> str:
        """将 LogRecord 格式化为单行 JSON 字符串

        Args:
            record: Python 标准 LogRecord 对象

        Returns:
            单行 JSON 格式的日志字符串
        """
        log_data: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.threadName,
            "process": record.process,
        }

        # 附加异常信息
        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False, default=str)


# ============================================================================
# 日志初始化
# ============================================================================


def _suppress_noisy_libraries() -> None:
    """抑制第三方库过于嘈杂的日志输出"""
    for lib_name, lib_level in _NOISY_LIBRARIES.items():
        lib_logger = logging.getLogger(lib_name)
        lib_logger.setLevel(lib_level)
        lib_logger.propagate = False


def setup_json_logging(level: str = "INFO") -> None:
    """初始化 K8s 环境的 JSON 结构化日志

    配置 root logger 仅输出 JSON 格式到 stdout，
    适合 K8s 容器标准输出被日志采集器采集的场景。

    此函数是 ``app.core.logging_config.setup_logging()`` 的轻量级替代方案，
    两者不应同时调用。

    Args:
        level: 全局日志级别，默认 "INFO"。
            支持标准 Python 日志级别：DEBUG, INFO, WARNING, ERROR, CRITICAL
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # 创建 JSON 格式化器
    json_formatter = JsonFormatter()

    # 配置 root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # 清除已有 Handler，避免重复添加
    root_logger.handlers.clear()

    # 仅添加 stdout Handler（K8s 场景不需要文件输出）
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(numeric_level)
    stdout_handler.setFormatter(json_formatter)
    root_logger.addHandler(stdout_handler)

    # 配置 audit logger（同样输出到 stdout）
    audit_logger = logging.getLogger("audit")
    audit_logger.setLevel(logging.INFO)
    audit_logger.handlers.clear()
    audit_logger.propagate = False

    audit_handler = logging.StreamHandler(sys.stdout)
    audit_handler.setLevel(logging.INFO)
    audit_handler.setFormatter(json_formatter)
    audit_logger.addHandler(audit_handler)

    # 抑制第三方库嘈杂日志
    _suppress_noisy_libraries()

    # 记录初始化完成
    root_logger.info(
        "K8s JSON 日志系统初始化完成：级别=%s",
        level.upper(),
    )


# ============================================================================
# 公开 API 声明
# ============================================================================

__all__ = [
    "JsonFormatter",
    "setup_json_logging",
]
