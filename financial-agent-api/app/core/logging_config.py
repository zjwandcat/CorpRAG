"""结构化日志配置模块

为企业内部办公知识库智能体系统提供完善的日志基础设施，包括：
- 文件日志输出（app.log / error.log / audit.log）
- 日志轮转（RotatingFileHandler，单文件 10MB，保留 5 个备份）
- JSON 结构化日志（方便 ELK/Grafana Loki 采集）
- 请求追踪 ID（基于 contextvars，异步安全）
- 业务辅助日志函数（RAG 检索、Agent 步骤、函数调用）

使用方式：
    from app.core.logging_config import setup_logging, get_logger

    # 应用启动时初始化
    setup_logging()

    # 获取 logger
    logger = get_logger(__name__)
    logger.info("服务启动")
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Sequence

# ============================================================================
# 常量定义
# ============================================================================

# 日志文件目录（相对于应用启动目录）
_LOG_DIR: Path = Path("logs")

# 日志文件名
_APP_LOG_FILE: str = "app.log"
_ERROR_LOG_FILE: str = "error.log"
_AUDIT_LOG_FILE: str = "audit.log"

# 日志轮转参数
_MAX_BYTES: int = 10 * 1024 * 1024  # 10MB
_BACKUP_COUNT: int = 5
_LOG_ENCODING: str = "utf-8"

# 控制台日志格式
_CONSOLE_FORMAT: str = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_CONSOLE_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# 需要抑制日志的第三方库及其级别
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
# 请求追踪 ID（基于 contextvars，异步安全）
# ============================================================================

_request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def set_request_id(request_id: str) -> None:
    """设置当前请求的追踪 ID

    在请求中间件中调用，将 request_id 绑定到当前异步上下文，
    后续所有日志输出将自动携带该 ID。

    Args:
        request_id: 请求唯一标识符
    """
    _request_id_var.set(request_id)


def get_request_id() -> str | None:
    """获取当前请求的追踪 ID

    Returns:
        当前上下文中的 request_id，未设置时返回 None
    """
    return _request_id_var.get()


# ============================================================================
# JSON 结构化日志格式化器
# ============================================================================


class JsonFormatter(logging.Formatter):
    """JSON 结构化日志格式化器

    将日志记录格式化为 JSON 字符串，包含以下字段：
    - timestamp: ISO 8601 格式的时间戳（含时区）
    - level: 日志级别
    - logger: logger 名称
    - message: 日志消息
    - module: 模块名
    - function: 函数名
    - line: 行号
    - thread: 线程名
    - request_id: 请求追踪 ID（可选，仅在上下文中设置时包含）
    """

    def format(self, record: logging.LogRecord) -> str:
        """将 LogRecord 格式化为 JSON 字符串"""
        log_data: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.threadName,
        }

        # 附加请求追踪 ID（如果存在）
        request_id = get_request_id()
        if request_id is not None:
            log_data["request_id"] = request_id

        # 附加异常信息
        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = self.formatException(record.exc_info)

        # 附加额外字段（通过 logger.info("msg", extra={...}) 传入）
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            log_data.update(record.extra_fields)

        return json.dumps(log_data, ensure_ascii=False, default=str)


# ============================================================================
# 请求追踪日志过滤器
# ============================================================================


class RequestIdFilter(logging.Filter):
    """请求追踪 ID 日志过滤器

    将 contextvars 中的 request_id 注入到 LogRecord 中，
    使控制台格式也能通过 %(request_id)s 引用。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """注入 request_id 到 LogRecord"""
        record.request_id = get_request_id() or "-"  # type: ignore[attr-defined]
        return True


# ============================================================================
# 日志初始化
# ============================================================================


def _ensure_log_dir() -> None:
    """确保日志目录存在，不存在则自动创建"""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def _create_file_handler(
    filename: str,
    level: int,
    formatter: logging.Formatter,
) -> RotatingFileHandler:
    """创建带轮转的文件日志 Handler

    Args:
        filename: 日志文件名（不含目录路径）
        level: 日志级别
        formatter: 日志格式化器

    Returns:
        配置好的 RotatingFileHandler 实例
    """
    file_path = _LOG_DIR / filename
    handler = RotatingFileHandler(
        filename=str(file_path),
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding=_LOG_ENCODING,
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())
    return handler


def _create_console_handler(
    level: int,
    formatter: logging.Formatter,
) -> logging.StreamHandler:
    """创建控制台日志 Handler

    Args:
        level: 日志级别
        formatter: 日志格式化器

    Returns:
        配置好的 StreamHandler 实例
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())
    return handler


def _suppress_noisy_libraries() -> None:
    """抑制第三方库过于嘈杂的日志输出"""
    for lib_name, lib_level in _NOISY_LIBRARIES.items():
        lib_logger = logging.getLogger(lib_name)
        lib_logger.setLevel(lib_level)
        # 禁止向上传播，避免 root logger 重复输出
        lib_logger.propagate = False


def setup_logging(level: int = logging.INFO) -> None:
    """一键初始化所有日志配置

    配置 root logger、audit logger 和第三方库日志级别。
    应在应用启动时调用一次。

    Args:
        level: 全局日志级别，默认 INFO
    """
    # 确保日志目录存在
    _ensure_log_dir()

    # ---- 格式化器 ----
    # 控制台：人类可读格式
    console_formatter = logging.Formatter(
        fmt=_CONSOLE_FORMAT,
        datefmt=_CONSOLE_DATE_FORMAT,
    )
    # 文件：JSON 结构化格式
    json_formatter = JsonFormatter()

    # ---- Root Logger 配置 ----
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 清除已有 Handler，避免重复添加（热重载场景）
    root_logger.handlers.clear()

    # 控制台 Handler
    root_logger.addHandler(_create_console_handler(level, console_formatter))

    # app.log - 记录 INFO 及以上级别
    root_logger.addHandler(_create_file_handler(_APP_LOG_FILE, level, json_formatter))

    # error.log - 仅记录 ERROR 及以上级别
    root_logger.addHandler(_create_file_handler(_ERROR_LOG_FILE, logging.ERROR, json_formatter))

    # ---- Audit Logger 配置 ----
    audit_logger = logging.getLogger("audit")
    audit_logger.setLevel(logging.INFO)
    # 清除已有 Handler，避免重复添加
    audit_logger.handlers.clear()
    # 禁止向上传播到 root logger，避免 audit 日志在 app.log 中重复
    audit_logger.propagate = False

    # audit.log - 审计日志专用文件
    audit_logger.addHandler(_create_file_handler(_AUDIT_LOG_FILE, logging.INFO, json_formatter))
    # 审计日志也输出到控制台
    audit_logger.addHandler(_create_console_handler(logging.INFO, console_formatter))

    # ---- 抑制第三方库嘈杂日志 ----
    _suppress_noisy_libraries()

    # 记录初始化完成
    root_logger.info(
        "日志系统初始化完成：级别=%s，目录=%s",
        logging.getLevelName(level),
        _LOG_DIR.resolve(),
    )


# ============================================================================
# 辅助函数
# ============================================================================


def get_logger(name: str) -> logging.Logger:
    """获取带请求追踪的 logger

    与 logging.getLogger(name) 功能相同，但明确标记为项目标准获取方式，
    便于后续统一扩展（如自动注入模块上下文）。

    Args:
        name: logger 名称，通常使用 __name__

    Returns:
        配置好的 Logger 实例
    """
    return logging.getLogger(name)


def log_function_call(
    func_name: str,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    duration_ms: float | None = None,
    result_summary: str | None = None,
) -> None:
    """记录函数调用日志

    以结构化方式记录函数调用的关键信息，便于追踪执行链路。

    Args:
        func_name: 函数名称
        args: 位置参数（可选）
        kwargs: 关键字参数（可选）
        duration_ms: 执行耗时（毫秒，可选）
        result_summary: 结果摘要（可选）
    """
    logger = get_logger("app.function_call")
    extra: dict[str, Any] = {
        "func_name": func_name,
    }
    if args is not None:
        extra["args"] = str(args)
    if kwargs is not None:
        extra["kwargs"] = str(kwargs)
    if duration_ms is not None:
        extra["duration_ms"] = round(duration_ms, 2)
    if result_summary is not None:
        extra["result_summary"] = result_summary

    logger.info(
        "函数调用：%s",
        func_name,
        extra={"extra_fields": extra},
    )


def log_rag_query(
    query: str,
    top_k: int,
    hit_count: int,
    duration_ms: float,
    scores: Sequence[float] | None = None,
) -> None:
    """记录 RAG 检索日志

    以结构化方式记录 RAG 检索的关键指标，便于分析检索质量和性能。

    Args:
        query: 用户查询文本
        top_k: 检索 top-k 参数
        hit_count: 实际命中数量
        duration_ms: 检索耗时（毫秒）
        scores: 相似度分数列表（可选）
    """
    logger = get_logger("app.rag_query")
    extra: dict[str, Any] = {
        "query": query[:200],  # 截断过长的查询
        "top_k": top_k,
        "hit_count": hit_count,
        "duration_ms": round(duration_ms, 2),
    }
    if scores is not None:
        extra["scores"] = [round(s, 4) for s in scores]

    logger.info(
        "RAG 检索：query=%s, top_k=%d, hit_count=%d, 耗时=%.1fms",
        query[:50],
        top_k,
        hit_count,
        duration_ms,
        extra={"extra_fields": extra},
    )


def log_agent_step(
    step_name: str,
    tool_name: str | None = None,
    duration_ms: float | None = None,
    status: str = "success",
) -> None:
    """记录 Agent 步骤日志

    以结构化方式记录 Agent 执行的每个步骤，便于追踪 Agent 执行链路。

    Args:
        step_name: 步骤名称（如 agent_node, retrieval_node, tool_node）
        tool_name: 工具名称（可选，仅在工具调用步骤时有值）
        duration_ms: 步骤耗时（毫秒，可选）
        status: 步骤状态，默认 "success"，可选 "error" / "timeout"
    """
    logger = get_logger("app.agent_step")
    extra: dict[str, Any] = {
        "step_name": step_name,
        "status": status,
    }
    if tool_name is not None:
        extra["tool_name"] = tool_name
    if duration_ms is not None:
        extra["duration_ms"] = round(duration_ms, 2)

    log_level = logging.ERROR if status == "error" else logging.INFO
    logger.log(
        log_level,
        "Agent 步骤：%s, 工具=%s, 状态=%s, 耗时=%s",
        step_name,
        tool_name or "-",
        status,
        f"{duration_ms:.1f}ms" if duration_ms is not None else "-",
        extra={"extra_fields": extra},
    )


# ============================================================================
# 公开 API 声明
# ============================================================================

__all__ = [
    "JsonFormatter",
    "RequestIdFilter",
    "get_logger",
    "get_request_id",
    "log_agent_step",
    "log_function_call",
    "log_rag_query",
    "set_request_id",
    "setup_logging",
]
