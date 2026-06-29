"""企业级审计日志模块

为 financial-agent-api 提供结构化的审计事件记录能力。
审计事件覆盖 API 调用、LLM 交互、安全检测等关键操作，
输出到专用的 ``audit`` logger，与业务日志隔离。

使用方式：
    from app.security.audit import AuditEvent, log_audit_event

    event = AuditEvent(
        event_type="api_request",
        user_hashed_key="a1b2c3...",
        user_role="admin",
        endpoint="/api/v1/chat",
        method="POST",
        status_code=200,
        duration_ms=150.3,
    )
    log_audit_event(event)
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("audit")


# ============================================================================
# 审计事件类
# ============================================================================


class AuditEvent:
    """审计事件数据载体

    封装一次请求或操作的完整审计信息，包括用户身份、请求详情、
    LLM 调用指标、安全检测报告等。

    Attributes:
        event_id: 事件唯一标识符（UUID v4）
        timestamp: 事件发生时间（UTC，ISO 8601）
        event_type: 事件类型（如 api_request、llm_call、tool_call）
        user_hashed_key: 用户 API Key 的哈希值
        user_role: 用户角色
        endpoint: 请求端点路径
        method: HTTP 请求方法
        status_code: HTTP 响应状态码
        duration_ms: 请求处理耗时（毫秒）
        session_id: 会话 ID
        tool_name: 工具名称（工具调用事件时填充）
        llm_provider: LLM 供应商名称
        llm_model: LLM 模型名称
        token_input: 输入 Token 数量
        token_output: 输出 Token 数量
        pii_report: PII 检测报告
        injection_report: Prompt 注入检测报告
        client_ip: 客户端 IP 地址
        error_message: 错误信息（异常时填充）
    """

    def __init__(
        self,
        event_type: str,
        user_hashed_key: str | None = None,
        user_role: str | None = None,
        endpoint: str | None = None,
        method: str | None = None,
        status_code: int | None = None,
        duration_ms: float | None = None,
        session_id: str | None = None,
        tool_name: str | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        token_input: int | None = None,
        token_output: int | None = None,
        pii_report: dict[str, Any] | None = None,
        injection_report: dict[str, Any] | None = None,
        client_ip: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """初始化审计事件

        Args:
            event_type: 事件类型标识
            user_hashed_key: 用户 API Key 哈希值
            user_role: 用户角色
            endpoint: 请求端点路径
            method: HTTP 请求方法
            status_code: HTTP 响应状态码
            duration_ms: 请求处理耗时（毫秒）
            session_id: 会话 ID
            tool_name: 工具名称
            llm_provider: LLM 供应商名称
            llm_model: LLM 模型名称
            token_input: 输入 Token 数量
            token_output: 输出 Token 数量
            pii_report: PII 检测报告
            injection_report: Prompt 注入检测报告
            client_ip: 客户端 IP 地址
            error_message: 错误信息
        """
        self.event_id: str = str(uuid.uuid4())
        self.timestamp: str = datetime.now(tz=timezone.utc).isoformat()
        self.event_type: str = event_type
        self.user_hashed_key: str | None = user_hashed_key
        self.user_role: str | None = user_role
        self.endpoint: str | None = endpoint
        self.method: str | None = method
        self.status_code: int | None = status_code
        self.duration_ms: float | None = duration_ms
        self.session_id: str | None = session_id
        self.tool_name: str | None = tool_name
        self.llm_provider: str | None = llm_provider
        self.llm_model: str | None = llm_model
        self.token_input: int | None = token_input
        self.token_output: int | None = token_output
        self.pii_report: dict[str, Any] | None = pii_report
        self.injection_report: dict[str, Any] | None = injection_report
        self.client_ip: str | None = client_ip
        self.error_message: str | None = error_message

    def to_dict(self) -> dict[str, Any]:
        """将审计事件转换为字典，过滤值为 None 的字段

        Returns:
            仅包含非 None 字段的字典
        """
        data: dict[str, Any] = {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
        }

        optional_fields: list[tuple[str, Any]] = [
            ("user_hashed_key", self.user_hashed_key),
            ("user_role", self.user_role),
            ("endpoint", self.endpoint),
            ("method", self.method),
            ("status_code", self.status_code),
            ("duration_ms", self.duration_ms),
            ("session_id", self.session_id),
            ("tool_name", self.tool_name),
            ("llm_provider", self.llm_provider),
            ("llm_model", self.llm_model),
            ("token_input", self.token_input),
            ("token_output", self.token_output),
            ("pii_report", self.pii_report),
            ("injection_report", self.injection_report),
            ("client_ip", self.client_ip),
            ("error_message", self.error_message),
        ]

        for field_name, field_value in optional_fields:
            if field_value is not None:
                data[field_name] = field_value

        return data

    def to_json(self) -> str:
        """将审计事件序列化为 JSON 字符串

        Returns:
            JSON 格式的审计事件字符串
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


# ============================================================================
# 审计日志输出
# ============================================================================


def log_audit_event(event: AuditEvent) -> None:
    """将审计事件输出到 audit logger

    使用 ``logging.getLogger("audit")`` 输出，与业务日志隔离。
    在 ``app.core.logging_config.setup_logging()`` 中，audit logger
    被配置为写入独立的 ``audit.log`` 文件。

    Args:
        event: 待记录的审计事件实例
    """
    logger.info(
        "审计事件: type=%s, event_id=%s",
        event.event_type,
        event.event_id,
        extra={"extra_fields": event.to_dict()},
    )


# ============================================================================
# 公开 API 声明
# ============================================================================

__all__ = [
    "AuditEvent",
    "log_audit_event",
]
