"""Human-in-the-Loop 审批管理器模块

管理高风险工具调用的人工审批流程，支持：
- 创建审批请求并生成唯一 approval_id
- 处理审批（approve/reject）
- 超时自动过期
- 线程安全的内存存储
"""

import threading
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from app.core.enums import HITLStatus
from app.core.logging_config import get_logger
from app.models.schemas import (
    HITLApprovalResponse,
    HITLApprovalResult,
)
from app.observability.metrics import HITL_APPROVAL_COUNT, HITL_APPROVAL_PENDING

logger = get_logger(__name__)

__all__ = ["HITLManager"]


class HITLManager:
    """Human-in-the-Loop 审批管理器

    为高风险工具调用提供人工审批机制，确保关键操作在执行前获得
    人类操作员的确认。所有审批状态变更通过 threading.Lock 保证
    线程安全。

    Attributes:
        _high_risk_tools: 高风险工具名称集合
        _approval_timeout: 审批超时时间（秒）
        _approvals: 审批记录内存存储
        _lock: 线程安全锁
    """

    def __init__(self, high_risk_tools: list[str], approval_timeout: int = 300) -> None:
        """初始化 HITL 审批管理器

        Args:
            high_risk_tools: 高风险工具名称列表
            approval_timeout: 审批超时时间（秒），默认 300 秒
        """
        self._high_risk_tools: set[str] = set(high_risk_tools)
        self._approval_timeout: int = approval_timeout
        self._approvals: dict[str, HITLApprovalResponse] = {}
        self._lock: threading.Lock = threading.Lock()
        logger.info(
            "HITLManager 初始化完成，高风险工具：%s，超时：%d秒",
            high_risk_tools,
            approval_timeout,
        )

    def create_approval(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        session_id: str,
        thread_id: str,
    ) -> HITLApprovalResponse:
        """创建审批请求

        为高风险工具调用创建一条待审批记录，生成唯一 approval_id，
        并将状态设为 PENDING。调用后 HITL_APPROVAL_PENDING 指标递增。

        Args:
            tool_name: 工具名称
            tool_args: 工具调用参数
            session_id: 会话 ID
            thread_id: LangGraph 线程 ID

        Returns:
            HITLApprovalResponse 审批响应对象

        Raises:
            ValueError: 当工具不在高风险列表中时
        """
        if tool_name not in self._high_risk_tools:
            raise ValueError(f"工具 '{tool_name}' 不在高风险工具列表中")

        approval_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()

        response = HITLApprovalResponse(
            approval_id=approval_id,
            status=HITLStatus.PENDING,
            tool_name=tool_name,
            tool_args=tool_args,
            created_at=now,
        )

        with self._lock:
            self._approvals[approval_id] = response
            HITL_APPROVAL_PENDING.inc()

        HITL_APPROVAL_COUNT.labels(tool_name=tool_name, action="created").inc()
        logger.info("创建审批请求：%s，工具：%s", approval_id, tool_name)

        return response

    def resolve_approval(
        self,
        approval_id: str,
        action: Literal["approve", "reject"],
        reason: str | None = None,
    ) -> HITLApprovalResult:
        """处理审批

        对指定的审批请求执行 approve 或 reject 操作，更新审批状态，
        并递减 HITL_APPROVAL_PENDING 指标。

        Args:
            approval_id: 审批唯一标识
            action: 审批动作，"approve" 或 "reject"
            reason: 审批理由（可选）

        Returns:
            HITLApprovalResult 审批结果对象

        Raises:
            KeyError: 当审批请求不存在时
            ValueError: 当审批已处理（非 PENDING 状态）时
        """
        with self._lock:
            approval = self._approvals.get(approval_id)
            if approval is None:
                raise KeyError(f"审批请求不存在：{approval_id}")
            if approval.status != HITLStatus.PENDING:
                raise ValueError(f"审批已处理，当前状态：{approval.status}")

            new_status = HITLStatus.APPROVED if action == "approve" else HITLStatus.REJECTED
            approval.status = new_status

            HITL_APPROVAL_PENDING.dec()
            HITL_APPROVAL_COUNT.labels(tool_name=approval.tool_name, action=action).inc()

        now = datetime.now(UTC).isoformat()
        result = HITLApprovalResult(
            approval_id=approval_id,
            status=new_status,
            resolved_at=now,
            reason=reason,
        )

        logger.info("审批已处理：%s，动作：%s", approval_id, action)
        return result

    def get_pending_approvals(self) -> list[HITLApprovalResponse]:
        """获取所有待审批项

        Returns:
            状态为 PENDING 的审批响应列表
        """
        with self._lock:
            return [a for a in self._approvals.values() if a.status == HITLStatus.PENDING]

    def get_approval(self, approval_id: str) -> HITLApprovalResponse | None:
        """获取指定审批项

        Args:
            approval_id: 审批唯一标识

        Returns:
            审批响应对象，不存在时返回 None
        """
        with self._lock:
            return self._approvals.get(approval_id)

    def is_high_risk_tool(self, tool_name: str) -> bool:
        """判断工具是否属于高风险列表

        Args:
            tool_name: 工具名称

        Returns:
            是否为高风险工具
        """
        return tool_name in self._high_risk_tools

    def cleanup_expired(self) -> int:
        """清理超时的审批请求

        遍历所有 PENDING 状态的审批记录，将超过 _approval_timeout
        的记录标记为 EXPIRED，并递减 HITL_APPROVAL_PENDING 指标。

        Returns:
            过期的审批数量
        """
        now = datetime.now(UTC)
        expired_count = 0

        with self._lock:
            for approval in self._approvals.values():
                if approval.status != HITLStatus.PENDING:
                    continue
                try:
                    created = datetime.fromisoformat(approval.created_at)
                except (ValueError, TypeError):
                    logger.warning(
                        "审批记录时间格式异常，跳过：%s，created_at=%s",
                        approval.approval_id,
                        approval.created_at,
                    )
                    continue
                if (now - created).total_seconds() > self._approval_timeout:
                    approval.status = HITLStatus.EXPIRED
                    HITL_APPROVAL_PENDING.dec()
                    expired_count += 1

        if expired_count > 0:
            logger.info("清理了 %d 个超时审批请求", expired_count)

        return expired_count
