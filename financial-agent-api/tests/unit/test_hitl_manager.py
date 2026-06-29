"""HITLManager 单元测试

覆盖：
- 创建审批请求（create_approval）
- 审批通过（resolve_approval - approve）
- 审批拒绝（resolve_approval - reject）
- 获取待审批列表（get_pending_approvals）
- 高风险工具判断（is_high_risk_tool）
- 超时清理（cleanup_expired）
- 并发安全（concurrent_access）
- 异常场景：非高风险工具、不存在的审批、已处理的审批
"""

import threading
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from app.agent.hitl_manager import HITLManager
from app.core.enums import HITLStatus
from app.models.schemas import HITLApprovalResponse, HITLApprovalResult


class TestHITLManager:
    """HITLManager 核心功能测试"""

    @pytest.fixture
    def hitl(self) -> HITLManager:
        """创建 HITLManager 实例，注册高风险工具"""
        with patch("app.agent.hitl_manager.HITL_APPROVAL_COUNT", new=self._mock_counter()), \
             patch("app.agent.hitl_manager.HITL_APPROVAL_PENDING", new=self._mock_gauge()):
            return HITLManager(
                high_risk_tools=["send_email_notification", "delete_data"],
                approval_timeout=300,
            )

    @staticmethod
    def _mock_counter() -> object:
        """创建 mock Counter"""
        mock = patch("app.observability.metrics._NoopMetric").start().return_value
        return mock

    @staticmethod
    def _mock_gauge() -> object:
        """创建 mock Gauge"""
        mock = patch("app.observability.metrics._NoopMetric").start().return_value
        return mock

    @pytest.fixture(autouse=True)
    def _patch_metrics(self) -> None:
        """自动 patch 所有 Prometheus 指标为 no-op"""
        with patch("app.agent.hitl_manager.HITL_APPROVAL_COUNT") as mock_counter, \
             patch("app.agent.hitl_manager.HITL_APPROVAL_PENDING") as mock_gauge:
            self._mock_counter_obj = mock_counter
            self._mock_gauge_obj = mock_gauge
            yield

    # =====================================================================
    # create_approval 测试
    # =====================================================================

    def test_create_approval_success(self, hitl: HITLManager) -> None:
        """创建审批请求成功"""
        response = hitl.create_approval(
            tool_name="send_email_notification",
            tool_args={"to": "test@company.com"},
            session_id="session-1",
            thread_id="thread-1",
        )
        assert isinstance(response, HITLApprovalResponse)
        assert response.approval_id
        assert response.status == HITLStatus.PENDING
        assert response.tool_name == "send_email_notification"
        assert response.tool_args == {"to": "test@company.com"}
        assert response.created_at

    def test_create_approval_non_high_risk_tool(self, hitl: HITLManager) -> None:
        """非高风险工具创建审批时抛出 ValueError"""
        with pytest.raises(ValueError, match="不在高风险工具列表中"):
            hitl.create_approval(
                tool_name="search_internal_documents",
                tool_args={"query": "测试"},
                session_id="session-1",
                thread_id="thread-1",
            )

    def test_create_approval_generates_unique_id(self, hitl: HITLManager) -> None:
        """每次创建审批生成唯一 ID"""
        r1 = hitl.create_approval(
            tool_name="send_email_notification",
            tool_args={"to": "a@company.com"},
            session_id="s1",
            thread_id="t1",
        )
        r2 = hitl.create_approval(
            tool_name="send_email_notification",
            tool_args={"to": "b@company.com"},
            session_id="s2",
            thread_id="t2",
        )
        assert r1.approval_id != r2.approval_id

    # =====================================================================
    # resolve_approval 测试
    # =====================================================================

    def test_resolve_approval_approve(self, hitl: HITLManager) -> None:
        """审批通过"""
        response = hitl.create_approval(
            tool_name="send_email_notification",
            tool_args={"to": "test@company.com"},
            session_id="s1",
            thread_id="t1",
        )
        result = hitl.resolve_approval(response.approval_id, action="approve")
        assert isinstance(result, HITLApprovalResult)
        assert result.status == HITLStatus.APPROVED
        assert result.approval_id == response.approval_id
        assert result.resolved_at

    def test_resolve_approval_reject(self, hitl: HITLManager) -> None:
        """审批拒绝"""
        response = hitl.create_approval(
            tool_name="send_email_notification",
            tool_args={"to": "test@company.com"},
            session_id="s1",
            thread_id="t1",
        )
        result = hitl.resolve_approval(
            response.approval_id, action="reject", reason="信息不完整"
        )
        assert result.status == HITLStatus.REJECTED
        assert result.reason == "信息不完整"

    def test_resolve_approval_nonexistent(self, hitl: HITLManager) -> None:
        """处理不存在的审批时抛出 KeyError"""
        with pytest.raises(KeyError, match="审批请求不存在"):
            hitl.resolve_approval("nonexistent-id", action="approve")

    def test_resolve_approval_already_resolved(self, hitl: HITLManager) -> None:
        """重复处理已审批的请求时抛出 ValueError"""
        response = hitl.create_approval(
            tool_name="send_email_notification",
            tool_args={"to": "test@company.com"},
            session_id="s1",
            thread_id="t1",
        )
        hitl.resolve_approval(response.approval_id, action="approve")
        with pytest.raises(ValueError, match="审批已处理"):
            hitl.resolve_approval(response.approval_id, action="reject")

    # =====================================================================
    # get_pending_approvals 测试
    # =====================================================================

    def test_get_pending_approvals_empty(self, hitl: HITLManager) -> None:
        """无待审批项时返回空列表"""
        pending = hitl.get_pending_approvals()
        assert pending == []

    def test_get_pending_approvals_with_items(self, hitl: HITLManager) -> None:
        """有待审批项时返回正确列表"""
        r1 = hitl.create_approval(
            tool_name="send_email_notification",
            tool_args={"to": "a@company.com"},
            session_id="s1",
            thread_id="t1",
        )
        hitl.create_approval(
            tool_name="delete_data",
            tool_args={"table": "users"},
            session_id="s2",
            thread_id="t2",
        )
        pending = hitl.get_pending_approvals()
        assert len(pending) == 2
        assert all(p.status == HITLStatus.PENDING for p in pending)

    def test_get_pending_approvals_after_resolve(self, hitl: HITLManager) -> None:
        """审批后待审批列表更新"""
        r1 = hitl.create_approval(
            tool_name="send_email_notification",
            tool_args={"to": "a@company.com"},
            session_id="s1",
            thread_id="t1",
        )
        hitl.resolve_approval(r1.approval_id, action="approve")
        pending = hitl.get_pending_approvals()
        assert len(pending) == 0

    # =====================================================================
    # is_high_risk_tool 测试
    # =====================================================================

    def test_is_high_risk_tool_true(self, hitl: HITLManager) -> None:
        """高风险工具返回 True"""
        assert hitl.is_high_risk_tool("send_email_notification") is True
        assert hitl.is_high_risk_tool("delete_data") is True

    def test_is_high_risk_tool_false(self, hitl: HITLManager) -> None:
        """非高风险工具返回 False"""
        assert hitl.is_high_risk_tool("search_internal_documents") is False
        assert hitl.is_high_risk_tool("get_employee_info") is False

    # =====================================================================
    # get_approval 测试
    # =====================================================================

    def test_get_approval_exists(self, hitl: HITLManager) -> None:
        """获取存在的审批"""
        response = hitl.create_approval(
            tool_name="send_email_notification",
            tool_args={"to": "test@company.com"},
            session_id="s1",
            thread_id="t1",
        )
        found = hitl.get_approval(response.approval_id)
        assert found is not None
        assert found.approval_id == response.approval_id

    def test_get_approval_not_exists(self, hitl: HITLManager) -> None:
        """获取不存在的审批返回 None"""
        found = hitl.get_approval("nonexistent-id")
        assert found is None

    # =====================================================================
    # cleanup_expired 测试
    # =====================================================================

    def test_cleanup_expired_no_expired(self, hitl: HITLManager) -> None:
        """无过期审批时返回 0"""
        hitl.create_approval(
            tool_name="send_email_notification",
            tool_args={"to": "test@company.com"},
            session_id="s1",
            thread_id="t1",
        )
        expired = hitl.cleanup_expired()
        assert expired == 0

    def test_cleanup_expired_with_expired(self) -> None:
        """有过期审批时正确清理"""
        with patch("app.agent.hitl_manager.HITL_APPROVAL_COUNT") as mock_counter, \
             patch("app.agent.hitl_manager.HITL_APPROVAL_PENDING") as mock_gauge:
            hitl = HITLManager(
                high_risk_tools=["send_email_notification"],
                approval_timeout=1,  # 1 秒超时
            )
        response = hitl.create_approval(
            tool_name="send_email_notification",
            tool_args={"to": "test@company.com"},
            session_id="s1",
            thread_id="t1",
        )
        # 手动将 created_at 设为过去时间
        past_time = (datetime.now(UTC) - timedelta(seconds=10)).isoformat()
        with hitl._lock:
            hitl._approvals[response.approval_id].created_at = past_time

        expired = hitl.cleanup_expired()
        assert expired == 1

        # 验证状态已更新
        approval = hitl.get_approval(response.approval_id)
        assert approval is not None
        assert approval.status == HITLStatus.EXPIRED

    def test_cleanup_expired_preserves_active(self) -> None:
        """清理过期审批不影响活跃审批"""
        with patch("app.agent.hitl_manager.HITL_APPROVAL_COUNT") as mock_counter, \
             patch("app.agent.hitl_manager.HITL_APPROVAL_PENDING") as mock_gauge:
            hitl = HITLManager(
                high_risk_tools=["send_email_notification"],
                approval_timeout=300,
            )
        # 创建一个活跃审批
        active = hitl.create_approval(
            tool_name="send_email_notification",
            tool_args={"to": "active@company.com"},
            session_id="s1",
            thread_id="t1",
        )
        expired = hitl.cleanup_expired()
        assert expired == 0

        # 活跃审批仍为 PENDING
        found = hitl.get_approval(active.approval_id)
        assert found is not None
        assert found.status == HITLStatus.PENDING

    # =====================================================================
    # concurrent_access 测试
    # =====================================================================

    def test_concurrent_create_and_resolve(self) -> None:
        """并发创建和审批的线程安全性"""
        with patch("app.agent.hitl_manager.HITL_APPROVAL_COUNT") as mock_counter, \
             patch("app.agent.hitl_manager.HITL_APPROVAL_PENDING") as mock_gauge:
            hitl = HITLManager(
                high_risk_tools=["send_email_notification"],
                approval_timeout=300,
            )

        errors: list[Exception] = []
        approval_ids: list[str] = []

        def create_approvals() -> None:
            try:
                for i in range(10):
                    r = hitl.create_approval(
                        tool_name="send_email_notification",
                        tool_args={"to": f"test{i}@company.com"},
                        session_id=f"s{i}",
                        thread_id=f"t{i}",
                    )
                    approval_ids.append(r.approval_id)
            except Exception as exc:
                errors.append(exc)

        def resolve_approvals() -> None:
            try:
                for _ in range(20):
                    pending = hitl.get_pending_approvals()
                    if pending:
                        hitl.resolve_approval(pending[0].approval_id, action="approve")
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=create_approvals)
        t2 = threading.Thread(target=resolve_approvals)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert len(errors) == 0, f"并发错误：{errors}"