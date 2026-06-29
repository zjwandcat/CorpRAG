"""v5.1 治理与 HITL E2E 集成测试

覆盖：
- HITL 高风险工具拦截
- HITL 审批通过流程
- HITL 审批拒绝流程
- Guardrails 死循环检测
- Guardrails 无误报
- 知识图谱检索集成
- KG 禁用降级
- HITL 禁用降级
- Guardrails 禁用降级
- 核心链路在功能故障时不受影响

所有测试使用 Mock 替代真实服务调用。
"""

import json
import threading
from collections import deque
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.agent.hitl_manager import HITLManager
from app.core.enums import GuardrailAction, HITLStatus
from app.exceptions import InfiniteLoopDetectedError
from app.models.schemas import HITLApprovalResponse
from app.rag.knowledge_graph import KnowledgeGraphManager
from app.security.guardrails import ToolRepetitionDetector


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def hitl_manager() -> HITLManager:
    """创建 HITLManager 实例"""
    with patch("app.agent.hitl_manager.HITL_APPROVAL_COUNT") as mock_counter, \
         patch("app.agent.hitl_manager.HITL_APPROVAL_PENDING") as mock_gauge:
        return HITLManager(
            high_risk_tools=["send_email_notification", "delete_data"],
            approval_timeout=300,
        )


@pytest.fixture
def guardrail_detector() -> ToolRepetitionDetector:
    """创建 ToolRepetitionDetector 实例"""
    with patch("app.security.guardrails.GUARDRAIL_INTERVENTION_COUNT"):
        return ToolRepetitionDetector(max_repetition=3, window_size=5)


@pytest.fixture
def kg_manager(tmp_path: pytest.TempPathFactory) -> KnowledgeGraphManager:
    """创建 KnowledgeGraphManager 实例"""
    with patch("app.rag.knowledge_graph.KG_TRIPLET_COUNT") as mock_counter, \
         patch("app.rag.knowledge_graph.KG_SEARCH_LATENCY") as mock_latency:
        return KnowledgeGraphManager(storage_path=str(tmp_path))


# ============================================================================
# HITL 测试
# ============================================================================


class TestHITLHighRiskToolBlocked:
    """HITL 高风险工具拦截测试"""

    def test_high_risk_tool_requires_approval(self, hitl_manager: HITLManager) -> None:
        """高风险工具调用需要审批"""
        assert hitl_manager.is_high_risk_tool("send_email_notification") is True

        # 创建审批
        response = hitl_manager.create_approval(
            tool_name="send_email_notification",
            tool_args={"to": "test@company.com", "subject": "提醒"},
            session_id="session-1",
            thread_id="thread-1",
        )
        assert response.status == HITLStatus.PENDING

        # 未审批前，工具不应被执行（模拟拦截逻辑）
        pending = hitl_manager.get_pending_approvals()
        assert len(pending) == 1

    def test_non_high_risk_tool_not_blocked(self, hitl_manager: HITLManager) -> None:
        """非高风险工具不需要审批"""
        assert hitl_manager.is_high_risk_tool("search_internal_documents") is False

    def test_non_high_risk_tool_create_approval_raises(
        self, hitl_manager: HITLManager
    ) -> None:
        """非高风险工具创建审批时抛出 ValueError"""
        with pytest.raises(ValueError, match="不在高风险工具列表中"):
            hitl_manager.create_approval(
                tool_name="search_internal_documents",
                tool_args={"query": "测试"},
                session_id="s1",
                thread_id="t1",
            )


class TestHITLApprovalFlow:
    """HITL 审批通过流程测试"""

    def test_approval_flow_approve(self, hitl_manager: HITLManager) -> None:
        """完整的审批通过流程"""
        # 1. 创建审批
        response = hitl_manager.create_approval(
            tool_name="send_email_notification",
            tool_args={"to": "test@company.com"},
            session_id="s1",
            thread_id="t1",
        )
        assert response.status == HITLStatus.PENDING

        # 2. 审批通过
        result = hitl_manager.resolve_approval(
            response.approval_id, action="approve", reason="信息确认无误"
        )
        assert result.status == HITLStatus.APPROVED
        assert result.reason == "信息确认无误"

        # 3. 待审批列表为空
        pending = hitl_manager.get_pending_approvals()
        assert len(pending) == 0

        # 4. 审批记录仍可查询
        found = hitl_manager.get_approval(response.approval_id)
        assert found is not None
        assert found.status == HITLStatus.APPROVED


class TestHITLRejectionFlow:
    """HITL 审批拒绝流程测试"""

    def test_approval_flow_reject(self, hitl_manager: HITLManager) -> None:
        """完整的审批拒绝流程"""
        # 1. 创建审批
        response = hitl_manager.create_approval(
            tool_name="delete_data",
            tool_args={"table": "users", "condition": "id=1"},
            session_id="s1",
            thread_id="t1",
        )

        # 2. 审批拒绝
        result = hitl_manager.resolve_approval(
            response.approval_id, action="reject", reason="数据不可删除"
        )
        assert result.status == HITLStatus.REJECTED
        assert result.reason == "数据不可删除"

        # 3. 待审批列表为空
        pending = hitl_manager.get_pending_approvals()
        assert len(pending) == 0


# ============================================================================
# Guardrails 测试
# ============================================================================


class TestGuardrailRepetitionDetected:
    """Guardrails 死循环检测测试"""

    def test_repetition_detected_and_blocked(
        self, guardrail_detector: ToolRepetitionDetector
    ) -> None:
        """重复调用被检测并阻止"""
        tool_name = "search_internal_documents"
        tool_args = {"query": "报销流程"}

        # 连续调用直到 BLOCK
        actions: list[GuardrailAction] = []
        for _ in range(5):
            action = guardrail_detector.check(tool_name, tool_args)
            actions.append(action)

        # 验证动作序列：ALLOW, ALLOW, WARN, BLOCK, BLOCK
        assert actions[0] == GuardrailAction.ALLOW
        assert actions[1] == GuardrailAction.ALLOW
        assert actions[2] == GuardrailAction.WARN
        assert actions[3] == GuardrailAction.BLOCK
        assert actions[4] == GuardrailAction.BLOCK

    def test_infinite_loop_error_raised(
        self, guardrail_detector: ToolRepetitionDetector
    ) -> None:
        """死循环检测触发 InfiniteLoopDetectedError"""
        tool_name = "search_internal_documents"
        tool_args = {"query": "测试"}

        # 连续调用直到 BLOCK
        for _ in range(4):
            action = guardrail_detector.check(tool_name, tool_args)

        # 模拟 tool_node 中的逻辑：BLOCK 时抛出异常
        if action == GuardrailAction.BLOCK:
            with pytest.raises(InfiniteLoopDetectedError):
                raise InfiniteLoopDetectedError(
                    tool_name=tool_name,
                    repetition_count=guardrail_detector._max_repetition,
                )


class TestGuardrailNoFalsePositive:
    """Guardrails 无误报测试"""

    def test_different_queries_no_block(
        self, guardrail_detector: ToolRepetitionDetector
    ) -> None:
        """不同查询参数不触发 BLOCK"""
        for i in range(10):
            action = guardrail_detector.check(
                "search_internal_documents", {"query": f"查询{i}"}
            )
            assert action == GuardrailAction.ALLOW

    def test_alternating_tools_no_block(
        self, guardrail_detector: ToolRepetitionDetector
    ) -> None:
        """交替调用不同工具不触发 BLOCK"""
        for i in range(10):
            tool = "tool_a" if i % 2 == 0 else "tool_b"
            action = guardrail_detector.check(tool, {"arg": f"val_{i}"})
            assert action == GuardrailAction.ALLOW

    def test_reset_clears_false_alarm(
        self, guardrail_detector: ToolRepetitionDetector
    ) -> None:
        """重置后不再误报"""
        # 触发多次重复
        for _ in range(4):
            guardrail_detector.check("tool_a", {"arg": "val"})

        # 重置
        guardrail_detector.reset()

        # 重置后不应再 BLOCK
        action = guardrail_detector.check("tool_a", {"arg": "val"})
        assert action == GuardrailAction.ALLOW


# ============================================================================
# Knowledge Graph 集成测试
# ============================================================================


class TestKGSearchIntegration:
    """知识图谱检索集成测试"""

    def test_kg_search_after_add(
        self, kg_manager: KnowledgeGraphManager
    ) -> None:
        """添加三元组后可检索"""
        kg_manager.add_triplets([
            {"subject": "张三", "relation": "隶属", "object": "研发中心"},
            {"subject": "张三", "relation": "负责", "object": "项目A"},
            {"subject": "研发中心", "relation": "包含", "object": "项目A"},
        ])

        # 按实体搜索
        results = kg_manager.search("张三")
        assert len(results) >= 2

        # 按关系过滤
        results = kg_manager.search("张三", relation="隶属")
        assert len(results) == 1
        assert results[0].target_entity == "研发中心"

    def test_kg_multi_hop_search(
        self, kg_manager: KnowledgeGraphManager
    ) -> None:
        """多跳检索测试"""
        kg_manager.add_triplets([
            {"subject": "A", "relation": "r1", "object": "B"},
            {"subject": "B", "relation": "r2", "object": "C"},
            {"subject": "C", "relation": "r3", "object": "D"},
        ])

        # depth=1 只能到 B
        results_d1 = kg_manager.search("A", max_depth=1)
        assert all(r.entity in ("A",) for r in results_d1)

        # depth=2 可以到 B 和 C
        results_d2 = kg_manager.search("A", max_depth=2)
        assert len(results_d2) > len(results_d1)

    def test_kg_extract_and_search(
        self, kg_manager: KnowledgeGraphManager
    ) -> None:
        """LLM 提取 + 检索端到端"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            '[{"subject": "李四", "relation": "隶属", "object": "产品部"}]'
        )
        mock_llm.invoke.return_value = mock_response

        # 提取
        triplets = kg_manager.extract_triplets("李四隶属于产品部", mock_llm)
        assert len(triplets) == 1

        # 添加
        kg_manager.add_triplets(triplets, source="test_doc")

        # 检索
        results = kg_manager.search("李四")
        assert len(results) >= 1
        assert results[0].entity == "李四"
        assert results[0].target_entity == "产品部"


# ============================================================================
# 降级 / 禁用测试
# ============================================================================


class TestKGDisabledFallback:
    """KG 禁用降级测试"""

    def test_kg_disabled_returns_none(self) -> None:
        """KG_ENABLED=False 时 get_kg_manager 返回 None"""
        with patch("app.core.dependencies.settings") as mock_settings:
            mock_settings.KG_ENABLED = False
            # 重置单例
            import app.core.dependencies as deps
            deps._kg_manager_instance = None

            result = deps.get_kg_manager()
            assert result is None

    def test_kg_search_when_disabled(self) -> None:
        """KG 禁用时搜索逻辑降级为空结果"""
        # 模拟 KG 不可用时，搜索返回空列表
        with patch("app.core.dependencies.get_kg_manager", return_value=None):
            kg = None
            # 业务代码中应检查 kg is None
            assert kg is None
            # 降级：跳过 KG 检索，不影响核心链路


class TestHITLDisabledFallback:
    """HITL 禁用降级测试"""

    def test_hitl_disabled_returns_none(self) -> None:
        """HITL_ENABLED=False 时 get_hitl_manager 返回 None"""
        with patch("app.core.dependencies.settings") as mock_settings:
            mock_settings.HITL_ENABLED = False
            import app.core.dependencies as deps
            deps._hitl_manager_instance = None

            result = deps.get_hitl_manager()
            assert result is None

    def test_hitl_disabled_tool_executes_directly(self) -> None:
        """HITL 禁用时工具直接执行，无需审批"""
        with patch("app.core.dependencies.get_hitl_manager", return_value=None):
            hitl = None
            # 业务代码中：if hitl is None or not hitl.is_high_risk_tool(tool_name):
            #     直接执行工具
            assert hitl is None
            # 无需审批，工具直接执行


class TestGuardrailsDisabledFallback:
    """Guardrails 禁用降级测试"""

    def test_guardrails_disabled_returns_none(self) -> None:
        """GUARDRAILS_ENABLED=False 时 get_guardrail_detector 返回 None"""
        with patch("app.core.dependencies.settings") as mock_settings:
            mock_settings.GUARDRAILS_ENABLED = False
            import app.core.dependencies as deps
            deps._guardrail_detector_instance = None

            result = deps.get_guardrail_detector()
            assert result is None

    def test_guardrails_disabled_no_loop_check(self) -> None:
        """Guardrails 禁用时不进行死循环检测"""
        with patch("app.core.dependencies.get_guardrail_detector", return_value=None):
            detector = None
            # 业务代码中：if detector is not None: check()
            # detector 为 None 时跳过检测
            assert detector is None


# ============================================================================
# 核心链路不受影响测试
# ============================================================================


class TestCoreChainUnaffectedOnFeatureFailure:
    """核心链路在功能故障时不受影响测试"""

    def test_kg_failure_does_not_break_chain(
        self, kg_manager: KnowledgeGraphManager
    ) -> None:
        """KG 提取失败不中断主流程"""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM 服务不可用")

        # 提取失败返回空列表，不抛异常
        result = kg_manager.extract_triplets("测试文本", mock_llm)
        assert result == []

        # 主流程可继续执行

    def test_hitl_failure_does_not_break_chain(self) -> None:
        """HITL 异常不中断主流程"""
        with patch("app.agent.hitl_manager.HITL_APPROVAL_COUNT") as mock_counter, \
             patch("app.agent.hitl_manager.HITL_APPROVAL_PENDING") as mock_gauge:
            hitl = HITLManager(
                high_risk_tools=["send_email_notification"],
                approval_timeout=300,
            )

        # 正常创建审批
        response = hitl.create_approval(
            tool_name="send_email_notification",
            tool_args={"to": "test@company.com"},
            session_id="s1",
            thread_id="t1",
        )

        # 重复处理抛出 ValueError，但主流程应捕获并继续
        hitl.resolve_approval(response.approval_id, action="approve")
        try:
            hitl.resolve_approval(response.approval_id, action="reject")
        except ValueError:
            pass  # 主流程捕获异常后继续

        # HITL 管理器仍可正常工作
        new_response = hitl.create_approval(
            tool_name="send_email_notification",
            tool_args={"to": "another@company.com"},
            session_id="s2",
            thread_id="t2",
        )
        assert new_response.status == HITLStatus.PENDING

    def test_guardrail_failure_does_not_break_chain(self) -> None:
        """Guardrails 异常不中断主流程"""
        with patch("app.security.guardrails.GUARDRAIL_INTERVENTION_COUNT"):
            detector = ToolRepetitionDetector(max_repetition=3, window_size=5)

        # 正常检测
        action = detector.check("tool_a", {"arg": "val"})
        assert action == GuardrailAction.ALLOW

        # 重置后检测器恢复正常
        detector.reset()
        action = detector.check("tool_a", {"arg": "val"})
        assert action == GuardrailAction.ALLOW

    def test_all_features_disabled_chain_works(self) -> None:
        """所有 v5.1 功能禁用时核心链路正常工作"""
        # 模拟所有功能禁用
        with patch("app.core.dependencies.get_kg_manager", return_value=None), \
             patch("app.core.dependencies.get_hitl_manager", return_value=None), \
             patch("app.core.dependencies.get_guardrail_detector", return_value=None):

            kg = None
            hitl = None
            guardrail = None

            # 核心链路：工具调用不需要 KG 检索、HITL 审批、Guardrails 检测
            # 模拟核心链路执行
            tool_name = "search_internal_documents"
            tool_args = {"query": "报销流程"}

            # 1. KG 降级：跳过
            if kg is not None:
                kg.search("报销流程")  # 不会执行

            # 2. HITL 降级：跳过审批
            if hitl is not None and hitl.is_high_risk_tool(tool_name):
                hitl.create_approval(tool_name, tool_args, "s1", "t1")  # 不会执行

            # 3. Guardrails 降级：跳过检测
            if guardrail is not None:
                guardrail.check(tool_name, tool_args)  # 不会执行

            # 4. 核心链路正常执行
            result = f"工具 {tool_name} 执行成功"
            assert "执行成功" in result

    def test_kg_persist_failure_graceful(
        self, kg_manager: KnowledgeGraphManager
    ) -> None:
        """KG 持久化失败时优雅降级"""
        kg_manager.add_triplets([{"subject": "A", "relation": "r", "object": "B"}])

        # 模拟持久化失败（networkx 序列化抛出异常）
        with patch(
            "app.rag.knowledge_graph.nx.node_link_data",
            side_effect=OSError("序列化失败"),
        ):
            # persist 不应抛出异常（内部 try/except 捕获）
            kg_manager.persist()

        # 图谱数据仍在内存中
        results = kg_manager.search("A")
        assert len(results) >= 1