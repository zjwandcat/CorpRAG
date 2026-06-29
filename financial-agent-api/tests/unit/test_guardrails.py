"""ToolRepetitionDetector 单元测试

覆盖：
- 正常调用允许（allow_normal_calls）
- 接近阈值时警告（warn_repeated_calls）
- 超过阈值时阻止（block_excessive_repetition）
- 滑动窗口机制（window_sliding）
- 重置检测器（reset_detector）
- 不同参数不计为重复（different_args_not_repeated）
- 不同工具不计为重复
"""

from unittest.mock import patch

import pytest

from app.core.enums import GuardrailAction
from app.security.guardrails import ToolRepetitionDetector


class TestToolRepetitionDetector:
    """ToolRepetitionDetector 核心功能测试"""

    @pytest.fixture
    def detector(self) -> ToolRepetitionDetector:
        """创建默认配置的检测器（max_repetition=3, window_size=5）"""
        with patch("app.security.guardrails.GUARDRAIL_INTERVENTION_COUNT"):
            return ToolRepetitionDetector(max_repetition=3, window_size=5)

    # =====================================================================
    # allow_normal_calls 测试
    # =====================================================================

    def test_allow_normal_calls_first(self, detector: ToolRepetitionDetector) -> None:
        """首次调用返回 ALLOW"""
        action = detector.check("search_internal_documents", {"query": "测试"})
        assert action == GuardrailAction.ALLOW

    def test_allow_normal_calls_second(self, detector: ToolRepetitionDetector) -> None:
        """第二次相同调用仍返回 ALLOW"""
        detector.check("search_internal_documents", {"query": "测试"})
        action = detector.check("search_internal_documents", {"query": "测试"})
        assert action == GuardrailAction.ALLOW

    def test_allow_different_tools(self, detector: ToolRepetitionDetector) -> None:
        """不同工具的调用互不影响"""
        for _ in range(5):
            detector.check("tool_a", {"arg": "value"})
        action = detector.check("tool_b", {"arg": "value"})
        assert action == GuardrailAction.ALLOW

    # =====================================================================
    # warn_repeated_calls 测试
    # =====================================================================

    def test_warn_on_threshold_minus_one(self, detector: ToolRepetitionDetector) -> None:
        """接近阈值时返回 WARN（max_repetition=3 时，第3次调用返回 WARN）"""
        detector.check("search_internal_documents", {"query": "测试"})  # 1st
        detector.check("search_internal_documents", {"query": "测试"})  # 2nd
        action = detector.check("search_internal_documents", {"query": "测试"})  # 3rd -> WARN
        assert action == GuardrailAction.WARN

    # =====================================================================
    # block_excessive_repetition 测试
    # =====================================================================

    def test_block_on_threshold_exceeded(self, detector: ToolRepetitionDetector) -> None:
        """超过阈值时返回 BLOCK（max_repetition=3 时，第4次调用返回 BLOCK）"""
        detector.check("search_internal_documents", {"query": "测试"})  # 1st -> ALLOW
        detector.check("search_internal_documents", {"query": "测试"})  # 2nd -> ALLOW
        detector.check("search_internal_documents", {"query": "测试"})  # 3rd -> WARN
        action = detector.check("search_internal_documents", {"query": "测试"})  # 4th -> BLOCK
        assert action == GuardrailAction.BLOCK

    def test_block_continues_after_threshold(self, detector: ToolRepetitionDetector) -> None:
        """超过阈值后持续返回 BLOCK"""
        for _ in range(4):
            detector.check("search_internal_documents", {"query": "测试"})
        action = detector.check("search_internal_documents", {"query": "测试"})
        assert action == GuardrailAction.BLOCK

    # =====================================================================
    # window_sliding 测试
    # =====================================================================

    def test_window_sliding_old_calls_expire(self) -> None:
        """滑动窗口：旧调用记录过期后不再计入"""
        with patch("app.security.guardrails.GUARDRAIL_INTERVENTION_COUNT"):
            # window_size=3，只保留最近3条记录
            detector = ToolRepetitionDetector(max_repetition=3, window_size=3)

        # 连续3次相同调用
        detector.check("tool_a", {"arg": "val"})  # 1st
        detector.check("tool_a", {"arg": "val"})  # 2nd

        # 插入2次不同调用，将旧记录挤出窗口
        detector.check("tool_b", {"arg": "other"})  # 挤出1st
        detector.check("tool_b", {"arg": "other"})  # 挤出2nd

        # 此时窗口中 tool_a 的记录为0，应返回 ALLOW
        action = detector.check("tool_a", {"arg": "val"})
        assert action == GuardrailAction.ALLOW

    def test_window_size_limits_history(self) -> None:
        """窗口大小限制历史记录数量"""
        with patch("app.security.guardrails.GUARDRAIL_INTERVENTION_COUNT"):
            detector = ToolRepetitionDetector(max_repetition=2, window_size=3)

        # 2次相同调用
        detector.check("tool_a", {"arg": "val"})  # ALLOW
        action = detector.check("tool_a", {"arg": "val"})  # WARN (count=2)
        assert action == GuardrailAction.WARN

        # 1次不同调用
        detector.check("tool_b", {"arg": "other"})

        # 窗口: [tool_a, tool_a, tool_b]，tool_a 仍出现2次
        action = detector.check("tool_a", {"arg": "val"})  # BLOCK (count=3 in window)
        assert action == GuardrailAction.BLOCK

    # =====================================================================
    # reset_detector 测试
    # =====================================================================

    def test_reset_clears_history(self, detector: ToolRepetitionDetector) -> None:
        """重置后历史记录清空"""
        for _ in range(4):
            detector.check("search_internal_documents", {"query": "测试"})
        # 此时应该 BLOCK
        action = detector.check("search_internal_documents", {"query": "测试"})
        assert action == GuardrailAction.BLOCK

        # 重置
        detector.reset()

        # 重置后应返回 ALLOW
        action = detector.check("search_internal_documents", {"query": "测试"})
        assert action == GuardrailAction.ALLOW

    # =====================================================================
    # different_args_not_repeated 测试
    # =====================================================================

    def test_different_args_not_repeated(self, detector: ToolRepetitionDetector) -> None:
        """不同参数的调用不计为重复"""
        for _ in range(5):
            detector.check("search_internal_documents", {"query": "测试A"})
        # 不同参数
        action = detector.check("search_internal_documents", {"query": "测试B"})
        assert action == GuardrailAction.ALLOW

    def test_different_args_same_tool(self, detector: ToolRepetitionDetector) -> None:
        """同一工具不同参数交替调用不会触发 BLOCK"""
        for i in range(10):
            action = detector.check("search_internal_documents", {"query": f"查询{i}"})
            assert action == GuardrailAction.ALLOW

    def test_arg_order_irrelevant(self, detector: ToolRepetitionDetector) -> None:
        """参数顺序不影响重复判定（JSON sort_keys=True）"""
        detector.check("tool_a", {"a": "1", "b": "2"})
        # 相同参数不同顺序
        action = detector.check("tool_a", {"b": "2", "a": "1"})
        assert action == GuardrailAction.ALLOW  # 第2次，仍 ALLOW

    # =====================================================================
    # 边界值测试
    # =====================================================================

    def test_max_repetition_one(self) -> None:
        """max_repetition=1 时，第2次调用即 BLOCK"""
        with patch("app.security.guardrails.GUARDRAIL_INTERVENTION_COUNT"):
            detector = ToolRepetitionDetector(max_repetition=1, window_size=5)

        detector.check("tool_a", {"arg": "val"})  # 1st -> ALLOW
        action = detector.check("tool_a", {"arg": "val"})  # 2nd -> BLOCK
        assert action == GuardrailAction.BLOCK

    def test_window_size_one(self) -> None:
        """window_size=1 时，窗口只保留1条记录，无法触发重复"""
        with patch("app.security.guardrails.GUARDRAIL_INTERVENTION_COUNT"):
            detector = ToolRepetitionDetector(max_repetition=3, window_size=1)

        for _ in range(10):
            action = detector.check("tool_a", {"arg": "val"})
            assert action == GuardrailAction.ALLOW

    def test_empty_args_consistent(self, detector: ToolRepetitionDetector) -> None:
        """空参数的重复调用正常检测"""
        detector.check("tool_a", {})
        detector.check("tool_a", {})
        action = detector.check("tool_a", {})  # 3rd -> WARN
        assert action == GuardrailAction.WARN