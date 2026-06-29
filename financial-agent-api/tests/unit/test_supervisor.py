"""SupervisorAgent 调度逻辑单元测试

测试 Supervisor 模式的任务分发、Worker 调度、结果汇总和降级策略。
所有 LLM 调用均通过 Mock 模拟，不依赖真实 LLM 服务。
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.agent.review.supervisor import SupervisorAgent
from app.core.enums import ReviewStatus, ReviewType
from app.models.schemas import ReviewResponse, WorkerResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm() -> MagicMock:
    """创建 Mock LLM 实例"""
    llm = MagicMock()
    # 默认返回一个简单的审查报告
    response = MagicMock()
    response.content = "# 审查报告\n代码质量良好。"
    llm.invoke.return_value = response
    return llm


@pytest.fixture
def supervisor(mock_llm: MagicMock) -> SupervisorAgent:
    """创建 SupervisorAgent 实例（使用 Mock LLM）"""
    return SupervisorAgent(llm=mock_llm, worker_timeout_seconds=5)


@pytest.fixture
def sql_injection_code() -> str:
    """包含 SQL 注入漏洞的示例代码"""
    return """
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    return cursor.fetchone()
"""


@pytest.fixture
def full_review_request() -> SimpleNamespace:
    """全面审查请求"""
    return SimpleNamespace(
        code_content="def hello(): pass",
        code_url=None,
        review_type=ReviewType.FULL,
        session_id="test-session-001",
    )


@pytest.fixture
def security_review_request() -> SimpleNamespace:
    """安全审查请求"""
    return SimpleNamespace(
        code_content="def hello(): pass",
        code_url=None,
        review_type=ReviewType.SECURITY,
        session_id="test-session-002",
    )


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------


class TestSupervisorAgent:
    """SupervisorAgent 调度逻辑测试"""

    def test_dispatch_full_review(
        self,
        supervisor: SupervisorAgent,
        full_review_request: SimpleNamespace,
    ) -> None:
        """全面审查分发到所有 Worker

        验证：review_type=full 时，结果包含 security/architecture/performance/style 四个维度
        """
        # Mock 各 Worker 的 safe_execute 方法
        for dimension, worker in supervisor._workers.items():
            worker.safe_execute = MagicMock(
                return_value=WorkerResult(
                    dimension=ReviewType(dimension),
                    status=ReviewStatus.COMPLETED,
                    findings=[],
                    duration_ms=100,
                )
            )

        # Mock SummaryAgent
        supervisor._summary_agent.generate_summary = MagicMock(
            return_value="# 审查报告\n代码质量良好。"
        )

        response = supervisor.dispatch(full_review_request)

        assert isinstance(response, ReviewResponse)
        assert response.review_type == ReviewType.FULL
        assert response.is_fallback is False
        assert len(response.results) == 4
        result_dimensions = {r.dimension for r in response.results}
        assert ReviewType.SECURITY in result_dimensions
        assert ReviewType.ARCHITECTURE in result_dimensions
        assert ReviewType.PERFORMANCE in result_dimensions
        assert ReviewType.STYLE in result_dimensions

    def test_dispatch_security_only(
        self,
        supervisor: SupervisorAgent,
        security_review_request: SimpleNamespace,
    ) -> None:
        """安全审查仅分发到 Security Agent

        验证：review_type=security 时，结果仅包含 security 维度
        """
        # Mock Security Worker 的 safe_execute
        supervisor._workers["security"].safe_execute = MagicMock(
            return_value=WorkerResult(
                dimension=ReviewType.SECURITY,
                status=ReviewStatus.COMPLETED,
                findings=[],
                duration_ms=100,
            )
        )

        # Mock SummaryAgent
        supervisor._summary_agent.generate_summary = MagicMock(return_value="# 安全审查报告")

        response = supervisor.dispatch(security_review_request)

        assert isinstance(response, ReviewResponse)
        assert response.review_type == ReviewType.SECURITY
        assert len(response.results) == 1
        assert response.results[0].dimension == ReviewType.SECURITY

    def test_worker_timeout_handling(
        self,
        supervisor: SupervisorAgent,
        full_review_request: SimpleNamespace,
    ) -> None:
        """Worker 超时标记 timeout 继续执行

        验证：某个 Worker 超时后，Supervisor 标记该维度为 timeout，
        其余 Worker 继续执行并正常返回结果
        """

        # Mock Security Worker 超时
        def _timeout_execute(**kwargs):
            raise TimeoutError("Worker 执行超时")

        supervisor._workers["security"].safe_execute = MagicMock(side_effect=_timeout_execute)

        # Mock 其余 Worker 正常完成
        for dimension in ["architecture", "performance", "style"]:
            supervisor._workers[dimension].safe_execute = MagicMock(
                return_value=WorkerResult(
                    dimension=ReviewType(dimension),
                    status=ReviewStatus.COMPLETED,
                    findings=[],
                    duration_ms=100,
                )
            )

        # Mock SummaryAgent
        supervisor._summary_agent.generate_summary = MagicMock(return_value="# 审查报告")

        response = supervisor.dispatch(full_review_request)

        assert isinstance(response, ReviewResponse)
        # 应有 4 个结果（1 个 timeout + 3 个 completed）
        assert len(response.results) == 4

        # 检查超时维度
        timeout_results = [r for r in response.results if r.status == ReviewStatus.TIMEOUT]
        assert len(timeout_results) >= 1
        assert any(r.dimension == ReviewType.SECURITY for r in timeout_results)

        # 检查其余维度正常完成
        completed_results = [r for r in response.results if r.status == ReviewStatus.COMPLETED]
        assert len(completed_results) >= 1

    def test_worker_failure_handling(
        self,
        supervisor: SupervisorAgent,
        full_review_request: SimpleNamespace,
    ) -> None:
        """Worker 失败标记 failed 继续执行

        验证：某个 Worker 执行失败后，Supervisor 标记该维度为 failed，
        其余 Worker 继续执行并正常返回结果
        """

        # Mock Architecture Worker 失败
        def _fail_execute(**kwargs):
            raise RuntimeError("Worker 内部错误")

        supervisor._workers["architecture"].safe_execute = MagicMock(side_effect=_fail_execute)

        # Mock 其余 Worker 正常完成
        for dimension in ["security", "performance", "style"]:
            supervisor._workers[dimension].safe_execute = MagicMock(
                return_value=WorkerResult(
                    dimension=ReviewType(dimension),
                    status=ReviewStatus.COMPLETED,
                    findings=[],
                    duration_ms=100,
                )
            )

        # Mock SummaryAgent
        supervisor._summary_agent.generate_summary = MagicMock(return_value="# 审查报告")

        response = supervisor.dispatch(full_review_request)

        assert isinstance(response, ReviewResponse)
        assert len(response.results) == 4

        # 检查失败维度
        failed_results = [r for r in response.results if r.status == ReviewStatus.FAILED]
        assert len(failed_results) >= 1
        assert any(r.dimension == ReviewType.ARCHITECTURE for r in failed_results)

        # 检查其余维度正常完成
        completed_results = [r for r in response.results if r.status == ReviewStatus.COMPLETED]
        assert len(completed_results) >= 1

    def test_all_workers_failed(
        self,
        supervisor: SupervisorAgent,
        full_review_request: SimpleNamespace,
    ) -> None:
        """所有 Worker 失败返回错误

        验证：所有 Worker 均失败时，Supervisor 返回的 ReviewResponse 中
        所有维度状态为 failed，且降级为单 Agent 模式
        """
        # Mock 所有 Worker 失败 — 通过让 _dispatch_workers 抛出异常触发降级
        with patch.object(
            supervisor, "_dispatch_workers", side_effect=RuntimeError("所有 Worker 均失败")
        ):
            # Mock 降级模式下的 LLM 调用
            response = supervisor.dispatch(full_review_request)

        assert isinstance(response, ReviewResponse)
        assert response.is_fallback is True
        assert "降级" in response.fallback_message or "失败" in response.fallback_message

    def test_fallback_to_single_agent(
        self,
        supervisor: SupervisorAgent,
        full_review_request: SimpleNamespace,
    ) -> None:
        """Supervisor 失败降级为单 Agent

        验证：Supervisor 调度失败时，系统降级为单 Agent 模式，
        返回 is_fallback=True 和 fallback_message
        """
        # Mock _dispatch_workers 抛出异常触发降级
        with patch.object(supervisor, "_dispatch_workers", side_effect=Exception("调度失败")):
            response = supervisor.dispatch(full_review_request)

        assert isinstance(response, ReviewResponse)
        assert response.is_fallback is True
        assert response.fallback_message != ""
        assert "降级" in response.fallback_message or "失败" in response.fallback_message
        # 降级模式下 results 为空
        assert len(response.results) == 0
        # 降级模式下 summary 应有内容
        assert response.summary != ""
