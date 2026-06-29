"""多Agent协作完整流程集成测试

测试 Supervisor → Workers → Summary 的完整审查流程，
包括全面审查、部分审查和 MCP 工具集成。
所有 LLM 调用均通过 Mock 模拟。
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.agent.review.supervisor import SupervisorAgent
from app.core.enums import MCPServerName, ReviewStatus, ReviewType
from app.models.schemas import (
    MCPToolInfo,
    ReviewFinding,
    ReviewResponse,
    WorkerResult,
)
from app.mcp.client import MCPClient
from app.mcp.registry import MCPRegistry
from app.review.features import FeatureFlags
from app.review.settings import ReviewSettings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm() -> MagicMock:
    """创建 Mock LLM 实例"""
    llm = MagicMock()
    response = MagicMock()
    response.content = "# 审查报告\n代码质量良好。"
    llm.invoke.return_value = response
    return llm


@pytest.fixture
def review_settings() -> ReviewSettings:
    """创建启用了多Agent和MCP的配置"""
    settings = ReviewSettings()
    settings.multi_agent_enabled = True
    settings.mcp_enabled = True
    settings.mcp_servers = {
        MCPServerName.GITHUB: True,
        MCPServerName.FILESYSTEM: True,
        MCPServerName.DATABASE: True,
        MCPServerName.WEBSEARCH: False,
    }
    return settings


@pytest.fixture
def feature_flags(review_settings: ReviewSettings) -> FeatureFlags:
    """创建 FeatureFlags 实例"""
    return FeatureFlags(review_settings)


@pytest.fixture
def registry() -> MCPRegistry:
    """创建 MCPRegistry 实例"""
    return MCPRegistry()


@pytest.fixture
def mcp_client(registry: MCPRegistry, feature_flags: FeatureFlags) -> MCPClient:
    """创建 MCPClient 实例"""
    return MCPClient(
        registry=registry,
        features=feature_flags,
        tool_call_timeout=5,
    )


@pytest.fixture
def supervisor(mock_llm: MagicMock) -> SupervisorAgent:
    """创建 SupervisorAgent 实例"""
    return SupervisorAgent(llm=mock_llm, worker_timeout_seconds=5)


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------


class TestReviewFlow:
    """多Agent协作完整流程测试"""

    def test_full_review_flow(self, supervisor: SupervisorAgent) -> None:
        """完整审查流程（Supervisor → Workers → Summary）

        验证：全面审查流程从 Supervisor 分发到所有 Worker 执行，
        最终由 Summary Agent 汇总生成报告
        """
        # Mock 各 Worker 的 safe_execute
        for dimension, worker in supervisor._workers.items():
            worker.safe_execute = MagicMock(
                return_value=WorkerResult(
                    dimension=ReviewType(dimension),
                    status=ReviewStatus.COMPLETED,
                    findings=[
                        ReviewFinding(
                            severity="medium",
                            description=f"{dimension} 维度发现了一个问题",
                            location="test.py:1",
                            suggestion="修复建议",
                        )
                    ],
                    duration_ms=200,
                )
            )

        # Mock SummaryAgent
        supervisor._summary_agent.generate_summary = MagicMock(
            return_value="# 代码审查报告\n\n## 总结\n代码整体质量良好，有少量改进建议。"
        )

        review_request = SimpleNamespace(
            code_content="def hello(): pass",
            code_url=None,
            review_type=ReviewType.FULL,
            session_id="integration-session-001",
        )

        response = supervisor.dispatch(review_request)

        assert isinstance(response, ReviewResponse)
        assert response.session_id == "integration-session-001"
        assert response.review_type == ReviewType.FULL
        assert response.is_fallback is False
        assert len(response.results) == 4

        # 验证所有维度都已完成
        for result in response.results:
            assert result.status == ReviewStatus.COMPLETED

        # 验证 Summary 已生成
        assert response.summary != ""

        # 验证所有 Worker 都被调用
        for dimension, worker in supervisor._workers.items():
            worker.safe_execute.assert_called_once()

    def test_partial_review_flow(self, supervisor: SupervisorAgent) -> None:
        """部分审查流程（仅安全+性能）

        验证：分别执行安全审查和性能审查，确认只调度对应的 Worker Agent。
        由于 ReviewType 枚举不支持组合类型，我们分别执行两次独立审查。
        """
        # Mock Security Worker
        supervisor._workers["security"].safe_execute = MagicMock(
            return_value=WorkerResult(
                dimension=ReviewType.SECURITY,
                status=ReviewStatus.COMPLETED,
                findings=[],
                duration_ms=100,
            )
        )
        # Mock Performance Worker
        supervisor._workers["performance"].safe_execute = MagicMock(
            return_value=WorkerResult(
                dimension=ReviewType.PERFORMANCE,
                status=ReviewStatus.COMPLETED,
                findings=[],
                duration_ms=150,
            )
        )

        # Mock SummaryAgent
        supervisor._summary_agent.generate_summary = MagicMock(return_value="# 安全+性能审查报告")

        # 执行安全审查
        security_request = SimpleNamespace(
            code_content="def hello(): pass",
            code_url=None,
            review_type=ReviewType.SECURITY,
            session_id="integration-session-002a",
        )
        security_response = supervisor.dispatch(security_request)
        assert isinstance(security_response, ReviewResponse)
        assert len(security_response.results) == 1
        assert security_response.results[0].dimension == ReviewType.SECURITY

        # 执行性能审查
        performance_request = SimpleNamespace(
            code_content="def hello(): pass",
            code_url=None,
            review_type=ReviewType.PERFORMANCE,
            session_id="integration-session-002b",
        )
        performance_response = supervisor.dispatch(performance_request)
        assert isinstance(performance_response, ReviewResponse)
        assert len(performance_response.results) == 1
        assert performance_response.results[0].dimension == ReviewType.PERFORMANCE

    def test_review_with_mcp_tools(
        self,
        supervisor: SupervisorAgent,
        mcp_client: MCPClient,
        registry: MCPRegistry,
    ) -> None:
        """审查中使用 MCP 工具获取代码差异

        验证：通过 MCP Client 调用 GitHub 工具获取代码差异，
        并将差异内容作为审查输入
        """
        # Mock GitHub 适配器
        github_adapter = MagicMock()
        github_adapter.server_name = "github"
        github_adapter.health_check.return_value = True
        github_adapter.validate_tool_definition.return_value = True

        # Mock 工具列表
        github_tool = MCPToolInfo(
            name="github_get_pr_diff",
            description="获取 PR 代码差异",
            parameters={
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                    "pr_number": {"type": "integer"},
                },
                "required": ["owner", "repo", "pr_number"],
            },
            server_name="github",
        )
        github_adapter.list_tools.return_value = [github_tool]

        # Mock call_tool 返回代码差异
        github_adapter.call_tool.return_value = (
            "diff --git a/main.py b/main.py\n"
            "+def vulnerable_function(user_input):\n"
            '+    query = f"SELECT * FROM users WHERE id = {user_input}"\n'
            "+    return query\n"
        )

        # 连接 GitHub Server
        mcp_client.connect_server("github", github_adapter)

        # 验证工具已注册
        assert registry.has_tool("github_get_pr_diff")

        # 调用工具获取代码差异
        diff_result = mcp_client.call_tool(
            "github_get_pr_diff",
            {"owner": "test", "repo": "repo", "pr_number": 1},
        )
        assert "vulnerable_function" in str(diff_result)

        # Mock Worker 执行
        for dimension, worker in supervisor._workers.items():
            worker.safe_execute = MagicMock(
                return_value=WorkerResult(
                    dimension=ReviewType(dimension),
                    status=ReviewStatus.COMPLETED,
                    findings=[],
                    duration_ms=100,
                )
            )
        supervisor._summary_agent.generate_summary = MagicMock(return_value="# 审查报告")

        # 使用代码差异作为审查输入
        review_request = SimpleNamespace(
            code_content=str(diff_result),
            code_url="https://github.com/test/repo/pull/1",
            review_type=ReviewType.FULL,
            session_id="integration-session-003",
        )

        response = supervisor.dispatch(review_request)
        assert isinstance(response, ReviewResponse)
        assert response.is_fallback is False

    def test_mixed_worker_results(self, supervisor: SupervisorAgent) -> None:
        """混合 Worker 结果（部分成功、部分超时、部分失败）

        验证：Supervisor 能正确处理混合状态的 Worker 结果
        """
        # Security 正常完成
        supervisor._workers["security"].safe_execute = MagicMock(
            return_value=WorkerResult(
                dimension=ReviewType.SECURITY,
                status=ReviewStatus.COMPLETED,
                findings=[
                    ReviewFinding(
                        severity="critical",
                        description="SQL 注入漏洞",
                        location="main.py:10",
                        suggestion="使用参数化查询",
                    )
                ],
                duration_ms=300,
            )
        )

        # Architecture 超时
        supervisor._workers["architecture"].safe_execute = MagicMock(
            side_effect=TimeoutError("架构审查超时")
        )

        # Performance 失败
        supervisor._workers["performance"].safe_execute = MagicMock(
            side_effect=RuntimeError("性能审查内部错误")
        )

        # Style 正常完成
        supervisor._workers["style"].safe_execute = MagicMock(
            return_value=WorkerResult(
                dimension=ReviewType.STYLE,
                status=ReviewStatus.COMPLETED,
                findings=[],
                duration_ms=100,
            )
        )

        # Mock SummaryAgent
        supervisor._summary_agent.generate_summary = MagicMock(
            return_value="# 审查报告\n部分维度审查异常。"
        )

        review_request = SimpleNamespace(
            code_content="def hello(): pass",
            code_url=None,
            review_type=ReviewType.FULL,
            session_id="integration-session-004",
        )

        response = supervisor.dispatch(review_request)

        assert isinstance(response, ReviewResponse)
        assert len(response.results) == 4

        # 验证各维度状态
        completed = [r for r in response.results if r.status == ReviewStatus.COMPLETED]
        failed = [r for r in response.results if r.status == ReviewStatus.FAILED]
        timeout = [r for r in response.results if r.status == ReviewStatus.TIMEOUT]

        assert len(completed) >= 1  # Security + Style
        assert len(failed) >= 1  # Performance
        assert len(timeout) >= 1  # Architecture
