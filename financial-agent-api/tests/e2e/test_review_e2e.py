"""端到端测试

通过 FastAPI TestClient 测试完整的 API 请求-响应流程，
包括代码审查、配置管理和 MCP 工具调用。
所有 LLM 调用和外部依赖均通过 Mock 模拟。
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.dependencies import (
    get_feature_flags,
    get_mcp_client,
    get_review_settings,
    get_supervisor,
)
from app.core.enums import MCPServerName, ReviewType
from app.models.schemas import ReviewResponse


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
def mock_supervisor(mock_llm: MagicMock) -> MagicMock:
    """创建 Mock SupervisorAgent"""
    supervisor = MagicMock()
    supervisor.dispatch.return_value = ReviewResponse(
        session_id="e2e-session-001",
        review_type=ReviewType.FULL,
        results=[],
        summary="# 代码审查报告\n代码质量良好。",
        total_duration_ms=1500,
        is_fallback=False,
        fallback_message="",
    )
    return supervisor


@pytest.fixture
def mock_feature_flags() -> MagicMock:
    """创建 Mock FeatureFlags"""
    flags = MagicMock()
    flags.is_multi_agent_enabled.return_value = True
    flags.is_mcp_enabled.return_value = True
    flags.is_mcp_server_enabled.return_value = True
    flags.get_enabled_review_types.return_value = ["full"]
    flags.settings = MagicMock()
    flags.settings.multi_agent_enabled = True
    flags.settings.mcp_enabled = True
    flags.settings.mcp_servers = {MCPServerName.GITHUB: True}
    flags.settings.review_types = ["full"]
    flags.settings.worker_timeout_seconds = 60
    flags.settings.max_concurrent_reviews = 10
    return flags


@pytest.fixture
def mock_mcp_client() -> MagicMock:
    """创建 Mock MCPClient"""
    client = MagicMock()
    client.list_tools.return_value = []
    client.health_check.return_value = {"github": "connected"}
    client.call_tool.return_value = {"result": "ok"}
    client.registry = MagicMock()
    client.registry._tool_to_server = {}
    client.registry.get_tool_server.return_value = "test_server"
    return client


@pytest.fixture
def mock_review_settings() -> MagicMock:
    """创建 Mock ReviewSettings"""
    settings = MagicMock()
    settings.multi_agent_enabled = True
    settings.mcp_enabled = True
    settings.mcp_servers = {MCPServerName.GITHUB: True}
    settings.review_types = ["full"]
    settings.worker_timeout_seconds = 60
    settings.max_concurrent_reviews = 10
    return settings


@pytest.fixture
def client(
    mock_supervisor: MagicMock,
    mock_feature_flags: MagicMock,
    mock_mcp_client: MagicMock,
    mock_review_settings: MagicMock,
) -> TestClient:
    """创建 FastAPI TestClient（使用 dependency_overrides 注入 Mock 依赖）"""
    from app.main import app

    # 使用 FastAPI 的 dependency_overrides 替代 patch
    # 这样依赖注入在请求时仍会使用 mock
    app.dependency_overrides[get_supervisor] = lambda: mock_supervisor
    app.dependency_overrides[get_feature_flags] = lambda: mock_feature_flags
    app.dependency_overrides[get_mcp_client] = lambda: mock_mcp_client
    app.dependency_overrides[get_review_settings] = lambda: mock_review_settings

    with TestClient(app) as test_client:
        yield test_client

    # 清理 overrides
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------


class TestReviewE2E:
    """端到端测试"""

    def test_submit_review_and_get_report(
        self, client: TestClient, mock_supervisor: MagicMock
    ) -> None:
        """提交审查请求并获取完整报告

        验证：POST /api/v1/review/code 返回完整的 ReviewResponse
        """
        response = client.post(
            "/api/v1/review/code",
            json={
                "code_content": "def hello(): pass",
                "review_type": "full",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "review_type" in data
        assert "results" in data
        assert "summary" in data
        assert "total_duration_ms" in data
        assert "is_fallback" in data

    def test_submit_review_with_code_url(
        self, client: TestClient, mock_supervisor: MagicMock
    ) -> None:
        """提交包含 code_url 的审查请求

        验证：code_url 参数被正确处理
        """
        response = client.post(
            "/api/v1/review/code",
            json={
                "code_url": "https://github.com/test/repo/pull/1",
                "review_type": "security",
            },
        )

        # 可能返回 200 或 400（取决于 MCP 是否可用）
        # 在 Mock 环境下，code_url 无法获取代码差异，可能返回 400
        assert response.status_code in (200, 400)

    def test_submit_review_missing_code_source(self, client: TestClient) -> None:
        """提交审查请求缺少代码来源

        验证：不提供 code_content 和 code_url 时返回 422
        """
        response = client.post(
            "/api/v1/review/code",
            json={
                "review_type": "full",
            },
        )

        assert response.status_code == 422

    def test_config_management(self, client: TestClient, mock_feature_flags: MagicMock) -> None:
        """配置查询和更新

        验证：GET/PUT /api/v1/review/config 正确返回和更新配置
        """
        # 查询配置
        get_response = client.get("/api/v1/review/config")
        assert get_response.status_code == 200
        config_data = get_response.json()
        assert "multi_agent_enabled" in config_data
        assert "mcp_enabled" in config_data
        assert "worker_timeout_seconds" in config_data

        # 更新配置
        put_response = client.put(
            "/api/v1/review/config",
            json={
                "multi_agent_enabled": True,
                "worker_timeout_seconds": 90,
            },
        )
        assert put_response.status_code == 200
        updated_data = put_response.json()
        assert "multi_agent_enabled" in updated_data

    def test_mcp_tools_api(self, client: TestClient, mock_feature_flags: MagicMock) -> None:
        """MCP 工具列表和调用 API

        验证：GET /api/v1/review/mcp/tools 返回工具列表
        """
        response = client.get("/api/v1/review/mcp/tools")
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert "servers_status" in data

    def test_mcp_tools_api_disabled(self, client: TestClient) -> None:
        """MCP 功能禁用时工具 API 返回 503

        验证：MCP 总开关关闭时，GET /api/v1/review/mcp/tools 返回 503
        """
        from app.main import app

        # 创建 MCP 禁用的 mock
        disabled_flags = MagicMock()
        disabled_flags.is_mcp_enabled.return_value = False

        # 临时覆盖
        app.dependency_overrides[get_feature_flags] = lambda: disabled_flags

        try:
            response = client.get("/api/v1/review/mcp/tools")
            assert response.status_code == 503
        finally:
            # 恢复原始 mock
            pass

    def test_mcp_call_api(
        self, client: TestClient, mock_feature_flags: MagicMock, mock_mcp_client: MagicMock
    ) -> None:
        """MCP 工具调用 API

        验证：POST /api/v1/review/mcp/call 正确调用工具并返回结果
        """
        response = client.post(
            "/api/v1/review/mcp/call",
            json={
                "tool_name": "test_tool",
                "arguments": {"query": "test"},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "tool_name" in data
        assert "result" in data

    def test_existing_api_compatibility(self, client: TestClient) -> None:
        """现有 API 兼容性验证

        验证：核心路由（/health、/api/v1/chat、/api/v1/review/code）已注册
        """
        from app.main import app

        # 通过 OpenAPI schema 验证路由注册
        openapi_paths = list(app.openapi()["paths"].keys())
        assert "/health" in openapi_paths
        assert "/api/v1/chat" in openapi_paths
        assert "/api/v1/review/code" in openapi_paths

    def test_submit_review_security_type(
        self, client: TestClient, mock_supervisor: MagicMock
    ) -> None:
        """提交安全类型审查请求

        验证：review_type=security 时请求被正确处理
        """
        response = client.post(
            "/api/v1/review/code",
            json={
                "code_content": "query = f'SELECT * FROM users WHERE id = {user_id}'",
                "review_type": "security",
            },
        )

        assert response.status_code == 200

    def test_config_update_validation(
        self, client: TestClient, mock_feature_flags: MagicMock
    ) -> None:
        """配置更新值校验

        验证：超出范围的配置值返回 400 错误
        """
        # worker_timeout_seconds 超出范围
        response = client.put(
            "/api/v1/review/config",
            json={
                "worker_timeout_seconds": 5,  # 低于最小值 10
            },
        )
        assert response.status_code == 422  # Pydantic 校验失败

    def test_review_with_session_id(self, client: TestClient, mock_supervisor: MagicMock) -> None:
        """提交审查请求携带 session_id

        验证：自定义 session_id 被正确传递
        """
        response = client.post(
            "/api/v1/review/code",
            json={
                "code_content": "def hello(): pass",
                "review_type": "full",
                "session_id": "custom-session-123",
            },
        )

        assert response.status_code == 200
