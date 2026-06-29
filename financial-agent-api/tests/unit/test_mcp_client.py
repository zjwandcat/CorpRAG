"""MCP 客户端单元测试

测试 MCPClient 的连接管理、工具发现、工具调用和超时控制。
所有外部依赖（适配器）均通过 Mock 模拟。
"""

from unittest.mock import MagicMock

import pytest

from app.exceptions import (
    MCPConnectionError,
    MCPToolCallError,
    MCPToolCallTimeoutError,
)
from app.mcp.client import MCPClient
from app.mcp.registry import MCPRegistry
from app.models.schemas import MCPToolInfo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry() -> MCPRegistry:
    """创建 MCPRegistry 实例"""
    return MCPRegistry()


@pytest.fixture
def mock_features() -> MagicMock:
    """创建 Mock FeatureFlags，默认 MCP 启用"""
    features = MagicMock()
    features.is_mcp_enabled.return_value = True
    features.is_mcp_server_enabled.return_value = True
    return features


@pytest.fixture
def mcp_client(registry: MCPRegistry, mock_features: MagicMock) -> MCPClient:
    """创建 MCPClient 实例"""
    return MCPClient(
        registry=registry,
        features=mock_features,
        tool_call_timeout=5,
    )


@pytest.fixture
def mock_adapter() -> MagicMock:
    """创建 Mock MCP 适配器"""
    adapter = MagicMock()
    adapter.server_name = "test_server"
    adapter.health_check.return_value = True

    # Mock list_tools 返回工具列表
    tool_info = MCPToolInfo(
        name="test_tool",
        description="测试工具",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        server_name="test_server",
    )
    adapter.list_tools.return_value = [tool_info]
    adapter.call_tool.return_value = {"result": "ok"}
    adapter.validate_tool_definition.return_value = True

    return adapter


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------


class TestMCPClient:
    """MCPClient 客户端测试"""

    def test_connect_server(
        self,
        mcp_client: MCPClient,
        mock_adapter: MagicMock,
    ) -> None:
        """连接 MCP Server

        验证：连接成功后 Server 状态为 connected，工具已注册
        """
        mcp_client.connect_server("test_server", mock_adapter)

        assert mcp_client.get_server_status("test_server") == "connected"
        # 验证适配器已注册到注册表
        assert mcp_client.registry.has_tool("test_tool")

    def test_disconnect_server(
        self,
        mcp_client: MCPClient,
        mock_adapter: MagicMock,
    ) -> None:
        """断开 MCP Server

        验证：断开后 Server 状态为 disconnected，工具已移除
        """
        # 先连接
        mcp_client.connect_server("test_server", mock_adapter)
        assert mcp_client.get_server_status("test_server") == "connected"

        # 断开
        mcp_client.disconnect_server("test_server")
        assert mcp_client.get_server_status("test_server") == "disconnected"
        # 验证工具已移除
        assert not mcp_client.registry.has_tool("test_tool")

    def test_list_tools(
        self,
        mcp_client: MCPClient,
        mock_adapter: MagicMock,
    ) -> None:
        """列出可用工具

        验证：连接 Server 后，list_tools 返回该 Server 的工具列表
        """
        mcp_client.connect_server("test_server", mock_adapter)

        tools = mcp_client.list_tools()
        assert len(tools) >= 1
        tool_names = [t.name for t in tools]
        assert "test_tool" in tool_names

    def test_call_tool(
        self,
        mcp_client: MCPClient,
        mock_adapter: MagicMock,
    ) -> None:
        """调用 MCP 工具

        验证：通过 MCPClient 调用工具，返回正确结果
        """
        mcp_client.connect_server("test_server", mock_adapter)

        result = mcp_client.call_tool("test_tool", {"query": "test"})
        assert result == {"result": "ok"}

    def test_tool_call_timeout(
        self,
        mcp_client: MCPClient,
        mock_adapter: MagicMock,
    ) -> None:
        """工具调用超时

        验证：工具调用超过超时时间后抛出 MCPToolCallTimeoutError
        """
        # 设置极短的超时时间
        mcp_client._tool_call_timeout = 1

        # Mock call_tool 模拟耗时操作
        def _slow_call(*args, **kwargs):
            import time

            time.sleep(5)
            return "should not reach"

        # 让注册表中的路由调用 mock_adapter 的 call_tool
        mcp_client.connect_server("test_server", mock_adapter)
        mock_adapter.call_tool = _slow_call
        # 重新注册以更新 call_tool
        mcp_client.registry.register_adapter(mock_adapter)

        with pytest.raises(MCPToolCallTimeoutError):
            mcp_client.call_tool("test_tool", {"query": "test"})

    def test_connection_failure(
        self,
        mcp_client: MCPClient,
    ) -> None:
        """连接失败降级

        验证：MCP Server 健康检查失败时，标记为 unavailable，
        不影响其他 Server 的连接
        """
        # 创建一个健康检查失败的适配器
        fail_adapter = MagicMock()
        fail_adapter.server_name = "fail_server"
        fail_adapter.health_check.return_value = False

        with pytest.raises(MCPConnectionError):
            mcp_client.connect_server("fail_server", fail_adapter)

        assert mcp_client.get_server_status("fail_server") == "unavailable"

    def test_mcp_disabled_list_tools(
        self,
        registry: MCPRegistry,
    ) -> None:
        """MCP 功能禁用时 list_tools 返回空列表

        验证：MCP 总开关关闭时，list_tools 返回空列表
        """
        features = MagicMock()
        features.is_mcp_enabled.return_value = False

        client = MCPClient(registry=registry, features=features)
        tools = client.list_tools()
        assert tools == []

    def test_mcp_disabled_call_tool(
        self,
        registry: MCPRegistry,
    ) -> None:
        """MCP 功能禁用时 call_tool 抛出错误

        验证：MCP 总开关关闭时，call_tool 抛出 MCPToolCallError
        """
        features = MagicMock()
        features.is_mcp_enabled.return_value = False

        client = MCPClient(registry=registry, features=features)
        with pytest.raises(MCPToolCallError):
            client.call_tool("any_tool", {})

    def test_health_check(
        self,
        mcp_client: MCPClient,
        mock_adapter: MagicMock,
    ) -> None:
        """健康检查返回各 Server 状态

        验证：health_check 返回已连接 Server 的状态映射
        """
        mcp_client.connect_server("test_server", mock_adapter)

        status = mcp_client.health_check()
        assert "test_server" in status
        assert status["test_server"] == "connected"

    def test_connect_mcp_disabled_skip(
        self,
        registry: MCPRegistry,
    ) -> None:
        """MCP 功能禁用时跳过连接

        验证：MCP 总开关关闭时，connect_server 不执行连接操作
        """
        features = MagicMock()
        features.is_mcp_enabled.return_value = False

        client = MCPClient(registry=registry, features=features)
        adapter = MagicMock()
        adapter.server_name = "skip_server"

        # 不会抛出异常，也不会连接
        client.connect_server("skip_server", adapter)
        # Server 状态应为 unknown（未尝试连接）
        assert client.get_server_status("skip_server") == "unknown"
