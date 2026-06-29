"""MCP 客户端模块

管理多 MCP Server 连接，提供工具发现、调用和健康检查能力。
支持连接失败降级、工具调用超时控制和 MCP 功能开关。
"""

import logging
import threading
import time
from typing import Any, TYPE_CHECKING

from app.exceptions import (
    MCPConnectionError,
    MCPToolCallError,
    MCPToolCallTimeoutError,
)
from app.mcp.registry import MCPRegistry
from app.mcp.server import BaseMCPAdapter

if TYPE_CHECKING:
    from app.review.features import FeatureFlags

logger = logging.getLogger(__name__)

# 默认工具调用超时时间（秒）
DEFAULT_TOOL_CALL_TIMEOUT: int = 30


class MCPClient:
    """MCP 客户端，管理多 MCP Server 连接

    负责连接和断开 MCP Server，提供统一的工具发现和调用接口。
    支持连接失败降级、工具调用超时控制和 MCP 功能开关。

    Attributes:
        registry: MCP 工具注册表
        features: 功能开关管理器（可选，用于检查 MCP 是否启用）
        tool_call_timeout: 工具调用超时时间（秒）
    """

    def __init__(
        self,
        registry: MCPRegistry,
        features: "FeatureFlags | None" = None,
        tool_call_timeout: int = DEFAULT_TOOL_CALL_TIMEOUT,
    ) -> None:
        """初始化 MCP 客户端

        Args:
            registry: MCP 工具注册表实例
            features: 功能开关管理器（可选）
            tool_call_timeout: 工具调用超时时间（秒）
        """
        self._registry = registry
        self._features = features
        self._tool_call_timeout = tool_call_timeout
        self._lock = threading.RLock()
        # server_name -> 连接状态 (connected/disconnected/unavailable/auth_failed)
        self._server_status: dict[str, str] = {}

    @property
    def registry(self) -> MCPRegistry:
        """获取工具注册表"""
        return self._registry

    def connect_server(self, server_name: str, adapter: BaseMCPAdapter) -> None:
        """连接 MCP Server

        执行健康检查，如果通过则注册适配器并标记为已连接；
        如果健康检查失败则标记为不可用，不影响其他 Server。

        Args:
            server_name: Server 名称
            adapter: MCP Server 适配器实例

        Raises:
            MCPConnectionError: 连接失败时抛出（但不会阻止其他 Server 连接）
        """
        # 检查 MCP 功能是否启用
        if self._features and not self._features.is_mcp_enabled():
            logger.info("MCP 功能未启用，跳过连接 Server: %s", server_name)
            return

        with self._lock:
            try:
                # 执行健康检查
                is_healthy = adapter.health_check()
                if not is_healthy:
                    self._server_status[server_name] = "unavailable"
                    logger.warning("MCP Server %s 健康检查失败，标记为不可用", server_name)
                    raise MCPConnectionError(
                        message=f"MCP Server 健康检查失败: {server_name}",
                        server_name=server_name,
                    )

                # 注册适配器
                self._registry.register_adapter(adapter)
                self._server_status[server_name] = "connected"
                logger.info("MCP Server %s 已连接", server_name)

            except MCPConnectionError:
                raise
            except Exception as exc:
                self._server_status[server_name] = "unavailable"
                logger.error("MCP Server %s 连接失败: %s", server_name, exc)
                raise MCPConnectionError(
                    message=f"MCP Server 连接失败: {server_name}",
                    server_name=server_name,
                    details=str(exc),
                ) from exc

    def disconnect_server(self, server_name: str) -> None:
        """断开 MCP Server 连接

        移除适配器及其工具，标记为已断开。

        Args:
            server_name: Server 名称
        """
        with self._lock:
            self._registry.unregister_adapter(server_name)
            self._server_status[server_name] = "disconnected"
            logger.info("MCP Server %s 已断开", server_name)

    def list_tools(self) -> list[Any]:
        """列出所有可用 MCP 工具

        仅返回已连接 Server 提供的工具。

        Returns:
            MCPToolInfo 对象列表
        """
        # 检查 MCP 功能是否启用
        if self._features and not self._features.is_mcp_enabled():
            logger.info("MCP 功能未启用，返回空工具列表")
            return []

        return self._registry.list_all_tools()

    def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        server_name: str | None = None,
    ) -> Any:
        """调用 MCP 工具，含超时控制

        通过注册表路由工具调用到对应的适配器。
        如果工具调用超过超时时间，取消调用并记录超时日志。

        Args:
            tool_name: 工具名称
            arguments: 工具调用参数
            server_name: 可选的目标 Server 名称

        Returns:
            工具执行结果

        Raises:
            MCPToolCallTimeoutError: 工具调用超时
            MCPToolCallError: 工具调用失败
        """
        # 检查 MCP 功能是否启用
        if self._features and not self._features.is_mcp_enabled():
            raise MCPToolCallError(
                message="MCP 功能未启用",
                tool_name=tool_name,
            )

        start_time = time.monotonic()

        try:
            # 使用带超时的调用
            result = self._call_tool_with_timeout(tool_name, arguments, server_name)
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            logger.info("MCP 工具调用完成: %s，耗时 %dms", tool_name, elapsed_ms)
            return result

        except MCPToolCallTimeoutError:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            logger.error(
                "MCP 工具调用超时: %s，耗时 %dms（超时阈值: %ds）",
                tool_name,
                elapsed_ms,
                self._tool_call_timeout,
            )
            raise
        except MCPToolCallError:
            raise
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            raise MCPToolCallError(
                message=f"MCP 工具调用失败: {tool_name}",
                tool_name=tool_name,
                details=str(exc),
            ) from exc

    def _call_tool_with_timeout(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        server_name: str | None = None,
    ) -> Any:
        """带超时控制的工具调用实现

        由于适配器的 call_tool 是同步方法，使用线程+Event 实现超时控制。

        Args:
            tool_name: 工具名称
            arguments: 工具调用参数
            server_name: 可选的目标 Server 名称

        Returns:
            工具执行结果

        Raises:
            MCPToolCallTimeoutError: 工具调用超时
        """
        result_container: dict[str, Any] = {}
        error_container: dict[str, Exception] = {}
        done_event = threading.Event()

        def _execute() -> None:
            try:
                result = self._registry.route_tool_call(tool_name, arguments, server_name)
                result_container["result"] = result
            except Exception as exc:
                error_container["error"] = exc
            finally:
                done_event.set()

        worker = threading.Thread(target=_execute, daemon=True)
        worker.start()

        if not done_event.wait(timeout=self._tool_call_timeout):
            logger.warning(
                "MCP 工具调用超时，工作线程仍在后台运行: tool_name=%s, "
                "timeout=%ds, thread_alive=%s",
                tool_name,
                self._tool_call_timeout,
                worker.is_alive(),
            )
            raise MCPToolCallTimeoutError(
                message=f"MCP 工具调用超时: {tool_name}（{self._tool_call_timeout}s）",
                tool_name=tool_name,
            )

        if "error" in error_container:
            raise error_container["error"]

        return result_container.get("result")

    def health_check(self) -> dict[str, str]:
        """检查各 MCP Server 连接状态

        对所有已连接的 Server 执行健康检查，更新连接状态。

        Returns:
            Server 名称到状态的映射，状态值：
            - connected: 已连接且健康
            - unavailable: 不可用
            - disconnected: 已断开
            - auth_failed: 认证失败
        """
        result: dict[str, str] = {}

        with self._lock:
            for server_name, status in self._server_status.items():
                if status == "connected":
                    # 对已连接的 Server 执行健康检查
                    adapter = self._registry.get_adapter(server_name)
                    if adapter:
                        try:
                            is_healthy = adapter.health_check()
                            result[server_name] = "connected" if is_healthy else "unavailable"
                        except Exception:
                            result[server_name] = "unavailable"
                    else:
                        result[server_name] = "unavailable"
                else:
                    result[server_name] = status

        return result

    def get_server_status(self, server_name: str) -> str:
        """获取指定 Server 的连接状态

        Args:
            server_name: Server 名称

        Returns:
            状态字符串，如果 Server 未注册则返回 "unknown"
        """
        with self._lock:
            return self._server_status.get(server_name, "unknown")

    def is_mcp_enabled(self) -> bool:
        """检查 MCP 功能是否启用

        Returns:
            True 表示 MCP 功能已启用
        """
        if self._features:
            return self._features.is_mcp_enabled()
        # 如果没有 features 管理器，默认启用
        return True


__all__ = [
    "MCPClient",
    "DEFAULT_TOOL_CALL_TIMEOUT",
]
