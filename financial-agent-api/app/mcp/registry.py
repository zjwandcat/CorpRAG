"""MCP 工具注册表模块

维护工具名到 Server/Adapter 的映射，支持工具注册、发现和调用路由。
将 MCP 工具转换为 LangChain BaseTool 格式，合并到现有 tools_by_name 映射表。
"""

import logging
import threading
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from app.exceptions import MCPToolCallError, MCPToolNotAllowedError
from app.mcp.server import BaseMCPAdapter, validate_tool_definition_impl

logger = logging.getLogger(__name__)


class MCPRegistry:
    """MCP 工具注册表

    维护工具名到 Server/Adapter 的映射，提供工具注册、发现和调用路由功能。
    支持将 MCP 工具转换为 LangChain BaseTool 格式，与现有工具系统兼容。

    线程安全：所有写操作使用 RLock 保护。
    """

    def __init__(self) -> None:
        """初始化工具注册表"""
        self._lock = threading.RLock()
        # server_name -> BaseMCPAdapter
        self._adapters: dict[str, BaseMCPAdapter] = {}
        # tool_name -> server_name
        self._tool_to_server: dict[str, str] = {}
        # tool_name -> MCPToolInfo (原始工具信息)
        self._tool_infos: dict[str, Any] = {}

    def register_adapter(self, adapter: BaseMCPAdapter) -> None:
        """注册适配器并收集其工具列表

        注册适配器后，自动发现其提供的所有工具，进行安全校验，
        并将通过校验的工具添加到注册表中。

        Args:
            adapter: MCP Server 适配器实例
        """
        with self._lock:
            server_name = adapter.server_name
            logger.info("注册 MCP Server 适配器: %s", server_name)

            # 如果该 Server 已注册，先移除旧的工具映射
            if server_name in self._adapters:
                self._remove_server_tools(server_name)

            self._adapters[server_name] = adapter

            # 发现并注册工具
            try:
                tools = adapter.list_tools()
            except Exception as exc:
                logger.error("获取 MCP Server %s 的工具列表失败: %s", server_name, exc)
                return

            registered_count = 0
            for tool in tools:
                tool_name = tool.name if hasattr(tool, "name") else str(tool)

                # 安全校验
                if not self.validate_tool_definition(tool):
                    logger.warning(
                        "MCP 工具 %s (来自 %s) 未通过安全校验，已跳过",
                        tool_name,
                        server_name,
                    )
                    continue

                # 检查工具名冲突
                if tool_name in self._tool_to_server:
                    existing_server = self._tool_to_server[tool_name]
                    logger.warning(
                        "MCP 工具名冲突: %s 已注册于 %s，%s 的注册被跳过",
                        tool_name,
                        existing_server,
                        server_name,
                    )
                    continue

                self._tool_to_server[tool_name] = server_name
                self._tool_infos[tool_name] = tool
                registered_count += 1
                logger.info("注册 MCP 工具: %s (来自 %s)", tool_name, server_name)

            logger.info(
                "MCP Server %s 注册完成，共注册 %d/%d 个工具",
                server_name,
                registered_count,
                len(tools),
            )

    def unregister_adapter(self, server_name: str) -> None:
        """移除适配器及其工具

        Args:
            server_name: 要移除的 Server 名称
        """
        with self._lock:
            if server_name not in self._adapters:
                logger.warning("MCP Server %s 未注册，无法移除", server_name)
                return

            self._remove_server_tools(server_name)
            del self._adapters[server_name]
            logger.info("MCP Server %s 已移除", server_name)

    def _remove_server_tools(self, server_name: str) -> None:
        """移除指定 Server 的所有工具映射（内部方法，需在锁内调用）

        Args:
            server_name: Server 名称
        """
        tools_to_remove = [
            tool_name
            for tool_name, srv_name in self._tool_to_server.items()
            if srv_name == server_name
        ]
        for tool_name in tools_to_remove:
            del self._tool_to_server[tool_name]
            self._tool_infos.pop(tool_name, None)

        if tools_to_remove:
            logger.info("移除 MCP Server %s 的 %d 个工具", server_name, len(tools_to_remove))

    def list_all_tools(self) -> list[Any]:
        """列出所有已注册工具

        Returns:
            MCPToolInfo 对象列表
        """
        with self._lock:
            return list(self._tool_infos.values())

    def route_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        server_name: str | None = None,
    ) -> Any:
        """工具调用路由

        根据工具名称路由到对应的适配器执行工具调用。
        如果指定了 server_name，则仅在该 Server 上查找工具。

        Args:
            tool_name: 工具名称
            arguments: 工具调用参数
            server_name: 可选的目标 Server 名称

        Returns:
            工具执行结果

        Raises:
            MCPToolCallError: 工具调用失败
            MCPToolNotAllowedError: 工具未注册或未授权
        """
        with self._lock:
            # 确定目标 Server
            target_server = server_name or self._tool_to_server.get(tool_name)

            if not target_server:
                raise MCPToolNotAllowedError(
                    message=f"未注册的 MCP 工具: {tool_name}",
                    tool_name=tool_name,
                )

            adapter = self._adapters.get(target_server)
            if not adapter:
                raise MCPToolCallError(
                    message=f"MCP Server 不可用: {target_server}",
                    tool_name=tool_name,
                )

            # 验证工具是否属于该 Server
            if tool_name not in self._tool_infos:
                raise MCPToolNotAllowedError(
                    message=f"未注册的 MCP 工具: {tool_name}",
                    tool_name=tool_name,
                )

        # 在锁外执行实际调用（避免长时间持锁）
        try:
            result = adapter.call_tool(tool_name, arguments)
            return result
        except Exception as exc:
            if isinstance(exc, (MCPToolCallError, MCPToolNotAllowedError)):
                raise
            raise MCPToolCallError(
                message=f"MCP 工具调用失败: {tool_name}",
                tool_name=tool_name,
                details=str(exc),
            ) from exc

    def validate_tool_definition(self, tool: Any) -> bool:
        """工具定义安全校验

        使用通用安全校验逻辑检查工具定义是否安全。

        Args:
            tool: 工具信息对象（MCPToolInfo）

        Returns:
            True 表示工具定义安全，False 表示不安全
        """
        return validate_tool_definition_impl(tool)

    def get_adapter(self, server_name: str) -> BaseMCPAdapter | None:
        """获取指定 Server 的适配器

        Args:
            server_name: Server 名称

        Returns:
            适配器实例，如果未注册则返回 None
        """
        with self._lock:
            return self._adapters.get(server_name)

    def get_registered_servers(self) -> list[str]:
        """获取所有已注册的 Server 名称

        Returns:
            Server 名称列表
        """
        with self._lock:
            return list(self._adapters.keys())

    def has_tool(self, tool_name: str) -> bool:
        """检查工具是否已注册

        Args:
            tool_name: 工具名称

        Returns:
            True 表示工具已注册
        """
        with self._lock:
            return tool_name in self._tool_to_server

    def get_tool_server(self, tool_name: str) -> str:
        """获取工具所属的 Server 名称

        Args:
            tool_name: 工具名称

        Returns:
            工具所属的 Server 名称，未找到时返回空字符串
        """
        with self._lock:
            return self._tool_to_server.get(tool_name, "")

    def to_langchain_tools(self) -> list[BaseTool]:
        """将所有已注册的 MCP 工具转换为 LangChain BaseTool 格式

        将 MCP 工具转换为 LangChain StructuredTool，使其可以与现有工具系统
        一起使用。每个 MCP 工具被包装为一个同步调用的 StructuredTool。

        Returns:
            LangChain BaseTool 列表
        """
        with self._lock:
            langchain_tools: list[BaseTool] = []

            for tool_name, tool_info in self._tool_infos.items():
                server_name = self._tool_to_server.get(tool_name, "")
                adapter = self._adapters.get(server_name)

                if not adapter:
                    continue

                # 构建工具描述
                description = ""
                if hasattr(tool_info, "description"):
                    description = tool_info.description or ""
                description += f"\n\n[MCP Server: {server_name}]"

                # 构建参数 Schema
                _parameters: dict[str, Any] = {}
                if hasattr(tool_info, "parameters"):
                    _parameters = (
                        tool_info.parameters if isinstance(tool_info.parameters, dict) else {}
                    )

                # 创建 LangChain StructuredTool
                try:

                    def _make_tool_func(tn: str, adp: BaseMCPAdapter) -> Any:
                        """创建工具调用函数，正确传递 kwargs 参数"""

                        def tool_func(**kwargs: Any) -> Any:
                            return adp.call_tool(tn, kwargs)

                        return tool_func

                    lc_tool = StructuredTool.from_function(
                        func=_make_tool_func(tool_name, adapter),
                        name=f"mcp_{tool_name}",
                        description=description,
                        args_schema=None,
                    )
                    langchain_tools.append(lc_tool)
                except Exception as exc:
                    logger.warning(
                        "MCP 工具 %s 转换为 LangChain 工具失败: %s",
                        tool_name,
                        exc,
                    )

            return langchain_tools

    def merge_to_tools_by_name(self, tools_by_name: dict[str, BaseTool]) -> dict[str, BaseTool]:
        """将 MCP 工具合并到现有 tools_by_name 映射表

        MCP 工具以 mcp_ 前缀注册，避免与现有工具名冲突。

        Args:
            tools_by_name: 现有的工具映射表

        Returns:
            合并后的工具映射表（新字典，不修改原字典）
        """
        merged = dict(tools_by_name)
        langchain_tools = self.to_langchain_tools()

        for tool in langchain_tools:
            if tool.name in merged:
                logger.warning("MCP 工具名与现有工具冲突: %s，MCP 工具被跳过", tool.name)
                continue
            merged[tool.name] = tool

        logger.info(
            "MCP 工具合并完成，原有 %d 个工具，新增 %d 个 MCP 工具",
            len(tools_by_name),
            len(merged) - len(tools_by_name),
        )
        return merged


__all__ = [
    "MCPRegistry",
]
