"""MCP 协议模块

实现 MCP (Model Context Protocol) 客户端、工具注册表和 Server 适配器，
支持动态工具发现、调用和结果返回，与现有 LangChain 工具系统兼容。

核心组件：
- BaseMCPAdapter: MCP Server 适配器抽象基类
- MCPRegistry: 工具注册表，维护工具名到 Server/Adapter 的映射
- MCPClient: MCP 客户端，管理多 MCP Server 连接和工具调用
"""

from app.mcp.server import BaseMCPAdapter

__all__ = [
    "BaseMCPAdapter",
]
