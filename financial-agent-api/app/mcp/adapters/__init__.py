"""MCP Server 适配器模块

提供内置的 MCP Server 适配器实现，包括：
- GitHubAdapter: GitHub PR 管理和代码差异获取
- FilesystemAdapter: 文件读写操作
- DatabaseAdapter: 数据库查询操作
- WebSearchAdapter: 网络搜索操作
"""

from app.mcp.adapters.github_adapter import GitHubAdapter
from app.mcp.adapters.filesystem_adapter import FilesystemAdapter
from app.mcp.adapters.database_adapter import DatabaseAdapter
from app.mcp.adapters.websearch_adapter import WebSearchAdapter

__all__ = [
    "DatabaseAdapter",
    "FilesystemAdapter",
    "GitHubAdapter",
    "WebSearchAdapter",
]
