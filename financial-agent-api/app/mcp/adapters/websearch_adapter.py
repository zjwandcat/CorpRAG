"""WebSearch MCP Server 适配器

提供网络搜索工具：搜索技术文档和最佳实践。
复用现有 duckduckgo-search 实现。
"""

import logging
from typing import Any

from app.exceptions import MCPToolCallError
from app.mcp.server import BaseMCPAdapter, MCPToolInfoProxy, validate_tool_definition_impl

logger = logging.getLogger(__name__)

# 工具定义
_WS_SEARCH_TECH_DOCS_TOOL: dict[str, Any] = {
    "name": "websearch_search_tech_docs",
    "description": (
        "搜索技术文档和最佳实践。当需要查找安全漏洞修复方案、"
        "架构设计模式、性能优化技巧等技术资料时调用此工具。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词，例如 'Python SQL injection prevention best practices'",
            },
            "max_results": {
                "type": "integer",
                "description": "返回结果数量限制，默认为 5",
            },
        },
        "required": ["query"],
    },
}

_WS_SEARCH_CODE_EXAMPLES_TOOL: dict[str, Any] = {
    "name": "websearch_search_code_examples",
    "description": (
        "搜索代码示例和实现参考。当需要查找特定功能的代码实现方式、"
        "API 使用示例或框架用法时调用此工具。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词，例如 'FastAPI dependency injection example'",
            },
            "language": {
                "type": "string",
                "description": "编程语言筛选，例如 'python'、'javascript'",
            },
            "max_results": {
                "type": "integer",
                "description": "返回结果数量限制，默认为 3",
            },
        },
        "required": ["query"],
    },
}


class WebSearchAdapter(BaseMCPAdapter):
    """网络搜索 MCP Server 适配器

    提供网络搜索工具，复用现有 duckduckgo-search 实现。
    """

    @property
    def server_name(self) -> str:
        """Server 名称"""
        return "websearch"

    def list_tools(self) -> list[Any]:
        """列出搜索相关工具定义

        Returns:
            工具信息列表
        """
        tools = [
            MCPToolInfoProxy(
                name=_WS_SEARCH_TECH_DOCS_TOOL["name"],
                description=_WS_SEARCH_TECH_DOCS_TOOL["description"],
                parameters=_WS_SEARCH_TECH_DOCS_TOOL["parameters"],
                server_name=self.server_name,
            ),
            MCPToolInfoProxy(
                name=_WS_SEARCH_CODE_EXAMPLES_TOOL["name"],
                description=_WS_SEARCH_CODE_EXAMPLES_TOOL["description"],
                parameters=_WS_SEARCH_CODE_EXAMPLES_TOOL["parameters"],
                server_name=self.server_name,
            ),
        ]
        return tools

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """调用搜索工具

        Args:
            tool_name: 工具名称
            arguments: 工具调用参数

        Returns:
            工具执行结果

        Raises:
            MCPToolCallError: 工具调用失败
        """
        if tool_name == "websearch_search_tech_docs":
            return self._search_tech_docs(arguments)
        elif tool_name == "websearch_search_code_examples":
            return self._search_code_examples(arguments)
        else:
            raise MCPToolCallError(
                message=f"未知的搜索工具: {tool_name}",
                tool_name=tool_name,
            )

    def _search_tech_docs(self, arguments: dict[str, Any]) -> list[dict[str, str]]:
        """搜索技术文档

        Args:
            arguments: 包含 query 和可选 max_results 的参数字典

        Returns:
            搜索结果列表
        """
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 5)

        if not query:
            raise MCPToolCallError(
                message="缺少必要参数: query",
                tool_name="websearch_search_tech_docs",
            )

        return self._perform_search(query, max_results)

    def _search_code_examples(self, arguments: dict[str, Any]) -> list[dict[str, str]]:
        """搜索代码示例

        Args:
            arguments: 包含 query 和可选 language、max_results 的参数字典

        Returns:
            搜索结果列表
        """
        query = arguments.get("query", "")
        language = arguments.get("language", "")
        max_results = arguments.get("max_results", 3)

        if not query:
            raise MCPToolCallError(
                message="缺少必要参数: query",
                tool_name="websearch_search_code_examples",
            )

        # 如果指定了编程语言，添加到查询中
        search_query = f"{query} {language}" if language else query
        return self._perform_search(search_query, max_results)

    def _perform_search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        """执行 DuckDuckGo 搜索

        复用现有 duckduckgo-search 库实现搜索功能。

        Args:
            query: 搜索关键词
            max_results: 最大结果数

        Returns:
            搜索结果列表
        """
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))

            if not results:
                logger.info("搜索无结果: %s", query)
                return []

            formatted: list[dict[str, str]] = []
            for r in results:
                formatted.append(
                    {
                        "title": r.get("title", "无标题"),
                        "body": r.get("body", "无摘要"),
                        "href": r.get("href", ""),
                    }
                )

            logger.info("搜索完成: %s，返回 %d 条结果", query, len(formatted))
            return formatted

        except ImportError:
            logger.warning("duckduckgo-search 未安装，搜索功能不可用")
            raise MCPToolCallError(
                message="duckduckgo-search 未安装，无法执行搜索",
                tool_name="websearch",
                details="请安装 duckduckgo-search: pip install duckduckgo-search",
            )
        except Exception as exc:
            if isinstance(exc, MCPToolCallError):
                raise
            raise MCPToolCallError(
                message=f"搜索执行失败: {query}",
                tool_name="websearch",
                details=str(exc),
            ) from exc

    def health_check(self) -> bool:
        """检查搜索服务可用性

        仅检查 duckduckgo_search 模块是否可导入，
        不执行真实搜索请求以避免网络依赖。

        Returns:
            True 表示搜索模块可用
        """
        try:
            import duckduckgo_search  # noqa: F401

            return True
        except ImportError:
            logger.warning("duckduckgo-search 未安装，搜索服务不可用")
            return False

    def validate_tool_definition(self, tool: Any) -> bool:
        """工具定义安全校验

        Args:
            tool: 工具信息对象

        Returns:
            True 表示工具定义安全
        """
        return validate_tool_definition_impl(tool)


__all__ = [
    "WebSearchAdapter",
]
