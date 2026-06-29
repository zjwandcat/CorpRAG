"""GitHub MCP Server 适配器

提供 PR 管理工具：获取 PR 信息、获取代码差异。
使用 httpx 调用 GitHub REST API，无需 PyGithub 依赖。
"""

import logging
from typing import Any

from app.exceptions import MCPToolCallError
from app.mcp.server import BaseMCPAdapter, MCPToolInfoProxy, validate_tool_definition_impl

logger = logging.getLogger(__name__)

# GitHub API 基础 URL
GITHUB_API_BASE_URL: str = "https://api.github.com"

# 工具定义
_GITHUB_GET_PR_INFO_TOOL: dict[str, Any] = {
    "name": "github_get_pr_info",
    "description": (
        "获取 GitHub Pull Request 的详细信息，包括标题、描述、状态、作者、分支等。"
        "当需要了解 PR 的基本信息时调用此工具。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "owner": {
                "type": "string",
                "description": "仓库所有者（用户名或组织名）",
            },
            "repo": {
                "type": "string",
                "description": "仓库名称",
            },
            "pr_number": {
                "type": "integer",
                "description": "Pull Request 编号",
            },
        },
        "required": ["owner", "repo", "pr_number"],
    },
}

_GITHUB_GET_PR_DIFF_TOOL: dict[str, Any] = {
    "name": "github_get_pr_diff",
    "description": (
        "获取 GitHub Pull Request 的代码差异（diff）。当需要审查 PR 的代码变更时调用此工具。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "owner": {
                "type": "string",
                "description": "仓库所有者（用户名或组织名）",
            },
            "repo": {
                "type": "string",
                "description": "仓库名称",
            },
            "pr_number": {
                "type": "integer",
                "description": "Pull Request 编号",
            },
        },
        "required": ["owner", "repo", "pr_number"],
    },
}

_GITHUB_LIST_PR_FILES_TOOL: dict[str, Any] = {
    "name": "github_list_pr_files",
    "description": (
        "获取 GitHub Pull Request 中变更的文件列表。当需要了解 PR 涉及了哪些文件时调用此工具。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "owner": {
                "type": "string",
                "description": "仓库所有者（用户名或组织名）",
            },
            "repo": {
                "type": "string",
                "description": "仓库名称",
            },
            "pr_number": {
                "type": "integer",
                "description": "Pull Request 编号",
            },
        },
        "required": ["owner", "repo", "pr_number"],
    },
}


class GitHubAdapter(BaseMCPAdapter):
    """GitHub MCP Server 适配器

    提供 PR 管理工具，使用 httpx 调用 GitHub REST API。

    Attributes:
        github_token: GitHub API Token（可选，有 Token 可提高速率限制）
        api_base_url: GitHub API 基础 URL
    """

    def __init__(
        self,
        github_token: str = "",
        api_base_url: str = GITHUB_API_BASE_URL,
    ) -> None:
        """初始化 GitHub 适配器

        Args:
            github_token: GitHub API Token（可选）
            api_base_url: GitHub API 基础 URL（默认为公共 API）
        """
        self._github_token = github_token
        self._api_base_url = api_base_url.rstrip("/")

    @property
    def server_name(self) -> str:
        """Server 名称"""
        return "github"

    def list_tools(self) -> list[Any]:
        """列出 GitHub 相关工具定义

        Returns:
            工具信息列表
        """
        tools = [
            MCPToolInfoProxy(
                name=_GITHUB_GET_PR_INFO_TOOL["name"],
                description=_GITHUB_GET_PR_INFO_TOOL["description"],
                parameters=_GITHUB_GET_PR_INFO_TOOL["parameters"],
                server_name=self.server_name,
            ),
            MCPToolInfoProxy(
                name=_GITHUB_GET_PR_DIFF_TOOL["name"],
                description=_GITHUB_GET_PR_DIFF_TOOL["description"],
                parameters=_GITHUB_GET_PR_DIFF_TOOL["parameters"],
                server_name=self.server_name,
            ),
            MCPToolInfoProxy(
                name=_GITHUB_LIST_PR_FILES_TOOL["name"],
                description=_GITHUB_LIST_PR_FILES_TOOL["description"],
                parameters=_GITHUB_LIST_PR_FILES_TOOL["parameters"],
                server_name=self.server_name,
            ),
        ]
        return tools

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """调用 GitHub 工具

        Args:
            tool_name: 工具名称
            arguments: 工具调用参数

        Returns:
            工具执行结果

        Raises:
            MCPToolCallError: 工具调用失败
        """
        if tool_name == "github_get_pr_info":
            return self._get_pr_info(arguments)
        elif tool_name == "github_get_pr_diff":
            return self._get_pr_diff(arguments)
        elif tool_name == "github_list_pr_files":
            return self._list_pr_files(arguments)
        else:
            raise MCPToolCallError(
                message=f"未知的 GitHub 工具: {tool_name}",
                tool_name=tool_name,
            )

    def _get_pr_info(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """获取 PR 详细信息

        Args:
            arguments: 包含 owner、repo、pr_number 的参数字典

        Returns:
            PR 信息字典
        """
        owner = arguments.get("owner", "")
        repo = arguments.get("repo", "")
        pr_number = arguments.get("pr_number", 0)

        if not owner or not repo or not pr_number:
            raise MCPToolCallError(
                message="缺少必要参数: owner, repo, pr_number",
                tool_name="github_get_pr_info",
            )

        url = f"{self._api_base_url}/repos/{owner}/{repo}/pulls/{pr_number}"
        return self._make_github_request(url, accept="application/vnd.github+json")

    def _get_pr_diff(self, arguments: dict[str, Any]) -> str:
        """获取 PR 代码差异

        Args:
            arguments: 包含 owner、repo、pr_number 的参数字典

        Returns:
            代码差异文本
        """
        owner = arguments.get("owner", "")
        repo = arguments.get("repo", "")
        pr_number = arguments.get("pr_number", 0)

        if not owner or not repo or not pr_number:
            raise MCPToolCallError(
                message="缺少必要参数: owner, repo, pr_number",
                tool_name="github_get_pr_diff",
            )

        url = f"{self._api_base_url}/repos/{owner}/{repo}/pulls/{pr_number}"
        result = self._make_github_request(url, accept="application/vnd.github.diff")
        return str(result)

    def _list_pr_files(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """获取 PR 变更文件列表

        Args:
            arguments: 包含 owner、repo、pr_number 的参数字典

        Returns:
            变更文件信息列表
        """
        owner = arguments.get("owner", "")
        repo = arguments.get("repo", "")
        pr_number = arguments.get("pr_number", 0)

        if not owner or not repo or not pr_number:
            raise MCPToolCallError(
                message="缺少必要参数: owner, repo, pr_number",
                tool_name="github_list_pr_files",
            )

        url = f"{self._api_base_url}/repos/{owner}/{repo}/pulls/{pr_number}/files"
        result = self._make_github_request(url, accept="application/vnd.github+json")
        return result if isinstance(result, list) else []

    def _make_github_request(self, url: str, accept: str = "application/vnd.github+json") -> Any:
        """发起 GitHub API 请求

        Args:
            url: 请求 URL
            accept: Accept 头

        Returns:
            API 响应数据

        Raises:
            MCPToolCallError: 请求失败
        """
        try:
            import httpx

            headers: dict[str, str] = {
                "Accept": accept,
                "X-GitHub-Api-Version": "2022-11-28",
            }
            if self._github_token:
                headers["Authorization"] = f"Bearer {self._github_token}"

            with httpx.Client(timeout=30.0) as client:
                response = client.get(url, headers=headers)
                response.raise_for_status()

                if accept.endswith(".diff"):
                    return response.text
                return response.json()

        except ImportError:
            logger.warning("httpx 未安装，GitHub API 调用不可用")
            raise MCPToolCallError(
                message="httpx 未安装，无法调用 GitHub API",
                tool_name="github",
                details="请安装 httpx: pip install httpx",
            )
        except Exception as exc:
            if isinstance(exc, MCPToolCallError):
                raise
            raise MCPToolCallError(
                message=f"GitHub API 请求失败: {url}",
                tool_name="github",
                details=str(exc),
            ) from exc

    def health_check(self) -> bool:
        """检查 GitHub API 可达性

        Returns:
            True 表示 GitHub API 可达
        """
        try:
            import httpx

            headers: dict[str, str] = {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            if self._github_token:
                headers["Authorization"] = f"Bearer {self._github_token}"

            with httpx.Client(timeout=10.0) as client:
                # 使用 rate_limit 端点检查可达性
                response = client.get(f"{self._api_base_url}/rate_limit", headers=headers)
                return response.status_code == 200

        except Exception as exc:
            logger.warning("GitHub API 健康检查失败: %s", exc)
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
    "GitHubAdapter",
    "GITHUB_API_BASE_URL",
]
