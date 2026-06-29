"""Filesystem MCP Server 适配器

提供文件读写工具：读取文件内容、列出目录。
通过 root_dir 配置项限制文件访问范围，防止路径遍历攻击。
"""

import logging
from pathlib import Path
from typing import Any

from app.exceptions import MCPToolCallError
from app.mcp.server import BaseMCPAdapter, MCPToolInfoProxy, validate_tool_definition_impl

logger = logging.getLogger(__name__)

# 文件类型白名单（仅允许写入这些扩展名的文件）
_ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".csv"}
)

# 单次写入大小限制（1MB）
_MAX_WRITE_SIZE_BYTES: int = 1 * 1024 * 1024

# 工具定义
_FS_READ_FILE_TOOL: dict[str, Any] = {
    "name": "filesystem_read_file",
    "description": (
        "读取指定文件的内容。仅允许访问 root_dir 目录下的文件。当需要查看代码文件内容时调用此工具。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径（相对于 root_dir 或绝对路径）",
            },
            "encoding": {
                "type": "string",
                "description": "文件编码，默认为 utf-8",
            },
        },
        "required": ["path"],
    },
}

_FS_LIST_DIRECTORY_TOOL: dict[str, Any] = {
    "name": "filesystem_list_directory",
    "description": (
        "列出指定目录下的文件和子目录。仅允许访问 root_dir 目录下的目录。"
        "当需要了解项目目录结构时调用此工具。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "目录路径（相对于 root_dir 或绝对路径）",
            },
            "recursive": {
                "type": "boolean",
                "description": "是否递归列出子目录，默认为 false",
            },
        },
        "required": ["path"],
    },
}

_FS_WRITE_FILE_TOOL: dict[str, Any] = {
    "name": "filesystem_write_file",
    "description": (
        "将内容写入指定文件。仅允许在 root_dir 目录下写入。当需要创建或修改文件时调用此工具。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径（相对于 root_dir 或绝对路径）",
            },
            "content": {
                "type": "string",
                "description": "要写入的文件内容",
            },
            "encoding": {
                "type": "string",
                "description": "文件编码，默认为 utf-8",
            },
        },
        "required": ["path", "content"],
    },
}


class FilesystemAdapter(BaseMCPAdapter):
    """文件系统 MCP Server 适配器

    提供文件读写工具，通过 root_dir 限制文件访问范围，防止路径遍历攻击。

    Attributes:
        root_dir: 文件访问的根目录，所有操作限制在此目录内
    """

    def __init__(self, root_dir: str | Path = ".") -> None:
        """初始化文件系统适配器

        Args:
            root_dir: 文件访问的根目录，默认为当前工作目录
        """
        self._root_dir = Path(root_dir).resolve()

    @property
    def server_name(self) -> str:
        """Server 名称"""
        return "filesystem"

    @property
    def root_dir(self) -> Path:
        """获取根目录路径"""
        return self._root_dir

    def _resolve_safe_path(self, path: str) -> Path:
        """安全解析路径，防止路径遍历攻击

        将相对路径解析为绝对路径，并验证其在 root_dir 内。

        Args:
            path: 输入路径

        Returns:
            解析后的安全路径

        Raises:
            MCPToolCallError: 路径超出 root_dir 范围
        """
        target = (self._root_dir / path).resolve()

        # 防止路径遍历攻击
        try:
            target.relative_to(self._root_dir)
        except ValueError:
            raise MCPToolCallError(
                message=f"路径超出允许范围: {path}（root_dir: {self._root_dir}）",
                tool_name="filesystem",
            )

        return target

    def list_tools(self) -> list[Any]:
        """列出文件系统相关工具定义

        Returns:
            工具信息列表
        """
        tools = [
            MCPToolInfoProxy(
                name=_FS_READ_FILE_TOOL["name"],
                description=_FS_READ_FILE_TOOL["description"],
                parameters=_FS_READ_FILE_TOOL["parameters"],
                server_name=self.server_name,
            ),
            MCPToolInfoProxy(
                name=_FS_LIST_DIRECTORY_TOOL["name"],
                description=_FS_LIST_DIRECTORY_TOOL["description"],
                parameters=_FS_LIST_DIRECTORY_TOOL["parameters"],
                server_name=self.server_name,
            ),
            MCPToolInfoProxy(
                name=_FS_WRITE_FILE_TOOL["name"],
                description=_FS_WRITE_FILE_TOOL["description"],
                parameters=_FS_WRITE_FILE_TOOL["parameters"],
                server_name=self.server_name,
            ),
        ]
        return tools

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """调用文件系统工具

        Args:
            tool_name: 工具名称
            arguments: 工具调用参数

        Returns:
            工具执行结果

        Raises:
            MCPToolCallError: 工具调用失败
        """
        if tool_name == "filesystem_read_file":
            return self._read_file(arguments)
        elif tool_name == "filesystem_list_directory":
            return self._list_directory(arguments)
        elif tool_name == "filesystem_write_file":
            return self._write_file(arguments)
        else:
            raise MCPToolCallError(
                message=f"未知的文件系统工具: {tool_name}",
                tool_name=tool_name,
            )

    def _read_file(self, arguments: dict[str, Any]) -> str:
        """读取文件内容

        Args:
            arguments: 包含 path 和可选 encoding 的参数字典

        Returns:
            文件内容字符串
        """
        path = arguments.get("path", "")
        encoding = arguments.get("encoding", "utf-8")

        if not path:
            raise MCPToolCallError(
                message="缺少必要参数: path",
                tool_name="filesystem_read_file",
            )

        safe_path = self._resolve_safe_path(path)

        if not safe_path.exists():
            raise MCPToolCallError(
                message=f"文件不存在: {path}",
                tool_name="filesystem_read_file",
            )

        if not safe_path.is_file():
            raise MCPToolCallError(
                message=f"路径不是文件: {path}",
                tool_name="filesystem_read_file",
            )

        try:
            content = safe_path.read_text(encoding=encoding)
            logger.info("读取文件成功: %s（%d 字符）", safe_path, len(content))
            return content
        except Exception as exc:
            raise MCPToolCallError(
                message=f"读取文件失败: {path}",
                tool_name="filesystem_read_file",
                details=str(exc),
            ) from exc

    def _list_directory(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """列出目录内容

        Args:
            arguments: 包含 path 和可选 recursive 的参数字典

        Returns:
            文件和目录信息列表
        """
        path = arguments.get("path", "")
        recursive = arguments.get("recursive", False)

        if not path:
            raise MCPToolCallError(
                message="缺少必要参数: path",
                tool_name="filesystem_list_directory",
            )

        safe_path = self._resolve_safe_path(path)

        if not safe_path.exists():
            raise MCPToolCallError(
                message=f"目录不存在: {path}",
                tool_name="filesystem_list_directory",
            )

        if not safe_path.is_dir():
            raise MCPToolCallError(
                message=f"路径不是目录: {path}",
                tool_name="filesystem_list_directory",
            )

        try:
            entries: list[dict[str, Any]] = []
            pattern = "**/*" if recursive else "*"

            for item in safe_path.glob(pattern):
                # 确保路径在 root_dir 内
                try:
                    item.relative_to(self._root_dir)
                except ValueError:
                    continue

                relative_path = str(item.relative_to(self._root_dir))
                entries.append(
                    {
                        "name": item.name,
                        "path": relative_path,
                        "type": "directory" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else 0,
                    }
                )

            logger.info("列出目录成功: %s（%d 个条目）", safe_path, len(entries))
            return entries

        except Exception as exc:
            if isinstance(exc, MCPToolCallError):
                raise
            raise MCPToolCallError(
                message=f"列出目录失败: {path}",
                tool_name="filesystem_list_directory",
                details=str(exc),
            ) from exc

    def _write_file(self, arguments: dict[str, Any]) -> str:
        """写入文件内容

        安全限制：
        - 仅允许写入白名单扩展名的文件
        - 单次写入大小不超过 1MB
        - 覆盖已有文件时记录 WARNING 日志

        Args:
            arguments: 包含 path、content 和可选 encoding 的参数字典

        Returns:
            操作结果消息
        """
        path = arguments.get("path", "")
        content = arguments.get("content", "")
        encoding = arguments.get("encoding", "utf-8")

        if not path:
            raise MCPToolCallError(
                message="缺少必要参数: path",
                tool_name="filesystem_write_file",
            )

        # 文件类型白名单校验
        file_ext = Path(path).suffix.lower()
        if file_ext and file_ext not in _ALLOWED_EXTENSIONS:
            raise MCPToolCallError(
                message=f"不允许写入此文件类型: {file_ext}（允许的类型: {', '.join(sorted(_ALLOWED_EXTENSIONS))}）",
                tool_name="filesystem_write_file",
            )

        # 写入大小限制校验
        content_size = len(content.encode(encoding))
        if content_size > _MAX_WRITE_SIZE_BYTES:
            raise MCPToolCallError(
                message=f"写入内容超过大小限制: {content_size} 字节（最大 {_MAX_WRITE_SIZE_BYTES} 字节）",
                tool_name="filesystem_write_file",
            )

        safe_path = self._resolve_safe_path(path)

        # 覆盖已有文件时记录警告
        if safe_path.exists():
            logger.warning(
                "覆盖已有文件: %s（大小: %d 字节）",
                safe_path,
                safe_path.stat().st_size if safe_path.is_file() else 0,
            )

        try:
            # 确保父目录存在
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            safe_path.write_text(content, encoding=encoding)
            logger.info("写入文件成功: %s（%d 字符）", safe_path, len(content))
            return f"文件已写入: {path}（{len(content)} 字符）"

        except Exception as exc:
            raise MCPToolCallError(
                message=f"写入文件失败: {path}",
                tool_name="filesystem_write_file",
                details=str(exc),
            ) from exc

    def health_check(self) -> bool:
        """检查文件系统可访问性

        Returns:
            True 表示 root_dir 目录存在且可访问
        """
        try:
            return self._root_dir.exists() and self._root_dir.is_dir()
        except Exception as exc:
            logger.warning("文件系统健康检查失败: %s", exc)
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
    "FilesystemAdapter",
]
