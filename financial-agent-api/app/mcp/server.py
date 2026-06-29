"""MCP Server 抽象基类模块

定义 MCP (Model Context Protocol) Server 适配器的抽象接口，
所有内置 MCP Server 适配器必须继承 BaseMCPAdapter 并实现其抽象方法。

安全校验规则：
- 拒绝包含 exec/eval/subprocess 危险参数的工具
- 拒绝参数类型为 code/command 的工具
- 限制工具参数的最大嵌套深度
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

# 安全校验常量
MAX_NESTING_DEPTH: int = 5
DANGEROUS_PARAM_NAMES: frozenset[str] = frozenset(
    {"exec", "eval", "subprocess", "system", "shell", "os_command", "runtime"}
)
DANGEROUS_PARAM_TYPES: frozenset[str] = frozenset({"code", "command", "script", "expression"})


def _check_nesting_depth(schema: dict[str, Any], current_depth: int = 0) -> bool:
    """递归检查 JSON Schema 的嵌套深度是否超过限制

    Args:
        schema: JSON Schema 字典
        current_depth: 当前嵌套深度

    Returns:
        True 如果嵌套深度在允许范围内，False 如果超过限制
    """
    if current_depth > MAX_NESTING_DEPTH:
        return False

    # 检查 properties 嵌套
    properties = schema.get("properties", {})
    for prop_schema in properties.values():
        if isinstance(prop_schema, dict):
            if not _check_nesting_depth(prop_schema, current_depth + 1):
                return False

    # 检查 items 嵌套（数组类型）
    items = schema.get("items")
    if isinstance(items, dict):
        if not _check_nesting_depth(items, current_depth + 1):
            return False

    # 检查 additionalProperties 嵌套
    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        if not _check_nesting_depth(additional, current_depth + 1):
            return False

    # 检查 anyOf/oneOf/allOf 嵌套
    for key in ("anyOf", "oneOf", "allOf"):
        variants = schema.get(key, [])
        if isinstance(variants, list):
            for variant in variants:
                if isinstance(variant, dict):
                    if not _check_nesting_depth(variant, current_depth + 1):
                        return False

    return True


def _check_dangerous_params(parameters: dict[str, Any]) -> tuple[bool, str]:
    """检查工具参数定义中是否包含危险参数

    Args:
        parameters: 工具参数的 JSON Schema 定义

    Returns:
        (is_safe, reason) 元组，is_safe 为 True 表示安全
    """
    properties = parameters.get("properties", {})

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue

        # 检查参数名是否为危险名称
        param_name_lower = param_name.lower()
        for dangerous_name in DANGEROUS_PARAM_NAMES:
            if dangerous_name in param_name_lower:
                return False, f"参数名包含危险关键字: {param_name}"

        # 检查参数类型是否为危险类型
        param_type = param_schema.get("type", "").lower()
        if param_type in DANGEROUS_PARAM_TYPES:
            return False, f"参数类型为危险类型: {param_name} (type={param_type})"

        # 检查 description 中是否包含危险关键词
        description = param_schema.get("description", "").lower()
        dangerous_keywords = ["exec(", "eval(", "subprocess", "os.system", "shell=True"]
        for keyword in dangerous_keywords:
            if keyword in description:
                return False, f"参数描述包含危险关键字: {param_name} ({keyword})"

    return True, ""


class MCPToolInfoProxy:
    """MCP 工具信息代理类

    当 app.models.schemas 中的 MCPToolInfo 不可用时，
    使用此代理类作为工具信息的载体。各适配器统一引用此类，
    避免在多个适配器中重复定义。

    Attributes:
        name: 工具名称
        description: 工具描述
        parameters: 工具参数定义（JSON Schema 格式）
        server_name: 所属 MCP Server 名称
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        server_name: str,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.server_name = server_name

    def __repr__(self) -> str:
        return f"MCPToolInfoProxy(name={self.name!r}, server_name={self.server_name!r})"


class BaseMCPAdapter(ABC):
    """MCP Server 适配器抽象基类

    所有 MCP Server 适配器必须继承此类并实现以下抽象方法和属性：
    - server_name: Server 名称标识
    - list_tools(): 列出该 Server 提供的工具
    - call_tool(): 调用该 Server 的工具
    - health_check(): 健康检查
    - validate_tool_definition(): 工具定义安全校验
    """

    @property
    @abstractmethod
    def server_name(self) -> str:
        """Server 名称标识

        Returns:
            Server 的唯一名称字符串
        """

    @abstractmethod
    def list_tools(self) -> list[Any]:
        """列出该 Server 提供的工具

        Returns:
            MCPToolInfo 对象列表，每个对象包含工具名称、描述、参数定义和所属 Server
        """

    @abstractmethod
    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """调用该 Server 的工具

        Args:
            tool_name: 工具名称
            arguments: 工具调用参数字典

        Returns:
            工具执行结果

        Raises:
            MCPToolCallError: 工具调用失败
            MCPToolCallTimeoutError: 工具调用超时
        """

    @abstractmethod
    def health_check(self) -> bool:
        """健康检查

        Returns:
            True 表示 Server 健康，False 表示不可用
        """

    @abstractmethod
    def validate_tool_definition(self, tool: Any) -> bool:
        """校验工具定义安全性

        安全校验规则：
        1. 拒绝包含 exec/eval/subprocess 危险参数的工具
        2. 拒绝参数类型为 code/command 的工具
        3. 限制工具参数的最大嵌套深度

        Args:
            tool: 工具信息对象（MCPToolInfo）

        Returns:
            True 表示工具定义安全，False 表示不安全
        """


def validate_tool_definition_impl(tool: Any) -> bool:
    """工具定义安全校验的通用实现

    可被各适配器复用的安全校验逻辑。

    Args:
        tool: 工具信息对象，需具有 name、description、parameters 属性

    Returns:
        True 表示工具定义安全，False 表示不安全
    """
    # 获取工具参数定义
    parameters: dict[str, Any] = {}
    if hasattr(tool, "parameters"):
        parameters = tool.parameters if isinstance(tool.parameters, dict) else {}
    elif hasattr(tool, "inputSchema"):
        parameters = tool.inputSchema if isinstance(tool.inputSchema, dict) else {}

    if not parameters:
        # 没有参数定义的工具视为安全
        return True

    # 检查危险参数
    is_safe, reason = _check_dangerous_params(parameters)
    if not is_safe:
        tool_name = getattr(tool, "name", "unknown")
        logger.warning("工具定义安全校验失败 - %s: %s", tool_name, reason)
        return False

    # 检查嵌套深度
    if not _check_nesting_depth(parameters):
        tool_name = getattr(tool, "name", "unknown")
        logger.warning(
            "工具定义安全校验失败 - %s: 参数嵌套深度超过限制 (%d)",
            tool_name,
            MAX_NESTING_DEPTH,
        )
        return False

    return True


__all__ = [
    "BaseMCPAdapter",
    "DANGEROUS_PARAM_NAMES",
    "DANGEROUS_PARAM_TYPES",
    "MAX_NESTING_DEPTH",
    "MCPToolInfoProxy",
    "validate_tool_definition_impl",
]
