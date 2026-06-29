"""自定义异常模块

定义项目中使用的所有自定义异常类，提供清晰的错误层级和上下文信息。
所有异常继承自 AgentError 基类。
"""


class AgentError(Exception):
    """智能体基础异常

    所有自定义异常的基类，提供统一的异常接口。

    Attributes:
        message: 错误消息
        details: 额外的错误详情（可选）
    """

    def __init__(self, message: str, details: str | None = None) -> None:
        """初始化异常

        Args:
            message: 错误消息
            details: 额外的错误详情（可选）
        """
        self.message = message
        self.details = details
        super().__init__(message)

    def __str__(self) -> str:
        """返回完整的错误信息"""
        if self.details:
            return f"{self.message} - 详情: {self.details}"
        return self.message


class ConfigurationError(AgentError):
    """配置错误

    当配置文件缺失、格式错误或配置项无效时抛出。
    """

    pass


class DocumentLoadError(AgentError):
    """文档加载错误

    当文档加载失败时抛出。
    """

    pass


class UnsupportedFormatError(DocumentLoadError):
    """不支持的文档格式错误

    当尝试加载不支持的文档格式时抛出。

    Attributes:
        format: 不支持的格式
    """

    def __init__(
        self,
        message: str = "不支持的文档格式",
        format: str | None = None,
        details: str | None = None,
    ) -> None:
        """初始化异常

        Args:
            message: 错误消息
            format: 不支持的格式
            details: 额外的错误详情
        """
        self.format = format
        if format:
            message = f"{message}: {format}"
        super().__init__(message, details)


class VectorStoreError(AgentError):
    """向量数据库错误

    当向量数据库操作失败时抛出。
    """

    pass


class LLMInvocationError(AgentError):
    """大模型调用错误

    当大模型调用失败时抛出。
    """

    pass


class RateLimitExceededError(LLMInvocationError):
    """请求频率超限错误

    当 API 请求频率超过限制时抛出。

    Attributes:
        retry_after: 建议的重试等待时间（秒）
    """

    def __init__(
        self,
        message: str = "请求频率超限，请稍后重试",
        retry_after: int | None = None,
        details: str | None = None,
    ) -> None:
        """初始化异常

        Args:
            message: 错误消息
            retry_after: 建议的重试等待时间（秒）
            details: 额外的错误详情
        """
        self.retry_after = retry_after
        if retry_after:
            message = f"{message}，建议等待 {retry_after} 秒后重试"
        super().__init__(message, details)


class ToolExecutionError(AgentError):
    """工具执行错误

    当工具执行失败时抛出。
    """

    pass


class ServiceConnectionError(AgentError):
    """服务连接错误

    当无法连接到外部服务时抛出。

    Attributes:
        service_name: 服务名称
        endpoint: 服务端点
    """

    def __init__(
        self,
        message: str = "无法连接到服务",
        service_name: str | None = None,
        endpoint: str | None = None,
        details: str | None = None,
    ) -> None:
        """初始化异常

        Args:
            message: 错误消息
            service_name: 服务名称
            endpoint: 服务端点
            details: 额外的错误详情
        """
        self.service_name = service_name
        self.endpoint = endpoint
        if service_name:
            message = f"{message}: {service_name}"
        if endpoint:
            message = f"{message} ({endpoint})"
        super().__init__(message, details)


class ValidationError(AgentError):
    """数据验证错误

    当数据验证失败时抛出。
    """

    pass


# =============================================================================
# 审查相关异常
# =============================================================================


class ReviewError(AgentError):
    """审查基础异常

    当代码审查过程中发生错误时抛出。
    """

    pass


class WorkerTimeoutError(ReviewError):
    """Worker 执行超时异常

    当 Worker Agent 在配置的超时时间内未返回结果时抛出。

    Attributes:
        dimension: 超时的审查维度
    """

    def __init__(
        self,
        message: str = "Worker 执行超时",
        dimension: str | None = None,
        details: str | None = None,
    ) -> None:
        """初始化异常

        Args:
            message: 错误消息
            dimension: 超时的审查维度
            details: 额外的错误详情
        """
        self.dimension = dimension
        if dimension:
            message = f"{message}: {dimension}"
        super().__init__(message, details)


class WorkerExecutionError(ReviewError):
    """Worker 执行失败异常

    当 Worker Agent 在执行过程中抛出异常时抛出。

    Attributes:
        dimension: 失败的审查维度
    """

    def __init__(
        self,
        message: str = "Worker 执行失败",
        dimension: str | None = None,
        details: str | None = None,
    ) -> None:
        """初始化异常

        Args:
            message: 错误消息
            dimension: 失败的审查维度
            details: 额外的错误详情
        """
        self.dimension = dimension
        if dimension:
            message = f"{message}: {dimension}"
        super().__init__(message, details)


class SupervisorDispatchError(ReviewError):
    """Supervisor 调度失败异常

    当 Supervisor Agent 无法正确分发任务时抛出。
    """

    pass


class MCPConnectionError(ReviewError):
    """MCP Server 连接失败异常

    当 MCP Server 不可达或返回连接错误时抛出。

    Attributes:
        server_name: 连接失败的 MCP Server 名称
    """

    def __init__(
        self,
        message: str = "MCP Server 连接失败",
        server_name: str | None = None,
        details: str | None = None,
    ) -> None:
        """初始化异常

        Args:
            message: 错误消息
            server_name: 连接失败的 MCP Server 名称
            details: 额外的错误详情
        """
        self.server_name = server_name
        if server_name:
            message = f"{message}: {server_name}"
        super().__init__(message, details)


class MCPToolCallError(ReviewError):
    """MCP 工具调用失败异常

    当 MCP 工具调用执行失败时抛出。

    Attributes:
        tool_name: 调用失败的工具名称
    """

    def __init__(
        self,
        message: str = "MCP 工具调用失败",
        tool_name: str | None = None,
        details: str | None = None,
    ) -> None:
        """初始化异常

        Args:
            message: 错误消息
            tool_name: 调用失败的工具名称
            details: 额外的错误详情
        """
        self.tool_name = tool_name
        if tool_name:
            message = f"{message}: {tool_name}"
        super().__init__(message, details)


class MCPToolCallTimeoutError(MCPToolCallError):
    """MCP 工具调用超时异常

    当 MCP 工具调用在配置的超时时间内未返回结果时抛出。
    """

    def __init__(
        self,
        message: str = "MCP 工具调用超时",
        tool_name: str | None = None,
        details: str | None = None,
    ) -> None:
        """初始化异常

        Args:
            message: 错误消息
            tool_name: 超时的工具名称
            details: 额外的错误详情
        """
        super().__init__(message, tool_name, details)


class MCPAuthenticationError(ReviewError):
    """MCP Server 认证失败异常

    当 MCP Server 要求认证但提供的凭据无效时抛出。

    Attributes:
        server_name: 认证失败的 MCP Server 名称
    """

    def __init__(
        self,
        message: str = "MCP Server 认证失败",
        server_name: str | None = None,
        details: str | None = None,
    ) -> None:
        """初始化异常

        Args:
            message: 错误消息
            server_name: 认证失败的 MCP Server 名称
            details: 额外的错误详情
        """
        self.server_name = server_name
        if server_name:
            message = f"{message}: {server_name}"
        super().__init__(message, details)


class MCPToolNotAllowedError(ReviewError):
    """MCP 工具调用未授权异常

    当未经授权的 Agent 尝试调用受限制的 MCP 工具时抛出。

    Attributes:
        tool_name: 未授权的工具名称
    """

    def __init__(
        self,
        message: str = "MCP 工具调用未授权",
        tool_name: str | None = None,
        details: str | None = None,
    ) -> None:
        """初始化异常

        Args:
            message: 错误消息
            tool_name: 未授权的工具名称
            details: 额外的错误详情
        """
        self.tool_name = tool_name
        if tool_name:
            message = f"{message}: {tool_name}"
        super().__init__(message, details)


class ConfigPersistenceError(ReviewError):
    """配置持久化失败异常

    当配置存储介质不可用（如磁盘满、权限不足）时抛出。
    """

    pass


class InfiniteLoopDetectedError(AgentError):
    """检测到工具调用死循环异常

    当同一工具被连续调用超过阈值且参数完全一致时抛出。

    Attributes:
        tool_name: 触发死循环的工具名称
        repetition_count: 重复调用次数
    """

    def __init__(
        self,
        tool_name: str | None = None,
        repetition_count: int = 0,
        details: str = "",
    ) -> None:
        """初始化异常

        Args:
            tool_name: 触发死循环的工具名称
            repetition_count: 重复调用次数
            details: 额外的错误详情
        """
        self.tool_name = tool_name
        self.repetition_count = repetition_count
        msg = "检测到工具调用死循环"
        if tool_name:
            msg = f"检测到工具调用死循环：{tool_name}（重复 {repetition_count} 次）"
        if details:
            msg = f"{msg} — {details}"
        super().__init__(msg)


# 公开 API
__all__ = [
    "AgentError",
    "ConfigPersistenceError",
    "ConfigurationError",
    "DocumentLoadError",
    "InfiniteLoopDetectedError",
    "LLMInvocationError",
    "MCPAuthenticationError",
    "MCPConnectionError",
    "MCPToolCallError",
    "MCPToolCallTimeoutError",
    "MCPToolNotAllowedError",
    "RateLimitExceededError",
    "ReviewError",
    "ServiceConnectionError",
    "SupervisorDispatchError",
    "ToolExecutionError",
    "UnsupportedFormatError",
    "ValidationError",
    "VectorStoreError",
    "WorkerExecutionError",
    "WorkerTimeoutError",
]
