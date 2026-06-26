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


# 公开 API
__all__ = [
    "AgentError",
    "ConfigurationError",
    "DocumentLoadError",
    "LLMInvocationError",
    "RateLimitExceededError",
    "ServiceConnectionError",
    "ToolExecutionError",
    "UnsupportedFormatError",
    "ValidationError",
    "VectorStoreError",
]
