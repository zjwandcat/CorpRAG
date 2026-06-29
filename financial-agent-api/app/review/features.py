"""功能开关管理器模块

实现 FeatureFlags 类，提供动态功能开关管理，
支持线程安全的热更新和配置持久化。
配置变更不影响进行中的审查任务。
"""

import copy
import logging
import threading

from app.review.settings import ReviewSettings

logger = logging.getLogger(__name__)


class FeatureFlags:
    """功能开关管理器

    提供动态功能开关查询和更新能力，支持线程安全的热更新。
    配置变更不影响进行中的审查任务（已创建会话使用创建时配置快照）。

    Attributes:
        _settings: 审查配置实例
        _lock: 线程安全锁（可重入）
    """

    def __init__(self, settings: ReviewSettings) -> None:
        """初始化功能开关管理器

        Args:
            settings: 审查配置实例
        """
        self._settings = settings
        self._lock = threading.RLock()

    def is_multi_agent_enabled(self) -> bool:
        """查询多Agent协作开关状态

        Returns:
            多Agent协作是否启用
        """
        with self._lock:
            return self._settings.multi_agent_enabled

    def is_mcp_enabled(self) -> bool:
        """查询 MCP 总开关状态

        Returns:
            MCP 功能是否启用
        """
        with self._lock:
            return self._settings.mcp_enabled

    def is_mcp_server_enabled(self, server_name: str) -> bool:
        """查询指定 MCP Server 的启用状态

        Args:
            server_name: MCP Server 名称

        Returns:
            该 MCP Server 是否启用；MCP 总开关关闭时始终返回 False
        """
        with self._lock:
            if not self._settings.mcp_enabled:
                return False
            return self._settings.mcp_servers.get(server_name, False)

    def get_enabled_review_types(self) -> list[str]:
        """获取启用的审查类型列表

        Returns:
            启用的审查类型列表的副本
        """
        with self._lock:
            return list(self._settings.review_types)

    def get_settings_snapshot(self) -> ReviewSettings:
        """获取当前配置的深拷贝快照

        用于审查会话创建时获取配置快照，确保配置变更
        不影响进行中的审查任务。

        Returns:
            当前配置的深拷贝
        """
        with self._lock:
            return copy.deepcopy(self._settings)

    def update(self, **kwargs: object) -> None:
        """线程安全的热更新配置

        更新配置后立即生效，无需重启服务。
        已创建的审查会话使用创建时的配置快照，不受影响。

        Args:
            **kwargs: 需要更新的配置项，支持的字段包括：
                - multi_agent_enabled: bool
                - mcp_enabled: bool
                - mcp_servers: dict[str, bool]
                - review_types: list[str]
                - worker_timeout_seconds: int
                - max_concurrent_reviews: int

        Raises:
            ValueError: 当更新后的配置值不在合法范围内时抛出
        """
        with self._lock:
            self._settings.update(**kwargs)
            logger.info("配置已热更新: %s", list(kwargs.keys()))

    def persist(self) -> None:
        """持久化配置到 JSON 文件

        将当前内存中的配置持久化到 review_config.json 文件。

        Raises:
            ConfigPersistenceError: 当配置持久化失败时抛出
        """
        with self._lock:
            self._settings.to_json()
            logger.info("配置已持久化")

    @property
    def settings(self) -> ReviewSettings:
        """获取当前配置实例（只读属性）

        注意：直接访问此属性获取的实例不是线程安全的。
        如需线程安全地读取配置，请使用对应的查询方法。
        """
        return self._settings


__all__ = ["FeatureFlags"]
