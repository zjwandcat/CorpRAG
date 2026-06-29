"""审查配置模块

实现 ReviewSettings dataclass，支持从 Settings 初始化默认值、
JSON 文件持久化、配置值校验，以及配置加载失败时使用默认配置启动。
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.core.config import settings
from app.core.enums import MCPServerName, ReviewType
from app.exceptions import ConfigPersistenceError

logger = logging.getLogger(__name__)


def _default_mcp_servers() -> dict[str, bool]:
    """生成默认 MCP Server 启用状态字典"""
    return {
        MCPServerName.GITHUB: False,
        MCPServerName.FILESYSTEM: False,
        MCPServerName.DATABASE: False,
        MCPServerName.WEBSEARCH: False,
    }


def _default_review_types() -> list[str]:
    """生成默认审查类型列表"""
    return [ReviewType.FULL]


@dataclass(slots=True)
class ReviewSettings:
    """代码审查配置

    管理多Agent协作、MCP服务器、审查类型等配置项。
    支持从 Settings（环境变量级别）初始化默认值，
    支持 JSON 文件持久化，配置加载失败时使用默认配置启动。

    Attributes:
        multi_agent_enabled: 多Agent协作开关
        mcp_enabled: MCP 总开关
        mcp_servers: 各 MCP Server 启用状态
        review_types: 启用的审查类型列表
        worker_timeout_seconds: Worker 超时时间（秒），范围 [10, 300]
        max_concurrent_reviews: 最大并发审查数，范围 [1, 100]
    """

    multi_agent_enabled: bool = False
    mcp_enabled: bool = False
    mcp_servers: dict[str, bool] = field(default_factory=_default_mcp_servers)
    review_types: list[str] = field(default_factory=_default_review_types)
    worker_timeout_seconds: int = 60
    max_concurrent_reviews: int = 10

    def __post_init__(self) -> None:
        """从 Settings 初始化默认值并校验配置"""
        self._init_from_settings()
        self._validate()

    def _init_from_settings(self) -> None:
        """从 Settings（环境变量级别）初始化默认值

        仅在当前值为默认值时使用 Settings 中的值覆盖，
        确保从 JSON 文件加载的配置不被环境变量覆盖。
        """
        self.multi_agent_enabled = settings.MULTI_AGENT_ENABLED
        self.mcp_enabled = settings.MCP_ENABLED
        self.mcp_servers = {
            MCPServerName.GITHUB: settings.MCP_GITHUB_ENABLED,
            MCPServerName.FILESYSTEM: settings.MCP_FILESYSTEM_ENABLED,
            MCPServerName.DATABASE: settings.MCP_DATABASE_ENABLED,
            MCPServerName.WEBSEARCH: settings.MCP_WEBSEARCH_ENABLED,
        }
        self.worker_timeout_seconds = settings.WORKER_TIMEOUT_SECONDS
        self.max_concurrent_reviews = settings.MAX_CONCURRENT_REVIEWS

    def _validate(self) -> None:
        """校验配置值

        Raises:
            ValueError: 当配置值不在合法范围内时抛出
        """
        if not (10 <= self.worker_timeout_seconds <= 300):
            raise ValueError(
                f"worker_timeout_seconds 必须在 [10, 300] 范围内，"
                f"当前值: {self.worker_timeout_seconds}"
            )
        if not (1 <= self.max_concurrent_reviews <= 100):
            raise ValueError(
                f"max_concurrent_reviews 必须在 [1, 100] 范围内，"
                f"当前值: {self.max_concurrent_reviews}"
            )

    @classmethod
    def from_json(cls, file_path: Path | None = None) -> "ReviewSettings":
        """从 JSON 文件加载配置

        配置加载失败时使用默认配置启动，记录错误日志。

        Args:
            file_path: JSON 文件路径，默认使用 Settings 中的 REVIEW_CONFIG_FILE

        Returns:
            加载后的 ReviewSettings 实例
        """
        config_file = file_path or settings.REVIEW_CONFIG_FILE
        instance = cls()

        if not config_file.exists():
            logger.info("配置文件 %s 不存在，使用默认配置", config_file)
            return instance

        try:
            content = config_file.read_text(encoding="utf-8")
            if not content.strip():
                logger.info("配置文件 %s 为空，使用默认配置", config_file)
                return instance

            data = json.loads(content)
            if not isinstance(data, dict):
                logger.warning("配置文件 %s 格式错误（非字典），使用默认配置", config_file)
                return instance

            # 逐字段更新，忽略未知字段
            if "multi_agent_enabled" in data:
                instance.multi_agent_enabled = bool(data["multi_agent_enabled"])
            if "mcp_enabled" in data:
                instance.mcp_enabled = bool(data["mcp_enabled"])
            if "mcp_servers" in data and isinstance(data["mcp_servers"], dict):
                instance.mcp_servers = {k: bool(v) for k, v in data["mcp_servers"].items()}
            if "review_types" in data and isinstance(data["review_types"], list):
                instance.review_types = [str(t) for t in data["review_types"]]
            if "worker_timeout_seconds" in data:
                instance.worker_timeout_seconds = int(data["worker_timeout_seconds"])
            if "max_concurrent_reviews" in data:
                instance.max_concurrent_reviews = int(data["max_concurrent_reviews"])

            # 校验加载的配置值
            instance._validate()
            logger.info("从 %s 加载配置成功", config_file)

        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.error("配置文件 %s 加载失败，使用默认配置: %s", config_file, exc)
            # 重新创建默认实例
            return cls()
        except OSError as exc:
            logger.error("配置文件 %s 读取失败，使用默认配置: %s", config_file, exc)
            return cls()

        return instance

    def to_json(self, file_path: Path | None = None) -> None:
        """持久化配置到 JSON 文件

        采用原子写入模式：先写入临时文件，再通过 rename 替换目标文件，
        确保进程崩溃时不会导致配置文件损坏。

        Args:
            file_path: JSON 文件路径，默认使用 Settings 中的 REVIEW_CONFIG_FILE

        Raises:
            ConfigPersistenceError: 当配置持久化失败时抛出
        """
        config_file = file_path or settings.REVIEW_CONFIG_FILE

        try:
            # 确保父目录存在
            config_file.parent.mkdir(parents=True, exist_ok=True)

            # 序列化当前配置
            data = asdict(self)
            json_str = json.dumps(data, ensure_ascii=False, indent=2)

            # 原子写入：先写入临时文件，再 rename 替换
            tmp_path = config_file.with_suffix(".tmp")
            tmp_path.write_text(json_str, encoding="utf-8")
            tmp_path.replace(config_file)

            logger.info("配置已原子写入到 %s", config_file)

        except OSError as exc:
            raise ConfigPersistenceError(
                message=f"配置持久化失败: {config_file}",
                details=str(exc),
            ) from exc

    def update(self, **kwargs: object) -> None:
        """更新配置值

        Args:
            **kwargs: 需要更新的配置项

        Raises:
            ValueError: 当更新后的配置值不在合法范围内时抛出
        """
        if "multi_agent_enabled" in kwargs and kwargs["multi_agent_enabled"] is not None:
            self.multi_agent_enabled = bool(kwargs["multi_agent_enabled"])
        if "mcp_enabled" in kwargs and kwargs["mcp_enabled"] is not None:
            self.mcp_enabled = bool(kwargs["mcp_enabled"])
        if "mcp_servers" in kwargs and kwargs["mcp_servers"] is not None:
            self.mcp_servers = {k: bool(v) for k, v in kwargs["mcp_servers"].items()}  # type: ignore[union-attr]
        if "review_types" in kwargs and kwargs["review_types"] is not None:
            self.review_types = [str(t) for t in kwargs["review_types"]]  # type: ignore[union-attr]
        if "worker_timeout_seconds" in kwargs and kwargs["worker_timeout_seconds"] is not None:
            self.worker_timeout_seconds = int(kwargs["worker_timeout_seconds"])
        if "max_concurrent_reviews" in kwargs and kwargs["max_concurrent_reviews"] is not None:
            self.max_concurrent_reviews = int(kwargs["max_concurrent_reviews"])

        self._validate()


__all__ = ["ReviewSettings"]
