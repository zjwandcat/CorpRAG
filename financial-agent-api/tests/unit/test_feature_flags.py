"""功能开关与配置管理单元测试

测试 FeatureFlags 和 ReviewSettings 的功能开关查询、热更新、持久化和校验逻辑。
"""

import json
import threading
from pathlib import Path

import pytest

from app.core.enums import MCPServerName
from app.review.features import FeatureFlags
from app.review.settings import ReviewSettings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def review_settings() -> ReviewSettings:
    """创建 ReviewSettings 实例（使用默认值）"""
    return ReviewSettings()


@pytest.fixture
def feature_flags(review_settings: ReviewSettings) -> FeatureFlags:
    """创建 FeatureFlags 实例"""
    return FeatureFlags(review_settings)


# ---------------------------------------------------------------------------
# FeatureFlags 测试
# ---------------------------------------------------------------------------


class TestFeatureFlags:
    """FeatureFlags 功能开关测试"""

    def test_is_multi_agent_enabled(self, feature_flags: FeatureFlags) -> None:
        """多Agent开关状态

        验证：is_multi_agent_enabled 返回正确的开关状态
        """
        # 默认值取决于环境变量，验证方法可正常调用
        result = feature_flags.is_multi_agent_enabled()
        assert isinstance(result, bool)

        # 更新后验证
        feature_flags.update(multi_agent_enabled=True)
        assert feature_flags.is_multi_agent_enabled() is True

        feature_flags.update(multi_agent_enabled=False)
        assert feature_flags.is_multi_agent_enabled() is False

    def test_hot_update(self, feature_flags: FeatureFlags) -> None:
        """配置热更新

        验证：update() 方法立即生效，无需重启服务
        """
        # 初始状态
        assert feature_flags.is_mcp_enabled() is False

        # 热更新
        feature_flags.update(mcp_enabled=True)
        assert feature_flags.is_mcp_enabled() is True

        # 再次更新
        feature_flags.update(mcp_enabled=False)
        assert feature_flags.is_mcp_enabled() is False

    def test_hot_update_thread_safety(self, feature_flags: FeatureFlags) -> None:
        """热更新的线程安全性

        验证：多线程并发更新不会导致数据竞争或异常
        """
        errors: list[Exception] = []

        def _update_worker(value: bool) -> None:
            try:
                for _ in range(50):
                    feature_flags.update(multi_agent_enabled=value)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=_update_worker, args=(True,)),
            threading.Thread(target=_update_worker, args=(False,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_persist_config(self, feature_flags: FeatureFlags, tmp_path: Path) -> None:
        """配置持久化

        验证：persist() 方法将配置写入 JSON 文件
        """
        config_file = tmp_path / "review_config.json"

        # 更新配置
        feature_flags.update(multi_agent_enabled=True, mcp_enabled=True)

        # 持久化
        feature_flags.settings.to_json(config_file)

        # 验证文件已创建且内容正确
        assert config_file.exists()
        data = json.loads(config_file.read_text(encoding="utf-8"))
        assert data["multi_agent_enabled"] is True
        assert data["mcp_enabled"] is True

    def test_persist_and_reload(self, feature_flags: FeatureFlags, tmp_path: Path) -> None:
        """配置持久化后重新加载

        验证：持久化后的配置可以被 ReviewSettings.from_json 正确加载
        """
        config_file = tmp_path / "review_config.json"

        # 更新并持久化
        feature_flags.update(
            multi_agent_enabled=True,
            worker_timeout_seconds=120,
        )
        feature_flags.settings.to_json(config_file)

        # 重新加载
        loaded = ReviewSettings.from_json(config_file)
        assert loaded.multi_agent_enabled is True
        assert loaded.worker_timeout_seconds == 120

    def test_is_mcp_server_enabled(self, feature_flags: FeatureFlags) -> None:
        """MCP Server 单独开关

        验证：MCP 总开关关闭时，所有 Server 开关返回 False
        """
        # MCP 总开关关闭
        feature_flags.update(mcp_enabled=False)
        assert feature_flags.is_mcp_server_enabled(MCPServerName.GITHUB) is False

        # MCP 总开关开启，但 GitHub Server 关闭
        feature_flags.update(
            mcp_enabled=True,
            mcp_servers={MCPServerName.GITHUB: False, MCPServerName.FILESYSTEM: True},
        )
        assert feature_flags.is_mcp_server_enabled(MCPServerName.GITHUB) is False
        assert feature_flags.is_mcp_server_enabled(MCPServerName.FILESYSTEM) is True

    def test_get_enabled_review_types(self, feature_flags: FeatureFlags) -> None:
        """获取启用的审查类型列表

        验证：get_enabled_review_types 返回正确的审查类型列表
        """
        feature_flags.update(review_types=["security", "performance"])
        types = feature_flags.get_enabled_review_types()
        assert "security" in types
        assert "performance" in types
        assert "architecture" not in types

    def test_get_settings_snapshot(self, feature_flags: FeatureFlags) -> None:
        """获取配置快照

        验证：get_settings_snapshot 返回配置的深拷贝，
        修改快照不影响原始配置
        """
        snapshot = feature_flags.get_settings_snapshot()
        # 修改快照
        snapshot.multi_agent_enabled = True
        # 原始配置不受影响
        assert feature_flags.is_multi_agent_enabled() is False


# ---------------------------------------------------------------------------
# ReviewSettings 测试
# ---------------------------------------------------------------------------


class TestReviewSettings:
    """ReviewSettings 配置测试"""

    def test_default_values(self) -> None:
        """默认配置值

        验证：ReviewSettings 的默认值符合预期
        """
        settings = ReviewSettings()
        assert isinstance(settings.multi_agent_enabled, bool)
        assert isinstance(settings.mcp_enabled, bool)
        assert isinstance(settings.mcp_servers, dict)
        assert isinstance(settings.review_types, list)
        assert settings.worker_timeout_seconds == 60
        assert settings.max_concurrent_reviews == 10

    def test_validation_worker_timeout(self) -> None:
        """配置值校验 — worker_timeout_seconds 范围

        验证：worker_timeout_seconds 必须在 [10, 300] 范围内
        """
        # 合法值
        settings = ReviewSettings()
        settings.worker_timeout_seconds = 10
        settings._validate()
        settings.worker_timeout_seconds = 300
        settings._validate()

        # 非法值
        with pytest.raises(ValueError, match="worker_timeout_seconds"):
            settings.worker_timeout_seconds = 9
            settings._validate()

        with pytest.raises(ValueError, match="worker_timeout_seconds"):
            settings.worker_timeout_seconds = 301
            settings._validate()

    def test_validation_max_concurrent_reviews(self) -> None:
        """配置值校验 — max_concurrent_reviews 范围

        验证：max_concurrent_reviews 必须在 [1, 100] 范围内
        """
        settings = ReviewSettings()

        # 合法值
        settings.max_concurrent_reviews = 1
        settings._validate()
        settings.max_concurrent_reviews = 100
        settings._validate()

        # 非法值
        with pytest.raises(ValueError, match="max_concurrent_reviews"):
            settings.max_concurrent_reviews = 0
            settings._validate()

        with pytest.raises(ValueError, match="max_concurrent_reviews"):
            settings.max_concurrent_reviews = 101
            settings._validate()

    def test_from_json_missing_file(self, tmp_path: Path) -> None:
        """从 JSON 加载 — 文件不存在

        验证：文件不存在时使用默认配置
        """
        config_file = tmp_path / "nonexistent.json"
        settings = ReviewSettings.from_json(config_file)
        assert settings.worker_timeout_seconds == 60

    def test_from_json_empty_file(self, tmp_path: Path) -> None:
        """从 JSON 加载 — 空文件

        验证：空文件时使用默认配置
        """
        config_file = tmp_path / "empty.json"
        config_file.write_text("", encoding="utf-8")
        settings = ReviewSettings.from_json(config_file)
        assert settings.worker_timeout_seconds == 60

    def test_from_json_corrupted_file(self, tmp_path: Path) -> None:
        """从 JSON 加载 — 损坏文件

        验证：损坏的 JSON 文件时使用默认配置
        """
        config_file = tmp_path / "corrupted.json"
        config_file.write_text("{invalid json", encoding="utf-8")
        settings = ReviewSettings.from_json(config_file)
        assert settings.worker_timeout_seconds == 60

    def test_to_json_and_reload(self, tmp_path: Path) -> None:
        """持久化后重新加载

        验证：配置持久化后可被正确加载
        """
        config_file = tmp_path / "test_config.json"
        settings = ReviewSettings()
        settings.multi_agent_enabled = True
        settings.mcp_enabled = True
        settings.worker_timeout_seconds = 120
        settings.to_json(config_file)

        loaded = ReviewSettings.from_json(config_file)
        assert loaded.multi_agent_enabled is True
        assert loaded.mcp_enabled is True
        assert loaded.worker_timeout_seconds == 120

    def test_update_method(self) -> None:
        """update() 方法更新配置

        验证：update() 方法正确更新指定字段
        """
        settings = ReviewSettings()
        settings.update(multi_agent_enabled=True, worker_timeout_seconds=90)
        assert settings.multi_agent_enabled is True
        assert settings.worker_timeout_seconds == 90

    def test_update_with_none_ignored(self) -> None:
        """update() 方法忽略 None 值

        验证：update() 中传入 None 的字段不会被更新
        """
        settings = ReviewSettings()
        original_timeout = settings.worker_timeout_seconds
        settings.update(multi_agent_enabled=True, worker_timeout_seconds=None)
        assert settings.multi_agent_enabled is True
        assert settings.worker_timeout_seconds == original_timeout
