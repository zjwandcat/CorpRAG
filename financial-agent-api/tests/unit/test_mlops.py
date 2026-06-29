"""LLMExperimentTracker 单元测试

测试 app.mlops.tracking 模块的核心功能：
- 正常记录参数和指标
- MLflow 不可达时的异常降级
- 敏感字段过滤
- 追踪开关（enabled/disabled）
- trace_id 生成

所有测试使用 mock，不调用真实 MLflow 服务。
"""

import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.mlops.tracking import LLMExperimentTracker


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_mlflow() -> MagicMock:
    """Mock mlflow 模块"""
    mock = MagicMock()
    mock.start_run.return_value = MagicMock(info=MagicMock(run_id="test-run-id-123"))
    return mock


@pytest.fixture
def tracker_enabled(mock_mlflow: MagicMock) -> LLMExperimentTracker:
    """创建启用的追踪器（mock MLflow）"""
    with patch("app.mlops.tracking.mlflow", mock_mlflow):
        with patch("app.mlops.tracking._MLFLOW_AVAILABLE", True):
            tracker = LLMExperimentTracker(
                tracking_uri="http://localhost:5000",
                experiment_name="test-experiment",
                enabled=True,
                request_timeout=5,
            )
            tracker._mlflow_available = True
            yield tracker


@pytest.fixture
def tracker_disabled() -> LLMExperimentTracker:
    """创建禁用的追踪器"""
    tracker = LLMExperimentTracker(
        tracking_uri="http://localhost:5000",
        experiment_name="test-experiment",
        enabled=False,
        request_timeout=5,
    )
    return tracker


@pytest.fixture
def tracker_unavailable() -> LLMExperimentTracker:
    """创建 MLflow 不可用的追踪器"""
    tracker = LLMExperimentTracker(
        tracking_uri="http://invalid-host:5000",
        experiment_name="test-experiment",
        enabled=True,
        request_timeout=5,
    )
    tracker._mlflow_available = False
    return tracker


# ===========================================================================
# TestLLMExperimentTracker — 测试正常路径
# ===========================================================================


class TestLLMExperimentTracker:
    """LLMExperimentTracker 核心功能测试"""

    def test_start_run_success(self, tracker_enabled: LLMExperimentTracker, mock_mlflow: MagicMock) -> None:
        """测试正常启动 Run

        验证：
        - start_run 返回有效的 run_id
        - trace_id 被正确生成
        - mlflow.start_run 被调用
        """
        with patch("app.mlops.tracking.mlflow", mock_mlflow):
            run_id = tracker_enabled.start_run(run_name="test-run")

            assert run_id == "test-run-id-123"
            assert tracker_enabled._trace_id is not None
            mock_mlflow.start_run.assert_called_once_with(run_name="test-run")

    def test_log_params_success(self, tracker_enabled: LLMExperimentTracker, mock_mlflow: MagicMock) -> None:
        """测试正常记录参数

        验证：
        - mlflow.log_params 被调用
        - 参数被正确传递
        """
        tracker_enabled._trace_id = "test-trace-id"

        params = {
            "provider": "nim",
            "model": "deepseek-v4",
            "top_k": 3,
            "chunk_size": 512,
        }

        with patch("app.mlops.tracking.mlflow", mock_mlflow):
            tracker_enabled.log_params(params)

            mock_mlflow.log_params.assert_called_once()
            call_args = mock_mlflow.log_params.call_args[0][0]
            assert call_args["provider"] == "nim"
            assert call_args["model"] == "deepseek-v4"
            assert call_args["top_k"] == "3"

    def test_log_metrics_success(self, tracker_enabled: LLMExperimentTracker, mock_mlflow: MagicMock) -> None:
        """测试正常记录指标

        验证：
        - mlflow.log_metrics 被调用
        - 指标被正确传递
        """
        tracker_enabled._trace_id = "test-trace-id"

        metrics = {
            "retrieval_latency_ms": 120.5,
            "total_tokens": 1500,
            "reranker_score": 0.85,
        }

        with patch("app.mlops.tracking.mlflow", mock_mlflow):
            tracker_enabled.log_metrics(metrics)

            mock_mlflow.log_metrics.assert_called_once_with(metrics)

    def test_track_rag_run_success(self, tracker_enabled: LLMExperimentTracker, mock_mlflow: MagicMock) -> None:
        """测试一次性追踪完整流程

        验证：
        - start_run、log_params、log_metrics、end_run 被依次调用
        - 返回有效的 run_id
        """
        params = {"provider": "nim", "model": "deepseek-v4"}
        metrics = {"latency_ms": 100.0}

        with patch("app.mlops.tracking.mlflow", mock_mlflow):
            run_id = tracker_enabled.track_rag_run(params=params, metrics=metrics, run_name="test-rag-run")

            assert run_id == "test-run-id-123"
            mock_mlflow.start_run.assert_called()
            mock_mlflow.log_params.assert_called()
            mock_mlflow.log_metrics.assert_called()
            mock_mlflow.end_run.assert_called_once_with(status="FINISHED")

    def test_end_run_success(self, tracker_enabled: LLMExperimentTracker, mock_mlflow: MagicMock) -> None:
        """测试正常结束 Run

        验证：
        - mlflow.end_run 被调用
        - 状态正确传递
        """
        tracker_enabled._current_run_id = "test-run-id"
        tracker_enabled._trace_id = "test-trace-id"

        with patch("app.mlops.tracking.mlflow", mock_mlflow):
            tracker_enabled.end_run(status="FINISHED")

            mock_mlflow.end_run.assert_called_once_with(status="FINISHED")
            assert tracker_enabled._current_run_id is None
            assert tracker_enabled._trace_id is None


# ===========================================================================
# TestSensitiveFieldFiltering — 测试敏感字段过滤
# ===========================================================================


class TestSensitiveFieldFiltering:
    """敏感字段过滤测试"""

    def test_filter_api_key(self, tracker_enabled: LLMExperimentTracker) -> None:
        """测试过滤 api_key 字段"""
        params = {
            "provider": "nim",
            "api_key": "sk-secret-key-123",
            "model": "deepseek-v4",
        }

        sanitized = tracker_enabled._sanitize_params(params)

        assert "api_key" not in sanitized
        assert sanitized["provider"] == "nim"
        assert sanitized["model"] == "deepseek-v4"

    def test_filter_password(self, tracker_enabled: LLMExperimentTracker) -> None:
        """测试过滤 password 字段"""
        params = {
            "host": "localhost",
            "password": "my-secret-password",
            "port": "5432",
        }

        sanitized = tracker_enabled._sanitize_params(params)

        assert "password" not in sanitized
        assert sanitized["host"] == "localhost"
        assert sanitized["port"] == "5432"

    def test_filter_token(self, tracker_enabled: LLMExperimentTracker) -> None:
        """测试过滤 token 相关字段"""
        params = {
            "access_token": "token-123",
            "auth_token": "auth-456",
            "user": "admin",
        }

        sanitized = tracker_enabled._sanitize_params(params)

        assert "access_token" not in sanitized
        assert "auth_token" not in sanitized
        assert sanitized["user"] == "admin"

    def test_filter_secret(self, tracker_enabled: LLMExperimentTracker) -> None:
        """测试过滤 secret 字段"""
        params = {
            "secret": "my-secret",
            "credential": "my-credential",
            "private_key": "my-private-key",
        }

        sanitized = tracker_enabled._sanitize_params(params)

        assert "secret" not in sanitized
        assert "credential" not in sanitized
        assert "private_key" not in sanitized

    def test_filter_case_insensitive(self, tracker_enabled: LLMExperimentTracker) -> None:
        """测试不区分大小写过滤"""
        params = {
            "API_KEY": "key-1",
            "Apikey": "key-2",
            "PASSWORD": "pass-1",
            "Secret": "secret-1",
        }

        sanitized = tracker_enabled._sanitize_params(params)

        assert len(sanitized) == 0

    def test_preserve_non_sensitive(self, tracker_enabled: LLMExperimentTracker) -> None:
        """测试保留非敏感字段"""
        params = {
            "provider": "nim",
            "model_name": "deepseek-v4",
            "chunk_size": 512,
            "top_k": 3,
            "reranker": "cohere",
        }

        sanitized = tracker_enabled._sanitize_params(params)

        assert len(sanitized) == 5
        assert sanitized["provider"] == "nim"
        assert sanitized["chunk_size"] == "512"


# ===========================================================================
# TestTrackingDisabled — 测试追踪开关
# ===========================================================================


class TestTrackingDisabled:
    """追踪开关测试"""

    def test_disabled_start_run(self, tracker_disabled: LLMExperimentTracker) -> None:
        """测试禁用时 start_run 返回空字符串"""
        run_id = tracker_disabled.start_run(run_name="test-run")
        assert run_id == ""

    def test_disabled_log_params(self, tracker_disabled: LLMExperimentTracker) -> None:
        """测试禁用时 log_params 为 no-op"""
        params = {"provider": "nim", "model": "deepseek-v4"}
        # 不应抛出异常
        tracker_disabled.log_params(params)

    def test_disabled_log_metrics(self, tracker_disabled: LLMExperimentTracker) -> None:
        """测试禁用时 log_metrics 为 no-op"""
        metrics = {"latency_ms": 100.0}
        # 不应抛出异常
        tracker_disabled.log_metrics(metrics)

    def test_disabled_track_rag_run(self, tracker_disabled: LLMExperimentTracker) -> None:
        """测试禁用时 track_rag_run 返回空字符串"""
        params = {"provider": "nim"}
        metrics = {"latency_ms": 100.0}

        run_id = tracker_disabled.track_rag_run(params=params, metrics=metrics)
        assert run_id == ""

    def test_disabled_is_available(self, tracker_disabled: LLMExperimentTracker) -> None:
        """测试禁用时 is_available 返回 False"""
        assert tracker_disabled.is_available() is False


# ===========================================================================
# TestMLflowUnavailable — 测试 MLflow 不可用降级
# ===========================================================================


class TestMLflowUnavailable:
    """MLflow 不可用降级测试"""

    def test_unavailable_start_run(self, tracker_unavailable: LLMExperimentTracker) -> None:
        """测试 MLflow 不可用时 start_run 返回空字符串"""
        run_id = tracker_unavailable.start_run(run_name="test-run")
        assert run_id == ""

    def test_unavailable_log_params(self, tracker_unavailable: LLMExperimentTracker) -> None:
        """测试 MLflow 不可用时 log_params 为 no-op"""
        params = {"provider": "nim", "model": "deepseek-v4"}
        # 不应抛出异常
        tracker_unavailable.log_params(params)

    def test_unavailable_log_metrics(self, tracker_unavailable: LLMExperimentTracker) -> None:
        """测试 MLflow 不可用时 log_metrics 为 no-op"""
        metrics = {"latency_ms": 100.0}
        # 不应抛出异常
        tracker_unavailable.log_metrics(metrics)

    def test_unavailable_is_available(self, tracker_unavailable: LLMExperimentTracker) -> None:
        """测试 MLflow 不可用时 is_available 返回 False"""
        assert tracker_unavailable.is_available() is False

    def test_unavailable_track_rag_run(self, tracker_unavailable: LLMExperimentTracker) -> None:
        """测试 MLflow 不可用时 track_rag_run 返回空字符串"""
        params = {"provider": "nim"}
        metrics = {"latency_ms": 100.0}

        run_id = tracker_unavailable.track_rag_run(params=params, metrics=metrics)
        assert run_id == ""


# ===========================================================================
# TestTraceIdGeneration — 测试 trace_id 生成
# ===========================================================================


class TestTraceIdGeneration:
    """trace_id 生成测试"""

    def test_trace_id_generated_on_start_run(self, tracker_enabled: LLMExperimentTracker, mock_mlflow: MagicMock) -> None:
        """测试 start_run 时生成 trace_id"""
        with patch("app.mlops.tracking.mlflow", mock_mlflow):
            assert tracker_enabled._trace_id is None

            tracker_enabled.start_run(run_name="test-run")

            assert tracker_enabled._trace_id is not None
            # 验证是有效的 UUID 格式
            uuid.UUID(tracker_enabled._trace_id)  # 不抛出异常即通过

    def test_trace_id_unique_per_run(self, tracker_enabled: LLMExperimentTracker, mock_mlflow: MagicMock) -> None:
        """测试每次 start_run 生成不同的 trace_id"""
        with patch("app.mlops.tracking.mlflow", mock_mlflow):
            tracker_enabled.start_run(run_name="run-1")
            trace_id_1 = tracker_enabled._trace_id

            tracker_enabled.end_run()

            tracker_enabled.start_run(run_name="run-2")
            trace_id_2 = tracker_enabled._trace_id

            assert trace_id_1 != trace_id_2

    def test_trace_id_cleared_on_end_run(self, tracker_enabled: LLMExperimentTracker, mock_mlflow: MagicMock) -> None:
        """测试 end_run 时清除 trace_id"""
        with patch("app.mlops.tracking.mlflow", mock_mlflow):
            tracker_enabled.start_run(run_name="test-run")
            assert tracker_enabled._trace_id is not None

            tracker_enabled.end_run()
            assert tracker_enabled._trace_id is None


# ===========================================================================
# TestTimeoutHandling — 测试超时处理
# ===========================================================================


class TestTimeoutHandling:
    """超时处理测试"""

    def test_timeout_on_start_run(self, mock_mlflow: MagicMock) -> None:
        """测试 start_run 超时时降级处理"""
        # 模拟超时：start_run 永远阻塞
        mock_mlflow.start_run.side_effect = lambda *args, **kwargs: __import__("time").sleep(10)

        tracker = LLMExperimentTracker(
            tracking_uri="http://localhost:5000",
            experiment_name="test-experiment",
            enabled=True,
            request_timeout=1,  # 1 秒超时
        )
        tracker._mlflow_available = True

        with patch("app.mlops.tracking.mlflow", mock_mlflow):
            run_id = tracker.start_run(run_name="test-run")
            # 超时应返回空字符串
            assert run_id == ""

    def test_timeout_on_log_params(self, mock_mlflow: MagicMock) -> None:
        """测试 log_params 超时时降级处理"""
        mock_mlflow.log_params.side_effect = lambda *args, **kwargs: __import__("time").sleep(10)

        tracker = LLMExperimentTracker(
            tracking_uri="http://localhost:5000",
            experiment_name="test-experiment",
            enabled=True,
            request_timeout=1,
        )
        tracker._mlflow_available = True
        tracker._trace_id = "test-trace-id"

        with patch("app.mlops.tracking.mlflow", mock_mlflow):
            # 不应抛出异常
            tracker.log_params({"provider": "nim"})


# ===========================================================================
# TestLogTag — 测试标签记录
# ===========================================================================


class TestLogTag:
    """标签记录测试"""

    def test_log_tag_success(self, tracker_enabled: LLMExperimentTracker, mock_mlflow: MagicMock) -> None:
        """测试正常记录标签"""
        tracker_enabled._trace_id = "test-trace-id"

        with patch("app.mlops.tracking.mlflow", mock_mlflow):
            tracker_enabled.log_tag("environment", "production")

            mock_mlflow.set_tag.assert_called_once_with("environment", "production")

    def test_log_tag_disabled(self, tracker_disabled: LLMExperimentTracker) -> None:
        """测试禁用时 log_tag 为 no-op"""
        # 不应抛出异常
        tracker_disabled.log_tag("environment", "production")