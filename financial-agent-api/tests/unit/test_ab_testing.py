"""ABTestRouter 单元测试

测试 app.mlops.ab_testing 模块的核心功能：
- 基于 session_id 的确定性分桶
- 分桶分布验证
- 动态配置更新
- 指标记录和持久化
- 异常降级

所有测试使用 mock，不调用真实外部服务。
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.core.enums import ABBucket, RAGStrategy
from app.mlops.ab_testing import (
    ABTestConfig,
    ABTestMetric,
    ABTestRouter,
    BucketMetrics,
)
from app.models.schemas import ABConfigRequest, ABConfigResponse, ABMetricsResponse


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def router_enabled() -> ABTestRouter:
    """创建启用的 A/B 测试路由器"""
    return ABTestRouter(
        bucket_a_ratio=0.5,
        bucket_a_strategy=RAGStrategy.SELF_HOSTED_RAG,
        bucket_b_strategy=RAGStrategy.BLUEPRINT_RAG,
        enabled=True,
    )


@pytest.fixture
def router_disabled() -> ABTestRouter:
    """创建禁用的 A/B 测试路由器"""
    return ABTestRouter(
        bucket_a_ratio=0.5,
        bucket_a_strategy=RAGStrategy.SELF_HOSTED_RAG,
        bucket_b_strategy=RAGStrategy.BLUEPRINT_RAG,
        enabled=False,
    )


@pytest.fixture
def router_with_metrics_file() -> ABTestRouter:
    """创建带指标持久化的路由器"""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        metrics_file = Path(f.name)

    router = ABTestRouter(
        bucket_a_ratio=0.5,
        bucket_a_strategy=RAGStrategy.SELF_HOSTED_RAG,
        bucket_b_strategy=RAGStrategy.BLUEPRINT_RAG,
        enabled=True,
        metrics_file=metrics_file,
    )

    yield router

    # 清理
    if metrics_file.exists():
        metrics_file.unlink()


@pytest.fixture
def router_70_30_split() -> ABTestRouter:
    """创建 70/30 分流的路由器"""
    return ABTestRouter(
        bucket_a_ratio=0.7,
        bucket_a_strategy=RAGStrategy.SELF_HOSTED_RAG,
        bucket_b_strategy=RAGStrategy.BLUEPRINT_RAG,
        enabled=True,
    )


# ===========================================================================
# TestDeterministicBucketing — 测试确定性分桶
# ===========================================================================


class TestDeterministicBucketing:
    """确定性分桶测试"""

    def test_same_session_same_bucket(self, router_enabled: ABTestRouter) -> None:
        """测试同一 session_id 始终分配到同一 Bucket

        验证：
        - 多次调用 get_bucket 返回相同结果
        """
        session_id = "test-session-123"

        bucket_1 = router_enabled.get_bucket(session_id)
        bucket_2 = router_enabled.get_bucket(session_id)
        bucket_3 = router_enabled.get_bucket(session_id)

        assert bucket_1 == bucket_2 == bucket_3

    def test_different_sessions_may_differ(self, router_enabled: ABTestRouter) -> None:
        """测试不同 session_id 可能分配到不同 Bucket"""
        buckets = set()

        for i in range(100):
            session_id = f"session-{i}"
            bucket = router_enabled.get_bucket(session_id)
            buckets.add(bucket)

        # 100 个 session 应该覆盖两个 Bucket
        assert len(buckets) == 2

    def test_bucket_is_valid_enum(self, router_enabled: ABTestRouter) -> None:
        """测试返回的 Bucket 是有效枚举值"""
        session_id = "test-session-456"
        bucket = router_enabled.get_bucket(session_id)

        assert bucket in (ABBucket.BUCKET_A, ABBucket.BUCKET_B)

    def test_deterministic_with_known_session_ids(self, router_enabled: ABTestRouter) -> None:
        """测试已知 session_id 的分桶结果确定性"""
        # 使用固定的 session_id，验证分桶结果稳定
        known_sessions = [
            "user-001",
            "user-002",
            "user-003",
            "user-004",
            "user-005",
        ]

        results = []
        for session_id in known_sessions:
            bucket = router_enabled.get_bucket(session_id)
            results.append((session_id, bucket))

        # 再次执行，结果应完全相同
        for session_id, expected_bucket in results:
            actual_bucket = router_enabled.get_bucket(session_id)
            assert actual_bucket == expected_bucket


# ===========================================================================
# TestBucketDistribution — 测试分桶分布
# ===========================================================================


class TestBucketDistribution:
    """分桶分布验证测试"""

    def test_50_50_distribution(self, router_enabled: ABTestRouter) -> None:
        """测试 50/50 分流分布

        验证：
        - Bucket A 和 B 的比例接近 50/50
        """
        bucket_a_count = 0
        bucket_b_count = 0
        total = 1000

        for i in range(total):
            session_id = f"session-{i}"
            bucket = router_enabled.get_bucket(session_id)
            if bucket == ABBucket.BUCKET_A:
                bucket_a_count += 1
            else:
                bucket_b_count += 1

        # 允许 5% 的误差
        ratio_a = bucket_a_count / total
        assert 0.45 <= ratio_a <= 0.55

    def test_70_30_distribution(self, router_70_30_split: ABTestRouter) -> None:
        """测试 70/30 分流分布"""
        bucket_a_count = 0
        total = 1000

        for i in range(total):
            session_id = f"session-{i}"
            bucket = router_70_30_split.get_bucket(session_id)
            if bucket == ABBucket.BUCKET_A:
                bucket_a_count += 1

        ratio_a = bucket_a_count / total
        # 允许 5% 的误差
        assert 0.65 <= ratio_a <= 0.75

    def test_extreme_100_0_distribution(self) -> None:
        """测试 100/0 极端分流"""
        router = ABTestRouter(
            bucket_a_ratio=1.0,  # 100% 到 Bucket A
            enabled=True,
        )

        for i in range(100):
            session_id = f"session-{i}"
            bucket = router.get_bucket(session_id)
            assert bucket == ABBucket.BUCKET_A


# ===========================================================================
# TestStrategyRetrieval — 测试策略获取
# ===========================================================================


class TestStrategyRetrieval:
    """策略获取测试"""

    def test_get_strategy_bucket_a(self, router_enabled: ABTestRouter) -> None:
        """测试获取 Bucket A 的策略"""
        strategy = router_enabled.get_strategy(ABBucket.BUCKET_A)
        assert strategy == RAGStrategy.SELF_HOSTED_RAG

    def test_get_strategy_bucket_b(self, router_enabled: ABTestRouter) -> None:
        """测试获取 Bucket B 的策略"""
        strategy = router_enabled.get_strategy(ABBucket.BUCKET_B)
        assert strategy == RAGStrategy.BLUEPRINT_RAG

    def test_strategy_consistency(self, router_enabled: ABTestRouter) -> None:
        """测试策略一致性"""
        # 多次获取应返回相同策略
        for _ in range(10):
            assert router_enabled.get_strategy(ABBucket.BUCKET_A) == RAGStrategy.SELF_HOSTED_RAG
            assert router_enabled.get_strategy(ABBucket.BUCKET_B) == RAGStrategy.BLUEPRINT_RAG


# ===========================================================================
# TestDynamicConfigUpdate — 测试动态配置更新
# ===========================================================================


class TestDynamicConfigUpdate:
    """动态配置更新测试"""

    def test_update_bucket_a_ratio(self, router_enabled: ABTestRouter) -> None:
        """测试更新 Bucket A 比例"""
        # 初始比例 0.5
        config = router_enabled.get_config()
        assert config.bucket_a_ratio == 0.5

        # 更新为 0.7
        update_request = ABConfigRequest(bucket_a_ratio=0.7)
        router_enabled.update_config(update_request)

        # 验证更新成功
        updated_config = router_enabled.get_config()
        assert updated_config.bucket_a_ratio == 0.7

    def test_update_strategy(self, router_enabled: ABTestRouter) -> None:
        """测试更新策略"""
        update_request = ABConfigRequest(
            bucket_a_strategy=RAGStrategy.BLUEPRINT_RAG.value,
            bucket_b_strategy=RAGStrategy.SELF_HOSTED_RAG.value,
        )
        router_enabled.update_config(update_request)

        assert router_enabled.get_strategy(ABBucket.BUCKET_A) == RAGStrategy.BLUEPRINT_RAG
        assert router_enabled.get_strategy(ABBucket.BUCKET_B) == RAGStrategy.SELF_HOSTED_RAG

    def test_update_enabled(self, router_enabled: ABTestRouter) -> None:
        """测试更新启用状态"""
        update_request = ABConfigRequest(enabled=False)
        router_enabled.update_config(update_request)

        assert router_enabled.is_enabled() is False

    def test_update_partial_config(self, router_enabled: ABTestRouter) -> None:
        """测试部分更新配置

        验证：
        - 只更新指定字段
        - 其他字段保持不变
        """
        original_config = router_enabled.get_config()

        # 只更新 bucket_a_ratio
        update_request = ABConfigRequest(bucket_a_ratio=0.8)
        router_enabled.update_config(update_request)

        updated_config = router_enabled.get_config()
        assert updated_config.bucket_a_ratio == 0.8
        # 其他字段应保持不变
        assert updated_config.bucket_a_strategy == original_config.bucket_a_strategy
        assert updated_config.bucket_b_strategy == original_config.bucket_b_strategy

    def test_update_invalid_ratio_raises_error(self, router_enabled: ABTestRouter) -> None:
        """测试更新无效比例抛出错误"""
        # 比例 > 1.0 - Pydantic 验证会先抛出错误
        with pytest.raises(Exception):  # Pydantic ValidationError
            router_enabled.update_config(ABConfigRequest(bucket_a_ratio=1.5))

        # 比例 < 0.0 - Pydantic 验证会先抛出错误
        with pytest.raises(Exception):  # Pydantic ValidationError
            router_enabled.update_config(ABConfigRequest(bucket_a_ratio=-0.1))

    def test_update_invalid_strategy_raises_error(self, router_enabled: ABTestRouter) -> None:
        """测试更新无效策略抛出错误"""
        with pytest.raises(ValueError, match="非法的 RAG 策略"):
            router_enabled.update_config(ABConfigRequest(bucket_a_strategy="invalid_strategy"))


# ===========================================================================
# TestMetricsRecording — 测试指标记录
# ===========================================================================


class TestMetricsRecording:
    """指标记录测试"""

    def test_record_metrics_bucket_a(self, router_enabled: ABTestRouter) -> None:
        """测试记录 Bucket A 指标"""
        router_enabled.record_metrics(
            bucket=ABBucket.BUCKET_A,
            latency_ms=100,
            tokens=500,
            feedback="positive",
        )

        metrics = router_enabled.get_bucket_metrics_detail(ABBucket.BUCKET_A)
        assert metrics.total_requests == 1
        assert metrics.total_latency_ms == 100
        assert metrics.total_tokens == 500
        assert metrics.feedback_positive == 1

    def test_record_metrics_bucket_b(self, router_enabled: ABTestRouter) -> None:
        """测试记录 Bucket B 指标"""
        router_enabled.record_metrics(
            bucket=ABBucket.BUCKET_B,
            latency_ms=150,
            tokens=600,
            feedback="negative",
        )

        metrics = router_enabled.get_bucket_metrics_detail(ABBucket.BUCKET_B)
        assert metrics.total_requests == 1
        assert metrics.total_latency_ms == 150
        assert metrics.total_tokens == 600
        assert metrics.feedback_negative == 1

    def test_record_multiple_metrics(self, router_enabled: ABTestRouter) -> None:
        """测试记录多条指标"""
        for i in range(10):
            router_enabled.record_metrics(
                bucket=ABBucket.BUCKET_A,
                latency_ms=100 + i * 10,
                tokens=500 + i * 50,
            )

        metrics = router_enabled.get_bucket_metrics_detail(ABBucket.BUCKET_A)
        assert metrics.total_requests == 10
        assert metrics.total_latency_ms == sum(100 + i * 10 for i in range(10))
        assert metrics.total_tokens == sum(500 + i * 50 for i in range(10))

    def test_get_metrics_avg_latency(self, router_enabled: ABTestRouter) -> None:
        """测试获取平均延迟"""
        # Bucket A: 3 次请求，延迟分别为 100, 200, 300
        router_enabled.record_metrics(ABBucket.BUCKET_A, 100, 100)
        router_enabled.record_metrics(ABBucket.BUCKET_A, 200, 100)
        router_enabled.record_metrics(ABBucket.BUCKET_A, 300, 100)

        # Bucket B: 2 次请求，延迟分别为 150, 250
        router_enabled.record_metrics(ABBucket.BUCKET_B, 150, 100)
        router_enabled.record_metrics(ABBucket.BUCKET_B, 250, 100)

        metrics = router_enabled.get_metrics()

        # Bucket A 平均延迟 = (100 + 200 + 300) / 3 = 200
        assert metrics.bucket_a_avg_latency_ms == 200.0
        # Bucket B 平均延迟 = (150 + 250) / 2 = 200
        assert metrics.bucket_b_avg_latency_ms == 200.0
        assert metrics.bucket_a_total_requests == 3
        assert metrics.bucket_b_total_requests == 2

    def test_feedback_counting(self, router_enabled: ABTestRouter) -> None:
        """测试反馈计数"""
        router_enabled.record_metrics(ABBucket.BUCKET_A, 100, 100, feedback="positive")
        router_enabled.record_metrics(ABBucket.BUCKET_A, 100, 100, feedback="positive")
        router_enabled.record_metrics(ABBucket.BUCKET_A, 100, 100, feedback="negative")
        router_enabled.record_metrics(ABBucket.BUCKET_A, 100, 100)  # 无反馈

        metrics = router_enabled.get_bucket_metrics_detail(ABBucket.BUCKET_A)
        assert metrics.feedback_positive == 2
        assert metrics.feedback_negative == 1


# ===========================================================================
# TestMetricsPersistence — 测试指标持久化
# ===========================================================================


class TestMetricsPersistence:
    """指标持久化测试"""

    def test_save_metrics_to_file(self, router_with_metrics_file: ABTestRouter) -> None:
        """测试保存指标到文件"""
        router = router_with_metrics_file

        router.record_metrics(ABBucket.BUCKET_A, 100, 500, "positive")
        router.record_metrics(ABBucket.BUCKET_B, 150, 600, "negative")

        # 读取文件验证
        metrics_file = router._metrics_file
        assert metrics_file.exists()

        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["bucket_a"]["total_requests"] == 1
        assert data["bucket_a"]["total_latency_ms"] == 100
        assert data["bucket_b"]["total_requests"] == 1
        assert "last_updated" in data

    def test_load_metrics_from_file(self) -> None:
        """测试从文件加载历史指标"""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            historical_data = {
                "bucket_a": {
                    "total_requests": 10,
                    "total_latency_ms": 1000.0,
                    "total_tokens": 5000,
                    "feedback_positive": 5,
                    "feedback_negative": 2,
                },
                "bucket_b": {
                    "total_requests": 5,
                    "total_latency_ms": 750.0,
                    "total_tokens": 3000,
                    "feedback_positive": 3,
                    "feedback_negative": 1,
                },
                "last_updated": "2024-01-01T00:00:00",
            }
            json.dump(historical_data, f)
            metrics_file = Path(f.name)

        router = ABTestRouter(
            bucket_a_ratio=0.5,
            enabled=True,
            metrics_file=metrics_file,
        )

        metrics_a = router.get_bucket_metrics_detail(ABBucket.BUCKET_A)
        assert metrics_a.total_requests == 10
        assert metrics_a.total_latency_ms == 1000.0
        assert metrics_a.feedback_positive == 5

        metrics_b = router.get_bucket_metrics_detail(ABBucket.BUCKET_B)
        assert metrics_b.total_requests == 5

        # 清理
        metrics_file.unlink()

    def test_reset_metrics(self, router_with_metrics_file: ABTestRouter) -> None:
        """测试重置指标"""
        router = router_with_metrics_file

        router.record_metrics(ABBucket.BUCKET_A, 100, 500)
        router.record_metrics(ABBucket.BUCKET_B, 150, 600)

        # 重置
        router.reset_metrics()

        metrics_a = router.get_bucket_metrics_detail(ABBucket.BUCKET_A)
        metrics_b = router.get_bucket_metrics_detail(ABBucket.BUCKET_B)

        assert metrics_a.total_requests == 0
        assert metrics_b.total_requests == 0


# ===========================================================================
# TestExceptionDegradation — 测试异常降级
# ===========================================================================


class TestExceptionDegradation:
    """异常降级测试"""

    def test_invalid_session_id_fallback_to_bucket_a(self, router_enabled: ABTestRouter) -> None:
        """测试无效 session_id 降级到 Bucket A"""
        # 空字符串 session_id
        bucket = router_enabled.get_bucket("")
        assert bucket == ABBucket.BUCKET_A

    def test_exception_in_hash_fallback_to_bucket_a(self, router_enabled: ABTestRouter) -> None:
        """测试哈希异常时降级到 Bucket A"""
        # 使用 mock 模拟哈希异常
        with patch("hashlib.md5") as mock_md5:
            mock_md5.side_effect = Exception("Hash error")

            bucket = router_enabled.get_bucket("test-session")
            assert bucket == ABBucket.BUCKET_A

    def test_metrics_file_write_failure(self) -> None:
        """测试指标文件写入失败时优雅降级"""
        # 使用无效路径
        invalid_path = Path("/nonexistent/path/metrics.json")

        router = ABTestRouter(
            bucket_a_ratio=0.5,
            enabled=True,
            metrics_file=invalid_path,
        )

        # 不应抛出异常
        router.record_metrics(ABBucket.BUCKET_A, 100, 500)


# ===========================================================================
# TestConfigRetrieval — 测试配置获取
# ===========================================================================


class TestConfigRetrieval:
    """配置获取测试"""

    def test_get_config(self, router_enabled: ABTestRouter) -> None:
        """测试获取当前配置"""
        config = router_enabled.get_config()

        assert isinstance(config, ABConfigResponse)
        assert config.bucket_a_ratio == 0.5
        assert config.bucket_a_strategy == RAGStrategy.SELF_HOSTED_RAG
        assert config.bucket_b_strategy == RAGStrategy.BLUEPRINT_RAG
        assert config.enabled is True

    def test_is_enabled(self, router_enabled: ABTestRouter) -> None:
        """测试检查启用状态"""
        assert router_enabled.is_enabled() is True

    def test_is_disabled(self, router_disabled: ABTestRouter) -> None:
        """测试检查禁用状态"""
        assert router_disabled.is_enabled() is False


# ===========================================================================
# TestABTestConfig — 测试配置数据结构
# ===========================================================================


class TestABTestConfig:
    """ABTestConfig 数据结构测试"""

    def test_default_config(self) -> None:
        """测试默认配置值"""
        config = ABTestConfig()

        assert config.bucket_a_ratio == 0.5
        assert config.bucket_a_strategy == RAGStrategy.SELF_HOSTED_RAG
        assert config.bucket_b_strategy == RAGStrategy.BLUEPRINT_RAG
        assert config.enabled is True


# ===========================================================================
# TestABTestMetric — 测试指标数据结构
# ===========================================================================


class TestABTestMetric:
    """ABTestMetric 数据结构测试"""

    def test_metric_fields(self) -> None:
        """测试指标字段"""
        metric = ABTestMetric(
            bucket=ABBucket.BUCKET_A,
            response_latency_ms=100.5,
            total_tokens=500,
            user_feedback="positive",
        )

        assert metric.bucket == ABBucket.BUCKET_A
        assert metric.response_latency_ms == 100.5
        assert metric.total_tokens == 500
        assert metric.user_feedback == "positive"
        assert metric.timestamp is not None


# ===========================================================================
# TestBucketMetrics — 测试 Bucket 指标数据结构
# ===========================================================================


class TestBucketMetrics:
    """BucketMetrics 数据结构测试"""

    def test_default_metrics(self) -> None:
        """测试默认指标值"""
        metrics = BucketMetrics()

        assert metrics.total_requests == 0
        assert metrics.total_latency_ms == 0.0
        assert metrics.total_tokens == 0
        assert metrics.feedback_positive == 0
        assert metrics.feedback_negative == 0

    def test_metrics_accumulation(self) -> None:
        """测试指标累积"""
        metrics = BucketMetrics()

        metrics.total_requests += 1
        metrics.total_latency_ms += 100.0
        metrics.total_tokens += 500
        metrics.feedback_positive += 1

        assert metrics.total_requests == 1
        assert metrics.total_latency_ms == 100.0
        assert metrics.feedback_positive == 1


# ===========================================================================
# TestThreadSafety — 测试线程安全
# ===========================================================================


class TestThreadSafety:
    """线程安全测试"""

    def test_concurrent_bucket_assignment(self, router_enabled: ABTestRouter) -> None:
        """测试并发分桶分配

        验证：
        - 多线程环境下分桶结果一致
        """
        import threading

        results = {}
        errors = []

        def assign_bucket(session_id: str) -> None:
            try:
                bucket = router_enabled.get_bucket(session_id)
                results[session_id] = bucket
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(100):
            session_id = f"session-{i}"
            thread = threading.Thread(target=assign_bucket, args=(session_id,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0
        assert len(results) == 100

    def test_concurrent_metrics_recording(self, router_enabled: ABTestRouter) -> None:
        """测试并发指标记录"""
        import threading

        errors = []

        def record_metrics() -> None:
            try:
                for _ in range(10):
                    router_enabled.record_metrics(
                        ABBucket.BUCKET_A,
                        latency_ms=100,
                        tokens=500,
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_metrics) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(errors) == 0

        metrics = router_enabled.get_bucket_metrics_detail(ABBucket.BUCKET_A)
        assert metrics.total_requests == 100  # 10 threads * 10 records