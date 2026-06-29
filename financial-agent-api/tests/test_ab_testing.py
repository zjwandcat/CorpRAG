"""ABTestRouter 单元测试

测试 A/B 测试路由器的核心功能：
- 哈希分桶的确定性一致性
- 动态配置更新
- 指标记录和持久化
- 异常降级
"""

import tempfile
from pathlib import Path

import pytest

from app.core.enums import ABBucket, RAGStrategy
from app.mlops.ab_testing import ABTestRouter
from app.models.schemas import ABConfigRequest


class TestABTestRouter:
    """ABTestRouter 测试类"""

    def test_init_default_config(self) -> None:
        """测试默认配置初始化"""
        router = ABTestRouter()

        assert router.is_enabled() is False
        config = router.get_config()
        assert config.bucket_a_ratio == 0.5
        assert config.bucket_a_strategy == RAGStrategy.SELF_HOSTED_RAG
        assert config.bucket_b_strategy == RAGStrategy.BLUEPRINT_RAG

    def test_init_custom_config(self) -> None:
        """测试自定义配置初始化"""
        router = ABTestRouter(
            bucket_a_ratio=0.7,
            bucket_a_strategy=RAGStrategy.BLUEPRINT_RAG,
            bucket_b_strategy=RAGStrategy.SELF_HOSTED_RAG,
            enabled=True,
        )

        assert router.is_enabled() is True
        config = router.get_config()
        assert config.bucket_a_ratio == 0.7
        assert config.bucket_a_strategy == RAGStrategy.BLUEPRINT_RAG
        assert config.bucket_b_strategy == RAGStrategy.SELF_HOSTED_RAG

    def test_get_bucket_deterministic(self) -> None:
        """测试分桶的确定性：同一 session_id 始终分配到同一 Bucket"""
        router = ABTestRouter(bucket_a_ratio=0.5, enabled=True)

        session_id = "test-session-12345"

        # 多次分桶应返回相同结果
        bucket1 = router.get_bucket(session_id)
        bucket2 = router.get_bucket(session_id)
        bucket3 = router.get_bucket(session_id)

        assert bucket1 == bucket2 == bucket3

    def test_get_bucket_distribution(self) -> None:
        """测试分桶的分布：大量 session_id 应大致符合配置比例"""
        router = ABTestRouter(bucket_a_ratio=0.7, enabled=True)

        bucket_a_count = 0
        bucket_b_count = 0
        total_sessions = 1000

        for i in range(total_sessions):
            session_id = f"session-{i}"
            bucket = router.get_bucket(session_id)
            if bucket == ABBucket.BUCKET_A:
                bucket_a_count += 1
            else:
                bucket_b_count += 1

        # 验证分布大致符合比例（允许 5% 误差）
        actual_ratio = bucket_a_count / total_sessions
        assert 0.65 <= actual_ratio <= 0.75, f"实际比例 {actual_ratio} 偏差过大"

    def test_get_bucket_consistency_across_instances(self) -> None:
        """测试不同实例对同一 session_id 的分桶一致性"""
        router1 = ABTestRouter(bucket_a_ratio=0.5)
        router2 = ABTestRouter(bucket_a_ratio=0.5)

        session_id = "consistent-session-test"

        bucket1 = router1.get_bucket(session_id)
        bucket2 = router2.get_bucket(session_id)

        assert bucket1 == bucket2

    def test_get_strategy(self) -> None:
        """测试获取 Bucket 策略"""
        router = ABTestRouter(
            bucket_a_strategy=RAGStrategy.SELF_HOSTED_RAG,
            bucket_b_strategy=RAGStrategy.BLUEPRINT_RAG,
        )

        assert router.get_strategy(ABBucket.BUCKET_A) == RAGStrategy.SELF_HOSTED_RAG
        assert router.get_strategy(ABBucket.BUCKET_B) == RAGStrategy.BLUEPRINT_RAG

    def test_record_metrics(self) -> None:
        """测试指标记录"""
        router = ABTestRouter()

        # 记录 Bucket A 指标
        router.record_metrics(
            bucket=ABBucket.BUCKET_A,
            latency_ms=100,
            tokens=50,
            feedback="positive",
        )

        # 记录 Bucket B 指标
        router.record_metrics(
            bucket=ABBucket.BUCKET_B,
            latency_ms=200,
            tokens=100,
            feedback="negative",
        )

        metrics = router.get_metrics()

        assert metrics.bucket_a_total_requests == 1
        assert metrics.bucket_b_total_requests == 1
        assert metrics.bucket_a_avg_latency_ms == 100.0
        assert metrics.bucket_b_avg_latency_ms == 200.0

    def test_record_metrics_multiple(self) -> None:
        """测试多次指标记录"""
        router = ABTestRouter()

        # 记录多次 Bucket A 指标
        for i in range(5):
            router.record_metrics(
                bucket=ABBucket.BUCKET_A,
                latency_ms=100 + i * 10,
                tokens=50,
            )

        metrics = router.get_metrics()

        assert metrics.bucket_a_total_requests == 5
        # 平均延迟 = (100 + 110 + 120 + 130 + 140) / 5 = 120
        assert metrics.bucket_a_avg_latency_ms == 120.0

    def test_update_config_ratio(self) -> None:
        """测试更新流量比例"""
        router = ABTestRouter(bucket_a_ratio=0.5)

        config = ABConfigRequest(bucket_a_ratio=0.8)
        router.update_config(config)

        updated_config = router.get_config()
        assert updated_config.bucket_a_ratio == 0.8

    def test_update_config_strategy(self) -> None:
        """测试更新策略"""
        router = ABTestRouter()

        config = ABConfigRequest(
            bucket_a_strategy="blueprint_rag",
            bucket_b_strategy="self_hosted_rag",
        )
        router.update_config(config)

        updated_config = router.get_config()
        assert updated_config.bucket_a_strategy == RAGStrategy.BLUEPRINT_RAG
        assert updated_config.bucket_b_strategy == RAGStrategy.SELF_HOSTED_RAG

    def test_update_config_enabled(self) -> None:
        """测试更新启用状态"""
        router = ABTestRouter(enabled=False)

        config = ABConfigRequest(enabled=True)
        router.update_config(config)

        assert router.is_enabled() is True

    def test_update_config_partial(self) -> None:
        """测试部分更新配置"""
        router = ABTestRouter(
            bucket_a_ratio=0.5,
            bucket_a_strategy=RAGStrategy.SELF_HOSTED_RAG,
        )

        # 仅更新 ratio，strategy 保持不变
        config = ABConfigRequest(bucket_a_ratio=0.7)
        router.update_config(config)

        updated_config = router.get_config()
        assert updated_config.bucket_a_ratio == 0.7
        assert updated_config.bucket_a_strategy == RAGStrategy.SELF_HOSTED_RAG

    def test_update_config_invalid_ratio(self) -> None:
        """测试非法流量比例（Pydantic 在模型层面验证）"""
        router = ABTestRouter()

        # 比例 > 1.0 - Pydantic 验证会先触发
        with pytest.raises(ValueError):
            router.update_config(ABConfigRequest(bucket_a_ratio=1.5))

        # 比例 < 0.0 - Pydantic 验证会先触发
        with pytest.raises(ValueError):
            router.update_config(ABConfigRequest(bucket_a_ratio=-0.1))

    def test_update_config_invalid_strategy(self) -> None:
        """测试非法策略"""
        router = ABTestRouter()

        with pytest.raises(ValueError, match="非法的 RAG 策略"):
            router.update_config(ABConfigRequest(bucket_a_strategy="invalid_strategy"))

    def test_metrics_persistence(self) -> None:
        """测试指标持久化到文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_file = Path(tmpdir) / "ab_metrics.json"
            router = ABTestRouter(metrics_file=metrics_file)

            # 记录指标
            router.record_metrics(
                bucket=ABBucket.BUCKET_A,
                latency_ms=100,
                tokens=50,
                feedback="positive",
            )
            router.record_metrics(
                bucket=ABBucket.BUCKET_B,
                latency_ms=200,
                tokens=100,
                feedback="negative",
            )

            # 创建新实例加载持久化指标
            router2 = ABTestRouter(metrics_file=metrics_file)
            metrics = router2.get_metrics()

            assert metrics.bucket_a_total_requests == 1
            assert metrics.bucket_b_total_requests == 1

    def test_reset_metrics(self) -> None:
        """测试重置指标"""
        router = ABTestRouter()

        # 记录指标
        router.record_metrics(
            bucket=ABBucket.BUCKET_A,
            latency_ms=100,
            tokens=50,
        )

        # 重置
        router.reset_metrics()

        metrics = router.get_metrics()
        assert metrics.bucket_a_total_requests == 0
        assert metrics.bucket_b_total_requests == 0

    def test_get_bucket_metrics_detail(self) -> None:
        """测试获取详细指标"""
        router = ABTestRouter()

        router.record_metrics(
            bucket=ABBucket.BUCKET_A,
            latency_ms=100,
            tokens=50,
            feedback="positive",
        )
        router.record_metrics(
            bucket=ABBucket.BUCKET_A,
            latency_ms=200,
            tokens=100,
            feedback="negative",
        )

        detail = router.get_bucket_metrics_detail(ABBucket.BUCKET_A)

        assert detail.total_requests == 2
        assert detail.total_latency_ms == 300.0
        assert detail.total_tokens == 150
        assert detail.feedback_positive == 1
        assert detail.feedback_negative == 1

    def test_bucket_ratio_edge_cases(self) -> None:
        """测试边界流量比例"""
        # ratio = 0.0，所有流量到 Bucket B
        router_all_b = ABTestRouter(bucket_a_ratio=0.0)
        for i in range(100):
            bucket = router_all_b.get_bucket(f"session-{i}")
            assert bucket == ABBucket.BUCKET_B

        # ratio = 1.0，所有流量到 Bucket A
        router_all_a = ABTestRouter(bucket_a_ratio=1.0)
        for i in range(100):
            bucket = router_all_a.get_bucket(f"session-{i}")
            assert bucket == ABBucket.BUCKET_A

    def test_empty_session_id(self) -> None:
        """测试空 session_id 的分桶"""
        router = ABTestRouter(bucket_a_ratio=0.5)

        # 空 session_id 也应返回确定的 Bucket
        bucket = router.get_bucket("")
        assert bucket in [ABBucket.BUCKET_A, ABBucket.BUCKET_B]

        # 多次调用应返回相同结果
        assert router.get_bucket("") == bucket

    def test_unicode_session_id(self) -> None:
        """测试 Unicode session_id 的分桶"""
        router = ABTestRouter(bucket_a_ratio=0.5)

        # Unicode session_id 也应正常工作
        session_id = "会话-测试-🚀"
        bucket1 = router.get_bucket(session_id)
        bucket2 = router.get_bucket(session_id)

        assert bucket1 == bucket2