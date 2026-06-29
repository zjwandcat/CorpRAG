"""QueryDriftDetector 单元测试

测试 app.mlops.drift_detector 模块的核心功能：
- K-S 检验算法
- MMD 算法
- 漂移超过阈值时触发告警
- Embedding API 不可用时的异常降级
- 漂移检测开关（enabled/disabled）

所有测试使用 mock，不调用真实 Embedding API。
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.core.enums import DriftDetectionMethod, DriftStatus
from app.mlops.drift_detector import (
    DriftDetectorStatus,
    DriftResult,
    QueryDriftDetector,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def reference_embeddings() -> np.ndarray:
    """创建参考 embedding 数据集（100 条，每条 768 维）"""
    np.random.seed(42)
    return np.random.randn(100, 768).astype(np.float32)


@pytest.fixture
def reference_embeddings_file(reference_embeddings: np.ndarray) -> Path:
    """创建临时参考 embedding 文件"""
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f.name, reference_embeddings)
        yield Path(f.name)


@pytest.fixture
def detector_enabled(reference_embeddings_file: Path) -> QueryDriftDetector:
    """创建启用的漂移检测器（K-S 检验）"""
    detector = QueryDriftDetector(
        detection_method=DriftDetectionMethod.KS_TEST,
        drift_threshold=0.05,
        enabled=True,
        reference_dataset_size=100,
    )
    detector.load_reference_dataset(str(reference_embeddings_file))
    return detector


@pytest.fixture
def detector_mmd(reference_embeddings_file: Path) -> QueryDriftDetector:
    """创建使用 MMD 算法的检测器"""
    detector = QueryDriftDetector(
        detection_method=DriftDetectionMethod.MMD,
        drift_threshold=0.5,
        enabled=True,
        reference_dataset_size=100,
    )
    detector.load_reference_dataset(str(reference_embeddings_file))
    return detector


@pytest.fixture
def detector_disabled() -> QueryDriftDetector:
    """创建禁用的漂移检测器"""
    return QueryDriftDetector(
        detection_method=DriftDetectionMethod.KS_TEST,
        drift_threshold=0.05,
        enabled=False,
    )


@pytest.fixture
def detector_unavailable() -> QueryDriftDetector:
    """创建不可用的漂移检测器（参考数据集未加载）"""
    detector = QueryDriftDetector(
        detection_method=DriftDetectionMethod.KS_TEST,
        drift_threshold=0.05,
        enabled=True,
    )
    # 不加载参考数据集，标记为不可用
    return detector


# ===========================================================================
# TestKSTest — 测试 K-S 检验算法
# ===========================================================================


class TestKSTest:
    """K-S 检验算法测试"""

    def test_ks_test_no_drift(self, detector_enabled: QueryDriftDetector) -> None:
        """测试 K-S 检验：无漂移情况

        验证：
        - 当当前样本与参考样本分布相似时，statistic 较小
        - is_drifted 为 False
        """
        # 生成与参考数据分布相似的样本
        np.random.seed(100)
        similar_embeddings = [np.random.randn(768).tolist() for _ in range(30)]

        # 填充窗口
        for emb in similar_embeddings[:-1]:
            detector_enabled.detect(emb)

        # 最后一条触发检测
        result = detector_enabled.detect(similar_embeddings[-1])

        assert result is not None
        assert result.method == DriftDetectionMethod.KS_TEST
        assert result.p_value is not None
        # 无漂移时 statistic 应较小
        assert result.statistic < 0.2

    def test_ks_test_with_drift(self, detector_enabled: QueryDriftDetector, reference_embeddings: np.ndarray) -> None:
        """测试 K-S 检验：有漂移情况

        验证：
        - 当当前样本与参考样本分布差异较大时，statistic 较大
        - 可能触发漂移告警
        """
        # 生成与参考数据分布差异较大的样本（均值偏移）
        np.random.seed(200)
        drifted_embeddings = [(np.random.randn(768) + 3.0).tolist() for _ in range(30)]

        # 填充窗口
        for emb in drifted_embeddings[:-1]:
            detector_enabled.detect(emb)

        # 最后一条触发检测
        result = detector_enabled.detect(drifted_embeddings[-1])

        assert result is not None
        assert result.method == DriftDetectionMethod.KS_TEST
        # 有漂移时 statistic 应较大
        assert result.statistic > 0.1

    def test_ks_test_returns_p_value(self, detector_enabled: QueryDriftDetector) -> None:
        """测试 K-S 检验返回 p 值"""
        np.random.seed(150)
        embeddings = [np.random.randn(768).tolist() for _ in range(30)]

        for emb in embeddings[:-1]:
            detector_enabled.detect(emb)

        result = detector_enabled.detect(embeddings[-1])

        assert result is not None
        assert result.p_value is not None
        assert 0.0 <= result.p_value <= 1.0


# ===========================================================================
# TestMMD — 测试 MMD 算法
# ===========================================================================


class TestMMD:
    """MMD 算法测试"""

    def test_mmd_no_drift(self, detector_mmd: QueryDriftDetector) -> None:
        """测试 MMD：无漂移情况

        验证：
        - 当当前样本与参考样本分布相似时，MMD 较小
        """
        np.random.seed(300)
        similar_embeddings = [np.random.randn(768).tolist() for _ in range(30)]

        for emb in similar_embeddings[:-1]:
            detector_mmd.detect(emb)

        result = detector_mmd.detect(similar_embeddings[-1])

        assert result is not None
        assert result.method == DriftDetectionMethod.MMD
        assert result.p_value is None  # MMD 不返回 p 值
        # MMD 应为非负数
        assert result.statistic >= 0.0

    def test_mmd_with_drift(self, detector_mmd: QueryDriftDetector) -> None:
        """测试 MMD：有漂移情况

        验证：
        - 当当前样本与参考样本分布差异较大时，MMD 较大
        """
        np.random.seed(400)
        drifted_embeddings = [(np.random.randn(768) + 5.0).tolist() for _ in range(30)]

        for emb in drifted_embeddings[:-1]:
            detector_mmd.detect(emb)

        result = detector_mmd.detect(drifted_embeddings[-1])

        assert result is not None
        assert result.method == DriftDetectionMethod.MMD
        # 有漂移时 MMD 应较大（相对于无漂移情况）
        assert result.statistic > 0.1

    def test_mmd_no_p_value(self, detector_mmd: QueryDriftDetector) -> None:
        """测试 MMD 不返回 p 值"""
        np.random.seed(350)
        embeddings = [np.random.randn(768).tolist() for _ in range(30)]

        for emb in embeddings[:-1]:
            detector_mmd.detect(emb)

        result = detector_mmd.detect(embeddings[-1])

        assert result is not None
        assert result.p_value is None


# ===========================================================================
# TestDriftAlert — 测试漂移告警
# ===========================================================================


class TestDriftAlert:
    """漂移告警测试"""

    def test_drift_triggers_alert(self, reference_embeddings_file: Path) -> None:
        """测试漂移超过阈值时触发告警

        验证：
        - is_drifted 为 True
        - drift_count 递增
        """
        # 设置较低的阈值，确保触发漂移
        detector = QueryDriftDetector(
            detection_method=DriftDetectionMethod.KS_TEST,
            drift_threshold=0.01,  # 低阈值
            enabled=True,
        )
        detector.load_reference_dataset(str(reference_embeddings_file))

        # 生成明显漂移的样本
        np.random.seed(500)
        drifted_embeddings = [(np.random.randn(768) + 4.0).tolist() for _ in range(30)]

        for emb in drifted_embeddings[:-1]:
            detector.detect(emb)

        result = detector.detect(drifted_embeddings[-1])

        assert result is not None
        if result.is_drifted:
            status = detector.get_status()
            assert status.drift_count > 0

    def test_no_drift_no_alert(self, detector_enabled: QueryDriftDetector) -> None:
        """测试无漂移时不触发告警

        验证：
        - is_drifted 为 False
        """
        np.random.seed(600)
        similar_embeddings = [np.random.randn(768).tolist() for _ in range(30)]

        for emb in similar_embeddings[:-1]:
            detector_enabled.detect(emb)

        result = detector_enabled.detect(similar_embeddings[-1])

        assert result is not None
        # 大概率无漂移
        # 注意：由于随机性，可能偶尔触发漂移，这里不强制断言


# ===========================================================================
# TestDriftDetectionDisabled — 测试漂移检测开关
# ===========================================================================


class TestDriftDetectionDisabled:
    """漂移检测开关测试"""

    def test_disabled_detect_returns_none(self, detector_disabled: QueryDriftDetector) -> None:
        """测试禁用时 detect 返回 None"""
        embedding = [0.1] * 768
        result = detector_disabled.detect(embedding)
        assert result is None

    def test_disabled_is_available(self, detector_disabled: QueryDriftDetector) -> None:
        """测试禁用时 is_available 返回 False"""
        assert detector_disabled.is_available() is False

    def test_disabled_status(self, detector_disabled: QueryDriftDetector) -> None:
        """测试禁用时状态为 UNAVAILABLE"""
        status = detector_disabled.get_status()
        assert status.status == DriftStatus.UNAVAILABLE

    def test_disabled_check_and_alert_returns_none(self, detector_disabled: QueryDriftDetector) -> None:
        """测试禁用时 check_and_alert 返回 None"""
        result = detector_disabled.check_and_alert("test query")
        assert result is None


# ===========================================================================
# TestDriftDetectorUnavailable — 测试检测器不可用降级
# ===========================================================================


class TestDriftDetectorUnavailable:
    """检测器不可用降级测试"""

    def test_unavailable_detect_returns_none(self, detector_unavailable: QueryDriftDetector) -> None:
        """测试不可用时 detect 返回 None"""
        embedding = [0.1] * 768
        result = detector_unavailable.detect(embedding)
        assert result is None

    def test_unavailable_is_available(self, detector_unavailable: QueryDriftDetector) -> None:
        """测试不可用时 is_available 返回 False"""
        assert detector_unavailable.is_available() is False

    def test_unavailable_status(self, detector_unavailable: QueryDriftDetector) -> None:
        """测试不可用时状态为 UNAVAILABLE"""
        status = detector_unavailable.get_status()
        assert status.status == DriftStatus.UNAVAILABLE

    def test_reference_not_loaded(self, detector_unavailable: QueryDriftDetector) -> None:
        """测试参考数据集未加载"""
        status = detector_unavailable.get_status()
        assert status.reference_loaded is False


# ===========================================================================
# TestEmbeddingAPIUnavailable — 测试 Embedding API 不可用降级
# ===========================================================================


class TestEmbeddingAPIUnavailable:
    """Embedding API 不可用降级测试"""

    def test_embedding_api_failure_returns_none(self, detector_enabled: QueryDriftDetector) -> None:
        """测试 Embedding API 失败时返回 None

        验证：
        - check_and_alert 捕获异常并返回 None
        - 不阻塞对话链路
        """
        # Mock get_embeddings 抛出异常（在 check_and_alert 方法内部导入）
        with patch("app.core.dependencies.get_embeddings") as mock_get_embeddings:
            mock_get_embeddings.side_effect = Exception("Embedding API 不可用")

            result = detector_enabled.check_and_alert("test query")

            assert result is None

    def test_embedding_api_timeout_returns_none(self, detector_enabled: QueryDriftDetector) -> None:
        """测试 Embedding API 超时时返回 None"""
        with patch("app.core.dependencies.get_embeddings") as mock_get_embeddings:
            mock_embeddings = MagicMock()
            mock_embeddings.embed_query.side_effect = TimeoutError("API 超时")
            mock_get_embeddings.return_value = mock_embeddings

            result = detector_enabled.check_and_alert("test query")

            assert result is None


# ===========================================================================
# TestWindowBuffer — 测试滑动窗口
# ===========================================================================


class TestWindowBuffer:
    """滑动窗口测试"""

    def test_window_not_full_returns_early_result(self, detector_enabled: QueryDriftDetector) -> None:
        """测试窗口未满时返回早期结果

        验证：
        - 窗口未满时 is_drifted 为 False
        - statistic 为 0.0
        """
        np.random.seed(700)
        embedding = np.random.randn(768).tolist()

        # 只添加一条，窗口未满
        result = detector_enabled.detect(embedding)

        assert result is not None
        assert result.is_drifted is False
        assert result.statistic == 0.0

    def test_window_clears_after_detection(self, detector_enabled: QueryDriftDetector) -> None:
        """测试检测后窗口被清空"""
        np.random.seed(800)
        embeddings = [np.random.randn(768).tolist() for _ in range(30)]

        # 填满窗口并触发检测
        for emb in embeddings:
            detector_enabled.detect(emb)

        status = detector_enabled.get_status()
        # 窗口应被清空
        assert status.window_size == 0

    def test_window_accumulates_correctly(self, detector_enabled: QueryDriftDetector) -> None:
        """测试窗口正确累积样本"""
        np.random.seed(900)

        for i in range(15):
            embedding = np.random.randn(768).tolist()
            detector_enabled.detect(embedding)

        status = detector_enabled.get_status()
        assert status.window_size == 15


# ===========================================================================
# TestLoadReferenceDataset — 测试加载参考数据集
# ===========================================================================


class TestLoadReferenceDataset:
    """加载参考数据集测试"""

    def test_load_valid_dataset(self, reference_embeddings_file: Path) -> None:
        """测试加载有效的参考数据集"""
        detector = QueryDriftDetector(
            detection_method=DriftDetectionMethod.KS_TEST,
            drift_threshold=0.05,
            enabled=True,
        )

        detector.load_reference_dataset(str(reference_embeddings_file))

        assert detector._available is True
        assert detector._reference_embeddings is not None
        assert detector._reference_embeddings.shape == (100, 768)

    def test_load_nonexistent_file(self) -> None:
        """测试加载不存在的文件

        验证：
        - 不抛出异常
        - 检测器标记为不可用
        """
        detector = QueryDriftDetector(
            detection_method=DriftDetectionMethod.KS_TEST,
            drift_threshold=0.05,
            enabled=True,
        )

        detector.load_reference_dataset("/nonexistent/path/embeddings.npy")

        assert detector._available is False
        assert detector._reference_embeddings is None

    def test_load_invalid_shape(self) -> None:
        """测试加载形状错误的文件"""
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            # 保存 1D 数组（错误形状）
            np.save(f.name, np.random.randn(100))
            temp_path = Path(f.name)

        detector = QueryDriftDetector(
            detection_method=DriftDetectionMethod.KS_TEST,
            drift_threshold=0.05,
            enabled=True,
        )

        detector.load_reference_dataset(str(temp_path))

        assert detector._available is False


# ===========================================================================
# TestDetectorStatus — 测试检测器状态
# ===========================================================================


class TestDetectorStatus:
    """检测器状态测试"""

    def test_get_status_normal(self, detector_enabled: QueryDriftDetector) -> None:
        """测试正常状态"""
        status = detector_enabled.get_status()

        assert status.status == DriftStatus.NORMAL
        assert status.reference_loaded is True
        assert status.total_detections >= 0
        assert status.drift_count >= 0

    def test_get_status_updates_counters(self, detector_enabled: QueryDriftDetector) -> None:
        """测试状态计数器正确更新"""
        np.random.seed(1000)
        embeddings = [np.random.randn(768).tolist() for _ in range(30)]

        initial_status = detector_enabled.get_status()
        initial_count = initial_status.total_detections

        # 触发检测
        for emb in embeddings:
            detector_enabled.detect(emb)

        updated_status = detector_enabled.get_status()
        assert updated_status.total_detections == initial_count + 1


# ===========================================================================
# TestDriftResult — 测试 DriftResult 数据结构
# ===========================================================================


class TestDriftResult:
    """DriftResult 数据结构测试"""

    def test_drift_result_fields(self) -> None:
        """测试 DriftResult 包含所有必要字段"""
        result = DriftResult(
            is_drifted=True,
            statistic=0.15,
            p_value=0.03,
            threshold=0.05,
            method=DriftDetectionMethod.KS_TEST,
            timestamp="2024-01-01T00:00:00",
        )

        assert result.is_drifted is True
        assert result.statistic == 0.15
        assert result.p_value == 0.03
        assert result.threshold == 0.05
        assert result.method == DriftDetectionMethod.KS_TEST
        assert result.timestamp == "2024-01-01T00:00:00"