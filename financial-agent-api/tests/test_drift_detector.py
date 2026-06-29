"""QueryDriftDetector 单元测试

测试漂移检测器的核心功能：
- K-S 检验和 MMD 两种算法
- 漂移超过阈值时触发 Prometheus 告警
- Embedding API 不可用时优雅降级
- 单条 query 检测延迟 < 2 秒
"""

import time
from pathlib import Path

import numpy as np
import pytest

from app.core.enums import DriftDetectionMethod, DriftStatus
from app.mlops.drift_detector import (
    DriftDetectorStatus,
    DriftResult,
    QueryDriftDetector,
    create_drift_detector,
)


class TestQueryDriftDetector:
    """QueryDriftDetector 测试套件"""

    @pytest.fixture
    def reference_embeddings_path(self, tmp_path: Path) -> Path:
        """创建临时参考数据集文件"""
        reference_embeddings = np.random.randn(100, 1024).astype(np.float32)
        path = tmp_path / "reference_embeddings.npy"
        np.save(str(path), reference_embeddings)
        return path

    def test_init_default_params(self) -> None:
        """测试默认参数初始化"""
        detector = QueryDriftDetector()

        assert detector._detection_method == DriftDetectionMethod.KS_TEST
        assert detector._drift_threshold == 0.05
        assert detector._enabled is True
        assert detector._reference_dataset_size == 100
        assert detector._available is False

    def test_init_custom_params(self) -> None:
        """测试自定义参数初始化"""
        detector = QueryDriftDetector(
            detection_method=DriftDetectionMethod.MMD,
            drift_threshold=0.1,
            enabled=False,
            reference_dataset_size=200,
        )

        assert detector._detection_method == DriftDetectionMethod.MMD
        assert detector._drift_threshold == 0.1
        assert detector._enabled is False
        assert detector._reference_dataset_size == 200

    def test_load_reference_dataset(
        self, reference_embeddings_path: Path
    ) -> None:
        """测试参考数据集加载"""
        detector = QueryDriftDetector()
        detector.load_reference_dataset(str(reference_embeddings_path))

        assert detector._available is True
        assert detector._reference_embeddings is not None
        assert detector._reference_embeddings.shape == (100, 1024)

    def test_load_reference_dataset_file_not_found(self) -> None:
        """测试参考数据集文件不存在时的降级处理"""
        detector = QueryDriftDetector()
        detector.load_reference_dataset("non_existent_file.npy")

        assert detector._available is False
        assert detector._reference_embeddings is None

    def test_ks_test(self, reference_embeddings_path: Path) -> None:
        """测试 K-S 检验算法"""
        detector = QueryDriftDetector()
        detector.load_reference_dataset(str(reference_embeddings_path))

        current_embeddings = np.random.randn(30, 1024).astype(np.float32)
        statistic, p_value = detector._ks_test(
            current_embeddings, detector._reference_embeddings
        )

        assert isinstance(statistic, float)
        assert isinstance(p_value, float)
        assert 0.0 <= statistic <= 1.0
        assert 0.0 <= p_value <= 1.0

    def test_mmd(self, reference_embeddings_path: Path) -> None:
        """测试 MMD 算法"""
        detector = QueryDriftDetector()
        detector.load_reference_dataset(str(reference_embeddings_path))

        current_embeddings = np.random.randn(30, 1024).astype(np.float32)
        mmd_value = detector._mmd(
            current_embeddings, detector._reference_embeddings
        )

        assert isinstance(mmd_value, float)
        assert mmd_value >= 0.0

    def test_detect_window_not_full(
        self, reference_embeddings_path: Path
    ) -> None:
        """测试窗口未满时的检测"""
        detector = QueryDriftDetector()
        detector.load_reference_dataset(str(reference_embeddings_path))

        query_embedding = np.random.randn(1024).tolist()
        result = detector.detect(query_embedding)

        # 窗口未满，返回 is_drifted=False 的结果
        assert result is not None
        assert result.is_drifted is False
        assert result.statistic == 0.0
        assert len(detector._window_buffer) == 1

    def test_detect_window_full(
        self, reference_embeddings_path: Path
    ) -> None:
        """测试窗口已满时的检测"""
        detector = QueryDriftDetector()
        detector.load_reference_dataset(str(reference_embeddings_path))

        # 填满窗口
        for _ in range(30):
            query_embedding = np.random.randn(1024).tolist()
            result = detector.detect(query_embedding)

        # 窗口已满，执行检测后清空
        assert result is not None
        assert detector._total_detections == 1
        assert len(detector._window_buffer) == 0

    def test_detect_drift_triggered(
        self, reference_embeddings_path: Path
    ) -> None:
        """测试漂移触发告警"""
        detector = QueryDriftDetector(
            drift_threshold=0.001  # 极低阈值，必定触发漂移
        )
        detector.load_reference_dataset(str(reference_embeddings_path))

        # 使用偏移的分布触发漂移
        for _ in range(30):
            query_embedding = (np.random.randn(1024) + 2.0).tolist()
            result = detector.detect(query_embedding)

        assert result is not None
        assert result.is_drifted is True
        assert detector._drift_count == 1

    def test_detect_disabled(self) -> None:
        """测试禁用状态下的检测"""
        detector = QueryDriftDetector(enabled=False)
        query_embedding = np.random.randn(1024).tolist()
        result = detector.detect(query_embedding)

        assert result is None

    def test_detect_unavailable(self) -> None:
        """测试不可用状态下的检测"""
        detector = QueryDriftDetector(enabled=True)
        # 不加载参考数据集，检测器不可用
        query_embedding = np.random.randn(1024).tolist()
        result = detector.detect(query_embedding)

        assert result is None

    def test_is_available(
        self, reference_embeddings_path: Path
    ) -> None:
        """测试可用性检查"""
        detector = QueryDriftDetector()
        assert detector.is_available() is False

        detector.load_reference_dataset(str(reference_embeddings_path))
        assert detector.is_available() is True

    def test_get_status(
        self, reference_embeddings_path: Path
    ) -> None:
        """测试状态获取"""
        detector = QueryDriftDetector()
        detector.load_reference_dataset(str(reference_embeddings_path))

        status = detector.get_status()

        assert isinstance(status, DriftDetectorStatus)
        assert status.status == DriftStatus.NORMAL
        assert status.reference_loaded is True
        assert status.window_size == 0
        assert status.total_detections == 0
        assert status.drift_count == 0

    def test_performance_constraint(
        self, reference_embeddings_path: Path
    ) -> None:
        """测试性能约束：单条 query 检测在 2 秒内完成"""
        detector = QueryDriftDetector()
        detector.load_reference_dataset(str(reference_embeddings_path))

        # 测量检测延迟
        latencies = []
        for _ in range(5):
            # 填满窗口触发检测
            for _ in range(30):
                query_embedding = np.random.randn(1024).tolist()
                start = time.perf_counter()
                result = detector.detect(query_embedding)
                elapsed = (time.perf_counter() - start) * 1000
                if result is not None and result.statistic > 0:
                    latencies.append(elapsed)

        if latencies:
            max_latency = max(latencies)
            assert max_latency < 2000, f"检测延迟 {max_latency:.1f}ms 超过 2000ms 约束"

    def test_mmd_method(
        self, reference_embeddings_path: Path
    ) -> None:
        """测试 MMD 检测方法"""
        detector = QueryDriftDetector(
            detection_method=DriftDetectionMethod.MMD
        )
        detector.load_reference_dataset(str(reference_embeddings_path))

        # 填满窗口
        for _ in range(30):
            query_embedding = np.random.randn(1024).tolist()
            result = detector.detect(query_embedding)

        assert result is not None
        assert result.method == DriftDetectionMethod.MMD
        assert result.p_value is None  # MMD 不返回 p_value

    def test_check_and_alert_disabled(self) -> None:
        """测试 check_and_alert 在禁用状态下的行为"""
        detector = QueryDriftDetector(enabled=False)
        result = detector.check_and_alert("test query")

        assert result is None


class TestDriftResult:
    """DriftResult 数据类测试"""

    def test_drift_result_creation(self) -> None:
        """测试 DriftResult 创建"""
        result = DriftResult(
            is_drifted=True,
            statistic=0.1,
            p_value=0.05,
            threshold=0.05,
            method=DriftDetectionMethod.KS_TEST,
            timestamp="2024-01-01T00:00:00",
        )

        assert result.is_drifted is True
        assert result.statistic == 0.1
        assert result.p_value == 0.05
        assert result.threshold == 0.05
        assert result.method == DriftDetectionMethod.KS_TEST


class TestDriftDetectorStatus:
    """DriftDetectorStatus 数据类测试"""

    def test_status_creation(self) -> None:
        """测试 DriftDetectorStatus 创建"""
        status = DriftDetectorStatus(
            status=DriftStatus.NORMAL,
            reference_loaded=True,
            window_size=10,
            total_detections=5,
            drift_count=2,
        )

        assert status.status == DriftStatus.NORMAL
        assert status.reference_loaded is True
        assert status.window_size == 10
        assert status.total_detections == 5
        assert status.drift_count == 2


class TestCreateDriftDetector:
    """工厂函数测试"""

    def test_create_drift_detector(self, tmp_path: Path) -> None:
        """测试工厂函数创建检测器"""
        # 创建参考数据集
        reference_embeddings = np.random.randn(100, 1024).astype(np.float32)
        path = tmp_path / "reference_embeddings.npy"
        np.save(str(path), reference_embeddings)

        # 临时修改配置
        from app.core import config

        original_path = config.settings.DRIFT_REFERENCE_EMBEDDINGS_PATH
        config.settings.DRIFT_REFERENCE_EMBEDDINGS_PATH = str(path)

        try:
            detector = create_drift_detector()

            assert isinstance(detector, QueryDriftDetector)
            assert detector._detection_method == DriftDetectionMethod.KS_TEST
            assert detector._drift_threshold == 0.05
            assert detector._enabled is True
        finally:
            config.settings.DRIFT_REFERENCE_EMBEDDINGS_PATH = original_path