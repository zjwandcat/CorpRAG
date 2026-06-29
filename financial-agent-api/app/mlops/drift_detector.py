"""查询漂移检测器模块

检测用户查询分布相对于参考数据集的偏移，支持 K-S 检验和 MMD 两种算法。

核心设计原则：
- 使用云端 Embedding API 提取 query 向量（禁止本地模型）
- 支持 K-S 检验和 MMD 两种检测算法
- 检测超过阈值时触发 Prometheus 告警
- 初始化失败时标记为不可用，不阻塞对话链路

性能约束：
- 单条 query 检测在 2 秒内完成
- 异常时优雅降级，记录 warning 日志
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings
from app.core.enums import DriftDetectionMethod, DriftStatus

logger = logging.getLogger(__name__)

# ============================================================================
# Prometheus 指标定义
# ============================================================================

_DRIFT_DETECTED_TOTAL: Any
_DRIFT_LAST_STATISTIC: Any
_DRIFT_STATUS_GAUGE: Any

try:
    from prometheus_client import Counter, Gauge

    _PROMETHEUS_AVAILABLE = True

    DRIFT_DETECTED_TOTAL = Counter(
        "agent_drift_detected_total",
        "Total count of drift detections by method and status",
        labelnames=["method", "status"],
    )

    DRIFT_LAST_STATISTIC = Gauge(
        "agent_drift_last_statistic",
        "Last drift statistic value by method",
        labelnames=["method"],
    )

    DRIFT_STATUS_GAUGE = Gauge(
        "agent_drift_status",
        "Drift detector status (1=healthy, 0.5=degraded, 0=unavailable)",
    )

    _DRIFT_DETECTED_TOTAL = DRIFT_DETECTED_TOTAL
    _DRIFT_LAST_STATISTIC = DRIFT_LAST_STATISTIC
    _DRIFT_STATUS_GAUGE = DRIFT_STATUS_GAUGE
except ImportError:
    _PROMETHEUS_AVAILABLE = False

    class _NoopMetric:
        """空操作指标桩类"""

        def labels(self, *args: Any, **kwargs: Any) -> "_NoopMetric":
            return self

        def inc(self, amount: float = 1.0) -> None:
            pass

        def set(self, value: float) -> None:
            pass

    _DRIFT_DETECTED_TOTAL = _NoopMetric()
    _DRIFT_LAST_STATISTIC = _NoopMetric()
    _DRIFT_STATUS_GAUGE = _NoopMetric()


# ============================================================================
# 数据结构定义
# ============================================================================


@dataclass(slots=True)
class DriftResult:
    """漂移检测结果

    Attributes:
        is_drifted: 是否检测到漂移
        statistic: 检测统计量
        p_value: p 值（K-S 检验有效，MMD 为 None）
        threshold: 漂移阈值
        method: 检测方法
        timestamp: 检测时间戳
    """

    is_drifted: bool
    statistic: float
    p_value: float | None
    threshold: float
    method: DriftDetectionMethod
    timestamp: str


@dataclass(slots=True)
class DriftDetectorStatus:
    """漂移检测器状态

    Attributes:
        status: 状态枚举值
        reference_loaded: 参考数据集是否已加载
        window_size: 当前窗口大小
        total_detections: 累计检测次数
        drift_count: 累计漂移次数
    """

    status: DriftStatus
    reference_loaded: bool
    window_size: int
    total_detections: int
    drift_count: int


# ============================================================================
# 核心检测器类
# ============================================================================


class QueryDriftDetector:
    """查询漂移检测器，检测用户查询分布相对于参考数据集的偏移。

    核心设计原则：
    - 使用云端 Embedding API 提取 query 向量（禁止本地模型）
    - 支持 K-S 检验和 MMD 两种检测算法
    - 检测超过阈值时触发 Prometheus 告警
    - 初始化失败时标记为不可用，不阻塞对话链路

    Example:
        >>> detector = QueryDriftDetector(
        ...     detection_method=DriftDetectionMethod.KS_TEST,
        ...     drift_threshold=0.05,
        ...     enabled=True,
        ... )
        >>> detector.load_reference_dataset("./reference_embeddings.npy")
        >>> result = detector.detect([0.1, 0.2, ...])
        >>> if result and result.is_drifted:
        ...     print("检测到查询漂移！")
    """

    def __init__(
        self,
        detection_method: DriftDetectionMethod = DriftDetectionMethod.KS_TEST,
        drift_threshold: float = 0.05,
        enabled: bool = True,
        reference_dataset_size: int = 100,
    ) -> None:
        """初始化漂移检测器。

        Args:
            detection_method: 检测方法（KS_TEST 或 MMD）
            drift_threshold: 漂移阈值，超过此值触发告警
            enabled: 是否启用漂移检测
            reference_dataset_size: 参考数据集大小（默认 100 条）
        """
        self._detection_method = detection_method
        self._drift_threshold = drift_threshold
        self._enabled = enabled
        self._reference_dataset_size = reference_dataset_size

        # 参考数据集 embedding 缓存
        self._reference_embeddings: np.ndarray | None = None

        # 滑动窗口缓冲区，用于积累当前样本
        self._window_buffer: list[list[float]] = []
        self._window_size = 30  # 窗口大小，积累 30 条 query 后执行检测

        # 可用性标志
        self._available = False

        # 统计计数器
        self._total_detections = 0
        self._drift_count = 0

        # 初始化 Prometheus 状态指标
        self._update_status_metric(DriftStatus.UNAVAILABLE)

        logger.info(
            "QueryDriftDetector 初始化完成：method=%s, threshold=%.4f, enabled=%s",
            detection_method,
            drift_threshold,
            enabled,
        )

    def load_reference_dataset(self, embeddings_path: str) -> None:
        """加载参考 embedding 缓存（100 条通用 query）。

        Args:
            embeddings_path: 缓存文件路径（.npy 格式）

        Raises:
            FileNotFoundError: 文件不存在时抛出
            ValueError: 文件格式错误时抛出
        """
        if not self._enabled:
            logger.info("漂移检测已禁用，跳过参考数据集加载")
            return

        path = Path(embeddings_path)

        try:
            if not path.exists():
                logger.warning(
                    "参考数据集文件不存在：%s，检测器标记为不可用",
                    embeddings_path,
                )
                self._available = False
                self._update_status_metric(DriftStatus.UNAVAILABLE)
                return

            # 加载 NumPy 数组
            loaded_array = np.load(str(path), allow_pickle=False)

            # 验证数组形状
            if loaded_array.ndim != 2:
                raise ValueError(
                    f"参考数据集维度错误，期望 2D 数组，实际为 {loaded_array.ndim}D"
                )

            if loaded_array.shape[0] < self._reference_dataset_size:
                logger.warning(
                    "参考数据集大小不足：期望 %d 条，实际 %d 条",
                    self._reference_dataset_size,
                    loaded_array.shape[0],
                )

            self._reference_embeddings = loaded_array.astype(np.float32)
            self._available = True
            self._update_status_metric(DriftStatus.NORMAL)

            logger.info(
                "参考数据集加载成功：shape=%s, dtype=%s",
                self._reference_embeddings.shape,
                self._reference_embeddings.dtype,
            )

        except Exception as exc:
            logger.warning(
                "参考数据集加载失败：%s，检测器标记为不可用",
                exc,
                exc_info=True,
            )
            self._available = False
            self._update_status_metric(DriftStatus.UNAVAILABLE)

    def detect(self, query_embedding: list[float]) -> DriftResult | None:
        """对单条 query 执行漂移检测。

        流程：
        1. 将 query 向量加入滑动窗口缓冲区
        2. 当窗口积累足够样本时，与 reference dataset 进行分布比较
        3. 超过阈值时触发 Prometheus 告警

        Args:
            query_embedding: query 向量（由云端 Embedding API 生成）

        Returns:
            DriftResult 检测结果，失败或窗口未满时返回 None
        """
        if not self._enabled or not self._available:
            return None

        if self._reference_embeddings is None:
            logger.warning("参考数据集未加载，跳过漂移检测")
            return None

        start_time = time.perf_counter()

        try:
            # 将 query 向量加入窗口缓冲区
            self._window_buffer.append(query_embedding)

            # 窗口未满，返回 None
            if len(self._window_buffer) < self._window_size:
                return DriftResult(
                    is_drifted=False,
                    statistic=0.0,
                    p_value=None,
                    threshold=self._drift_threshold,
                    method=self._detection_method,
                    timestamp=datetime.now().isoformat(),
                )

            # 窗口已满，执行分布比较
            current_embeddings = np.array(self._window_buffer, dtype=np.float32)

            # 根据检测方法选择算法
            if self._detection_method == DriftDetectionMethod.KS_TEST:
                statistic, p_value = self._ks_test(
                    current_embeddings, self._reference_embeddings
                )
            else:  # MMD
                statistic = self._mmd(current_embeddings, self._reference_embeddings)
                p_value = None

            # 判断是否漂移
            is_drifted = statistic > self._drift_threshold

            # 更新统计计数器
            self._total_detections += 1
            if is_drifted:
                self._drift_count += 1

            # 触发 Prometheus 告警
            if is_drifted:
                self._check_and_alert(statistic, p_value)

            # 清空窗口缓冲区（滑动窗口重置）
            self._window_buffer.clear()

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(
                "漂移检测完成：method=%s, statistic=%.4f, threshold=%.4f, is_drifted=%s, elapsed=%.1fms",
                self._detection_method,
                statistic,
                self._drift_threshold,
                is_drifted,
                elapsed_ms,
            )

            return DriftResult(
                is_drifted=is_drifted,
                statistic=statistic,
                p_value=p_value,
                threshold=self._drift_threshold,
                method=self._detection_method,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as exc:
            logger.warning(
                "漂移检测执行失败：%s",
                exc,
                exc_info=True,
            )
            self._window_buffer.clear()
            return None

    def _ks_test(
        self,
        current_embeddings: np.ndarray,
        reference_embeddings: np.ndarray,
    ) -> tuple[float, float]:
        """K-S 检验实现。

        使用 scipy.stats.ks_2samp 计算两个样本分布的 K-S 统计量。

        Args:
            current_embeddings: 当前样本 embedding（shape: [n, d]）
            reference_embeddings: 参考样本 embedding（shape: [m, d]）

        Returns:
            (statistic, p_value) 元组
        """
        try:
            from scipy.stats import ks_2samp
        except ImportError:
            logger.warning("scipy 未安装，K-S 检验不可用，返回默认值")
            return (0.0, 1.0)

        # 将多维 embedding 展平为 1D，计算全局分布
        current_flat = current_embeddings.flatten()
        reference_flat = reference_embeddings.flatten()

        # 执行 K-S 检验
        statistic, p_value = ks_2samp(current_flat, reference_flat)

        return (float(statistic), float(p_value))

    def _mmd(
        self,
        current_embeddings: np.ndarray,
        reference_embeddings: np.ndarray,
    ) -> float:
        """MMD（Maximum Mean Discrepancy）计算实现。

        使用 RBF 核计算两个样本分布的最大均值差异。

        Args:
            current_embeddings: 当前样本 embedding（shape: [n, d]）
            reference_embeddings: 参考样本 embedding（shape: [m, d]）

        Returns:
            MMD 统计量
        """
        # RBF 核带宽参数（使用中值启发式）
        sigma = 1.0

        def rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
            """RBF 核函数：k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))"""
            dist_sq = np.sum((x - y) ** 2)
            return float(np.exp(-dist_sq / (2 * sigma**2)))

        n = len(current_embeddings)
        m = len(reference_embeddings)

        # 计算 k(X, X) 均值
        k_xx_sum = 0.0
        for i in range(n):
            for j in range(n):
                k_xx_sum += rbf_kernel(current_embeddings[i], current_embeddings[j], sigma)
        k_xx_mean = k_xx_sum / (n * n)

        # 计算 k(Y, Y) 均值
        k_yy_sum = 0.0
        for i in range(m):
            for j in range(m):
                k_yy_sum += rbf_kernel(reference_embeddings[i], reference_embeddings[j], sigma)
        k_yy_mean = k_yy_sum / (m * m)

        # 计算 k(X, Y) 均值
        k_xy_sum = 0.0
        for i in range(n):
            for j in range(m):
                k_xy_sum += rbf_kernel(current_embeddings[i], reference_embeddings[j], sigma)
        k_xy_mean = k_xy_sum / (n * m)

        # MMD^2 = k(X,X) + k(Y,Y) - 2*k(X,Y)
        mmd_squared = k_xx_mean + k_yy_mean - 2 * k_xy_mean

        # 确保 MMD 为非负数
        mmd = float(np.sqrt(max(0.0, mmd_squared)))

        return mmd

    def check_and_alert(self, query: str) -> DriftResult | None:
        """完整流程：获取 embedding → 检测 → 触发 Prometheus 告警。

        此方法封装了完整的漂移检测流程，包括：
        1. 调用云端 Embedding API 获取 query 向量
        2. 执行漂移检测
        3. 检测到漂移时触发 Prometheus 告警

        Args:
            query: 用户查询文本

        Returns:
            DriftResult 检测结果，失败时返回 None
        """
        if not self._enabled or not self._available:
            return None

        start_time = time.perf_counter()

        try:
            # 调用云端 Embedding API（禁止本地模型）
            from app.core.dependencies import get_embeddings

            embeddings = get_embeddings()
            query_embedding = embeddings.embed_query(query)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # 性能约束检查：单条 query 检测应在 2 秒内完成
            if elapsed_ms > 2000:
                logger.warning(
                    "Embedding 获取耗时过长：%.1fms > 2000ms",
                    elapsed_ms,
                )

            # 执行漂移检测
            result = self.detect(query_embedding)

            return result

        except Exception as exc:
            # 异常降级：Embedding API 不可用时跳过检测，记录 warning 日志
            logger.warning(
                "Embedding API 不可用，跳过漂移检测：%s",
                exc,
                exc_info=True,
            )
            self._update_status_metric(DriftStatus.DEGRADED)
            return None

    def _check_and_alert(self, statistic: float, p_value: float | None) -> None:
        """检查漂移统计量是否超过阈值，超过则触发 Prometheus 告警。

        Args:
            statistic: 检测统计量
            p_value: p 值（K-S 检验有效，MMD 为 None）
        """
        method_label = self._detection_method.value

        # 递增漂移检测计数
        _DRIFT_DETECTED_TOTAL.labels(method=method_label, status="drifted").inc()

        # 设置最近一次漂移统计量
        _DRIFT_LAST_STATISTIC.labels(method=method_label).set(statistic)

        logger.warning(
            "检测到查询漂移：method=%s, statistic=%.4f, threshold=%.4f, p_value=%s",
            self._detection_method,
            statistic,
            self._drift_threshold,
            p_value,
        )

    def is_available(self) -> bool:
        """检查漂移检测器是否可用。

        Returns:
            True 表示可用，False 表示不可用
        """
        return self._available and self._enabled

    def get_status(self) -> DriftDetectorStatus:
        """获取漂移检测器状态。

        Returns:
            DriftDetectorStatus 状态对象
        """
        if not self._enabled:
            status = DriftStatus.UNAVAILABLE
        elif not self._available:
            status = DriftStatus.UNAVAILABLE
        else:
            status = DriftStatus.NORMAL

        return DriftDetectorStatus(
            status=status,
            reference_loaded=self._reference_embeddings is not None,
            window_size=len(self._window_buffer),
            total_detections=self._total_detections,
            drift_count=self._drift_count,
        )

    def _update_status_metric(self, status: DriftStatus) -> None:
        """更新 Prometheus 状态指标。

        Args:
            status: 状态枚举值
        """
        status_value = {
            DriftStatus.NORMAL: 1.0,
            DriftStatus.DEGRADED: 0.5,
            DriftStatus.UNAVAILABLE: 0.0,
        }.get(status, 0.0)

        _DRIFT_STATUS_GAUGE.set(status_value)


# ============================================================================
# 工厂函数
# ============================================================================


def create_drift_detector() -> QueryDriftDetector:
    """创建漂移检测器实例（从配置读取参数）。

    Returns:
        QueryDriftDetector 实例
    """
    # 解析检测方法
    method_str = settings.DRIFT_DETECTION_METHOD.lower()
    detection_method = (
        DriftDetectionMethod.KS_TEST
        if method_str == "ks_test"
        else DriftDetectionMethod.MMD
    )

    detector = QueryDriftDetector(
        detection_method=detection_method,
        drift_threshold=settings.DRIFT_THRESHOLD,
        enabled=settings.DRIFT_ENABLED,
        reference_dataset_size=settings.DRIFT_REFERENCE_DATASET_SIZE,
    )

    # 加载参考数据集
    if settings.DRIFT_ENABLED:
        detector.load_reference_dataset(settings.DRIFT_REFERENCE_EMBEDDINGS_PATH)

    return detector


__all__ = [
    "DriftDetectorStatus",
    "DriftResult",
    "QueryDriftDetector",
    "create_drift_detector",
]