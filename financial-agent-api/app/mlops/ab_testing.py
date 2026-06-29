"""A/B 测试路由器模块

基于 session_id 哈希分桶实现流量分发，支持动态策略更新和指标记录。

核心设计原则：
- 同一 session_id 始终分配到同一 Bucket（会话一致性）
- 策略变更不影响正在执行的对话请求
- 异常时降级为默认策略（Bucket A）
- 支持动态策略更新（无需重启服务）
"""

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.enums import ABBucket, RAGStrategy
from app.models.schemas import (
    ABConfigRequest,
    ABConfigResponse,
    ABMetricsResponse,
)

__all__ = ["ABTestRouter", "ABTestConfig", "ABTestMetric"]

logger = logging.getLogger(__name__)


@dataclass
class ABTestConfig:
    """A/B 测试配置"""

    bucket_a_ratio: float = 0.5
    bucket_a_strategy: RAGStrategy = RAGStrategy.SELF_HOSTED_RAG
    bucket_b_strategy: RAGStrategy = RAGStrategy.BLUEPRINT_RAG
    enabled: bool = True


@dataclass
class ABTestMetric:
    """A/B 测试指标"""

    bucket: ABBucket
    response_latency_ms: float
    total_tokens: int
    user_feedback: str | None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BucketMetrics:
    """单个 Bucket 的累计指标"""

    total_requests: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    feedback_positive: int = 0
    feedback_negative: int = 0


class ABTestRouter:
    """A/B 测试路由器，基于 session_id 哈希分桶实现流量分发。

    核心设计原则：
    - 同一 session_id 始终分配到同一 Bucket（会话一致性）
    - 策略变更不影响正在执行的对话请求
    - 异常时降级为默认策略（Bucket A）
    - 支持动态策略更新（无需重启服务）
    """

    def __init__(
        self,
        bucket_a_ratio: float = 0.5,
        bucket_a_strategy: RAGStrategy = RAGStrategy.SELF_HOSTED_RAG,
        bucket_b_strategy: RAGStrategy = RAGStrategy.BLUEPRINT_RAG,
        enabled: bool = False,
        metrics_file: Path | None = None,
    ) -> None:
        """初始化 A/B 测试路由器。

        Args:
            bucket_a_ratio: Bucket A 的流量比例（0.0-1.0）
            bucket_a_strategy: Bucket A 的 RAG 策略
            bucket_b_strategy: Bucket B 的 RAG 策略
            enabled: 是否启用 A/B 测试
            metrics_file: 指标持久化文件路径，为 None 时仅内存存储
        """
        self._config = ABTestConfig(
            bucket_a_ratio=bucket_a_ratio,
            bucket_a_strategy=bucket_a_strategy,
            bucket_b_strategy=bucket_b_strategy,
            enabled=enabled,
        )
        self._config_lock = threading.Lock()
        self._metrics_file = metrics_file

        # 初始化指标存储
        self._bucket_a_metrics = BucketMetrics()
        self._bucket_b_metrics = BucketMetrics()
        self._metrics_lock = threading.Lock()

        # 加载历史指标（如果存在）
        if metrics_file and metrics_file.exists():
            self._load_metrics_from_file()

        logger.info(
            "ABTestRouter 初始化完成：enabled=%s, bucket_a_ratio=%.2f, "
            "bucket_a_strategy=%s, bucket_b_strategy=%s",
            enabled,
            bucket_a_ratio,
            bucket_a_strategy,
            bucket_b_strategy,
        )

    def get_bucket(self, session_id: str) -> ABBucket:
        """基于 session_id 哈希确定性分桶。

        算法：hash(session_id) % 1000 / 1000.0 < bucket_a_ratio → A, 否则 → B
        使用 hashlib.md5 保证确定性。

        Args:
            session_id: 会话 ID

        Returns:
            ABBucket.BUCKET_A 或 ABBucket.BUCKET_B

        异常降级：
            分桶逻辑异常时降级为 Bucket A，记录 warning 日志
        """
        try:
            with self._config_lock:
                bucket_a_ratio = self._config.bucket_a_ratio

            # 使用 MD5 哈希保证确定性
            hash_hex = hashlib.md5(session_id.encode("utf-8")).hexdigest()
            hash_int = int(hash_hex[:8], 16)  # 取前 8 位十六进制
            bucket_value = (hash_int % 1000) / 1000.0

            if bucket_value < bucket_a_ratio:
                return ABBucket.BUCKET_A
            return ABBucket.BUCKET_B

        except Exception as exc:
            # 异常降级为 Bucket A
            logger.warning(
                "分桶逻辑异常，降级为 Bucket A：session_id=%s, error=%s",
                session_id,
                exc,
            )
            return ABBucket.BUCKET_A

    def get_strategy(self, bucket: ABBucket) -> RAGStrategy:
        """获取指定 Bucket 的 RAG 策略。

        Args:
            bucket: 分桶标识

        Returns:
            对应的 RAG 策略
        """
        with self._config_lock:
            if bucket == ABBucket.BUCKET_A:
                return self._config.bucket_a_strategy
            return self._config.bucket_b_strategy

    def record_metrics(
        self,
        bucket: ABBucket,
        latency_ms: int,
        tokens: int,
        feedback: str | None = None,
    ) -> None:
        """记录 Bucket 指标。

        Args:
            bucket: 分桶标识
            latency_ms: 响应延迟（毫秒）
            tokens: 总 Token 数
            feedback: 用户反馈（positive/negative/None）
        """
        with self._metrics_lock:
            if bucket == ABBucket.BUCKET_A:
                metrics = self._bucket_a_metrics
            else:
                metrics = self._bucket_b_metrics

            metrics.total_requests += 1
            metrics.total_latency_ms += latency_ms
            metrics.total_tokens += tokens

            if feedback == "positive":
                metrics.feedback_positive += 1
            elif feedback == "negative":
                metrics.feedback_negative += 1

        # 持久化指标
        if self._metrics_file:
            self._save_metrics_to_file()

        logger.debug(
            "记录 A/B 测试指标：bucket=%s, latency_ms=%d, tokens=%d, feedback=%s",
            bucket,
            latency_ms,
            tokens,
            feedback,
        )

    def update_config(self, config: ABConfigRequest) -> None:
        """动态更新策略配置，立即生效。

        使用线程锁确保策略更新不影响正在执行的请求。
        仅更新非 None 字段，更新后立即生效。

        Args:
            config: 配置更新请求
        """
        with self._config_lock:
            if config.bucket_a_ratio is not None:
                # 校验范围
                if not 0.0 <= config.bucket_a_ratio <= 1.0:
                    raise ValueError(
                        f"bucket_a_ratio 必须在 [0.0, 1.0] 范围内，当前值：{config.bucket_a_ratio}"
                    )
                self._config.bucket_a_ratio = config.bucket_a_ratio

            if config.bucket_a_strategy is not None:
                strategy = self._validate_strategy(config.bucket_a_strategy)
                self._config.bucket_a_strategy = strategy

            if config.bucket_b_strategy is not None:
                strategy = self._validate_strategy(config.bucket_b_strategy)
                self._config.bucket_b_strategy = strategy

            if config.enabled is not None:
                self._config.enabled = config.enabled

        logger.info(
            "A/B 测试配置已更新：bucket_a_ratio=%.2f, "
            "bucket_a_strategy=%s, bucket_b_strategy=%s, enabled=%s",
            self._config.bucket_a_ratio,
            self._config.bucket_a_strategy,
            self._config.bucket_b_strategy,
            self._config.enabled,
        )

    def get_config(self) -> ABConfigResponse:
        """获取当前配置。

        Returns:
            当前 A/B 测试配置
        """
        with self._config_lock:
            return ABConfigResponse(
                bucket_a_ratio=self._config.bucket_a_ratio,
                bucket_a_strategy=self._config.bucket_a_strategy,
                bucket_b_strategy=self._config.bucket_b_strategy,
                enabled=self._config.enabled,
            )

    def get_metrics(self) -> ABMetricsResponse:
        """获取各 Bucket 指标。

        Returns:
            各 Bucket 的累计指标统计
        """
        with self._metrics_lock:
            # 计算 Bucket A 平均延迟
            if self._bucket_a_metrics.total_requests > 0:
                bucket_a_avg_latency = (
                    self._bucket_a_metrics.total_latency_ms
                    / self._bucket_a_metrics.total_requests
                )
            else:
                bucket_a_avg_latency = 0.0

            # 计算 Bucket B 平均延迟
            if self._bucket_b_metrics.total_requests > 0:
                bucket_b_avg_latency = (
                    self._bucket_b_metrics.total_latency_ms
                    / self._bucket_b_metrics.total_requests
                )
            else:
                bucket_b_avg_latency = 0.0

            return ABMetricsResponse(
                bucket_a_avg_latency_ms=bucket_a_avg_latency,
                bucket_b_avg_latency_ms=bucket_b_avg_latency,
                bucket_a_total_requests=self._bucket_a_metrics.total_requests,
                bucket_b_total_requests=self._bucket_b_metrics.total_requests,
            )

    def is_enabled(self) -> bool:
        """检查 A/B 测试是否启用。

        Returns:
            是否启用
        """
        with self._config_lock:
            return self._config.enabled

    def get_bucket_metrics_detail(self, bucket: ABBucket) -> BucketMetrics:
        """获取指定 Bucket 的详细指标。

        Args:
            bucket: 分桶标识

        Returns:
            Bucket 的详细指标
        """
        with self._metrics_lock:
            if bucket == ABBucket.BUCKET_A:
                return BucketMetrics(
                    total_requests=self._bucket_a_metrics.total_requests,
                    total_latency_ms=self._bucket_a_metrics.total_latency_ms,
                    total_tokens=self._bucket_a_metrics.total_tokens,
                    feedback_positive=self._bucket_a_metrics.feedback_positive,
                    feedback_negative=self._bucket_a_metrics.feedback_negative,
                )
            return BucketMetrics(
                total_requests=self._bucket_b_metrics.total_requests,
                total_latency_ms=self._bucket_b_metrics.total_latency_ms,
                total_tokens=self._bucket_b_metrics.total_tokens,
                feedback_positive=self._bucket_b_metrics.feedback_positive,
                feedback_negative=self._bucket_b_metrics.feedback_negative,
            )

    def reset_metrics(self) -> None:
        """重置所有指标。"""
        with self._metrics_lock:
            self._bucket_a_metrics = BucketMetrics()
            self._bucket_b_metrics = BucketMetrics()

        if self._metrics_file and self._metrics_file.exists():
            self._metrics_file.unlink()

        logger.info("A/B 测试指标已重置")

    def _validate_strategy(self, strategy_str: str) -> RAGStrategy:
        """验证并转换策略字符串。

        Args:
            strategy_str: 策略字符串

        Returns:
            RAGStrategy 枚举值

        Raises:
            ValueError: 策略字符串非法
        """
        try:
            return RAGStrategy(strategy_str)
        except ValueError:
            valid_strategies = [s.value for s in RAGStrategy]
            raise ValueError(
                f"非法的 RAG 策略：{strategy_str}，合法值：{valid_strategies}"
            )

    def _save_metrics_to_file(self) -> None:
        """保存指标到 JSON 文件。"""
        if not self._metrics_file:
            return

        try:
            with self._metrics_lock:
                data = {
                    "bucket_a": {
                        "total_requests": self._bucket_a_metrics.total_requests,
                        "total_latency_ms": self._bucket_a_metrics.total_latency_ms,
                        "total_tokens": self._bucket_a_metrics.total_tokens,
                        "feedback_positive": self._bucket_a_metrics.feedback_positive,
                        "feedback_negative": self._bucket_a_metrics.feedback_negative,
                    },
                    "bucket_b": {
                        "total_requests": self._bucket_b_metrics.total_requests,
                        "total_latency_ms": self._bucket_b_metrics.total_latency_ms,
                        "total_tokens": self._bucket_b_metrics.total_tokens,
                        "feedback_positive": self._bucket_b_metrics.feedback_positive,
                        "feedback_negative": self._bucket_b_metrics.feedback_negative,
                    },
                    "last_updated": datetime.now().isoformat(),
                }

            # 确保目录存在
            self._metrics_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self._metrics_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as exc:
            logger.warning("保存 A/B 测试指标失败：%s", exc)

    def _load_metrics_from_file(self) -> None:
        """从 JSON 文件加载历史指标。"""
        if not self._metrics_file or not self._metrics_file.exists():
            return

        try:
            with open(self._metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            with self._metrics_lock:
                # 加载 Bucket A 指标
                if "bucket_a" in data:
                    bucket_a_data = data["bucket_a"]
                    self._bucket_a_metrics = BucketMetrics(
                        total_requests=bucket_a_data.get("total_requests", 0),
                        total_latency_ms=bucket_a_data.get("total_latency_ms", 0.0),
                        total_tokens=bucket_a_data.get("total_tokens", 0),
                        feedback_positive=bucket_a_data.get("feedback_positive", 0),
                        feedback_negative=bucket_a_data.get("feedback_negative", 0),
                    )

                # 加载 Bucket B 指标
                if "bucket_b" in data:
                    bucket_b_data = data["bucket_b"]
                    self._bucket_b_metrics = BucketMetrics(
                        total_requests=bucket_b_data.get("total_requests", 0),
                        total_latency_ms=bucket_b_data.get("total_latency_ms", 0.0),
                        total_tokens=bucket_b_data.get("total_tokens", 0),
                        feedback_positive=bucket_b_data.get("feedback_positive", 0),
                        feedback_negative=bucket_b_data.get("feedback_negative", 0),
                    )

            logger.info(
                "从文件加载 A/B 测试指标：%s，Bucket A 请求数=%d，Bucket B 请求数=%d",
                self._metrics_file,
                self._bucket_a_metrics.total_requests,
                self._bucket_b_metrics.total_requests,
            )

        except Exception as exc:
            logger.warning("加载 A/B 测试指标失败：%s，将使用空指标", exc)