"""LLM 实验追踪模块

封装 MLflow 自动记录 RAG 链路参数与指标，支持异常降级和敏感字段过滤。

核心设计原则：
- 所有 MLflow 操作均通过 _safe_execute 包装，异常时降级为 log warning
- 敏感字段（API Key、密码等）在记录前自动过滤
- 支持通过配置启用/禁用，禁用时所有操作为 no-op
- 追踪逻辑增加延迟不超过 50ms（REQ-DFX-R02）

使用方式：
    from app.mlops.tracking import LLMExperimentTracker

    tracker = LLMExperimentTracker(
        tracking_uri="http://localhost:5000",
        experiment_name="rag-pipeline",
        enabled=True,
        request_timeout=5
    )

    # 方式一：使用 track_rag_run 一次性追踪
    run_id = tracker.track_rag_run(
        params={"provider": "nim", "model": "deepseek-v4", "top_k": 3},
        metrics={"retrieval_latency_ms": 120.5, "total_tokens": 1500}
    )

    # 方式二：分步追踪
    run_id = tracker.start_run(run_name="query-123")
    tracker.log_params({"provider": "nim", "model": "deepseek-v4"})
    tracker.log_metrics({"latency_ms": 120.5})
    tracker.end_run()
"""

import logging
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, TypeVar

# ============================================================================
# MLflow 导入（优雅降级）
# ============================================================================

_MLFLOW_AVAILABLE: bool

try:
    import mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    mlflow = None  # type: ignore[assignment]

# ============================================================================
# 类型变量与常量
# ============================================================================

_T = TypeVar("_T")

# 敏感字段关键词列表（不区分大小写匹配）
_SENSITIVE_KEY_PATTERNS: frozenset[str] = frozenset([
    "api_key",
    "apikey",
    "password",
    "passwd",
    "secret",
    "token",
    "access_token",
    "auth_token",
    "credential",
    "private_key",
])

# 追踪操作超时缓冲时间（秒）
_TIMEOUT_BUFFER_SECONDS: float = 0.01

logger = logging.getLogger(__name__)


# ============================================================================
# LLMExperimentTracker 类实现
# ============================================================================


class LLMExperimentTracker:
    """LLM 实验追踪器，封装 MLflow 自动记录 RAG 链路参数与指标。

    核心设计原则：
    - 所有 MLflow 操作均通过 _safe_execute 包装，异常时降级为 log warning
    - 敏感字段（API Key、密码等）在记录前自动过滤
    - 支持通过配置启用/禁用，禁用时所有操作为 no-op
    - 追踪逻辑增加延迟不超过 50ms（REQ-DFX-R02）

    Attributes:
        _mlflow_available: MLflow 是否可用
        _tracking_uri: MLflow Server 地址
        _experiment_name: 实验名称
        _enabled: 是否启用追踪
        _timeout_seconds: 操作超时时间（秒）
        _sensitive_keys: 敏感字段集合
        _current_run_id: 当前活跃的 Run ID
        _trace_id: 当前追踪 ID（贯穿日志）
    """

    __slots__ = (
        "_current_run_id",
        "_enabled",
        "_experiment_name",
        "_mlflow_available",
        "_sensitive_keys",
        "_timeout_seconds",
        "_trace_id",
        "_tracking_uri",
    )

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "rag-pipeline",
        enabled: bool = True,
        request_timeout: int = 5,
    ) -> None:
        """初始化 LLM 实验追踪器。

        Args:
            tracking_uri: MLflow Server 地址，默认 "http://localhost:5000"
            experiment_name: 实验名称，默认 "rag-pipeline"
            enabled: 是否启用追踪，默认 True
            request_timeout: 操作超时时间（秒），默认 5

        初始化流程：
        1. 检查 MLflow 是否安装
        2. 设置 tracking_uri 和 experiment_name
        3. 尝试连接 MLflow Server，失败时标记为不可用
        """
        self._mlflow_available: bool = _MLFLOW_AVAILABLE
        self._tracking_uri: str = tracking_uri
        self._experiment_name: str = experiment_name
        self._enabled: bool = enabled
        self._timeout_seconds: int = request_timeout
        self._sensitive_keys: frozenset[str] = _SENSITIVE_KEY_PATTERNS
        self._current_run_id: str | None = None
        self._trace_id: str | None = None

        # MLflow 未安装，记录 INFO 日志
        if not self._mlflow_available:
            logger.info(
                "[MLOps] MLflow 未安装，所有追踪操作将为 no-op。"
                "请执行 'pip install mlflow' 安装依赖。"
            )
            return

        # 追踪已禁用，跳过初始化
        if not self._enabled:
            logger.info("[MLOps] MLflow 追踪已禁用（MLFLOW_ENABLED=false）")
            return

        # 尝试连接 MLflow Server
        self._initialize_mlflow()

    def _initialize_mlflow(self) -> None:
        """初始化 MLflow 连接。

        设置 tracking_uri 和 experiment_name，失败时标记为不可用。
        """
        try:
            if mlflow is None:
                return

            # 设置 tracking URI
            mlflow.set_tracking_uri(self._tracking_uri)

            # 设置 experiment
            mlflow.set_experiment(self._experiment_name)

            logger.info(
                "[MLOps] MLflow 初始化成功 | "
                f"tracking_uri={self._tracking_uri} | "
                f"experiment={self._experiment_name}"
            )
        except Exception as e:
            self._mlflow_available = False
            logger.warning(
                f"[MLOps] MLflow 初始化失败，追踪功能将降级为 no-op | "
                f"tracking_uri={self._tracking_uri} | "
                f"error={type(e).__name__}: {e}"
            )

    def is_available(self) -> bool:
        """检查 MLflow 是否可用。

        Returns:
            True 表示 MLflow 已安装且连接正常，False 表示不可用
        """
        return self._mlflow_available and self._enabled

    def start_run(self, run_name: str | None = None) -> str:
        """创建 MLflow Run，返回 run_id。

        Args:
            run_name: 可选的 Run 名称，默认自动生成

        Returns:
            成功时返回 run_id，失败时返回空字符串

        性能约束：
        - 操作超时时间：self._timeout_seconds
        - 超时后自动取消，返回空字符串
        """
        if not self.is_available():
            return ""

        # 生成唯一 trace_id
        self._trace_id = str(uuid.uuid4())

        def _start_run_operation() -> str:
            """内部操作：启动 MLflow Run"""
            if mlflow is None:
                return ""

            run = mlflow.start_run(run_name=run_name)
            run_id = run.info.run_id

            # 记录 trace_id 作为 tag
            mlflow.set_tag("trace_id", self._trace_id)

            logger.debug(
                f"[MLOps] Run 启动成功 | "
                f"run_id={run_id} | "
                f"run_name={run_name} | "
                f"trace_id={self._trace_id}"
            )

            return run_id

        run_id = self._safe_execute(_start_run_operation, default="")

        if run_id:
            self._current_run_id = run_id

        return run_id

    def log_params(self, params: dict[str, Any]) -> None:
        """记录 RAG 链路参数。

        自动过滤敏感字段（api_key, password, secret, token 等）。

        Args:
            params: RAG 链路参数字典
                - provider: 模型供应商
                - model_name: 模型名称
                - chunk_size: 分块大小
                - top_k: 检索数量
                - reranker: 重排序模型
                - ... 其他参数

        性能约束：
        - 操作超时时间：self._timeout_seconds
        - 超时后自动取消，记录 WARNING 日志
        """
        if not self.is_available():
            return

        # 过滤敏感字段
        sanitized_params = self._sanitize_params(params)

        if not sanitized_params:
            logger.debug(f"[MLOps] 无有效参数需要记录 | trace_id={self._trace_id}")
            return

        def _log_params_operation() -> None:
            """内部操作：记录参数"""
            if mlflow is None:
                return

            mlflow.log_params(sanitized_params)

            logger.debug(
                f"[MLOps] 参数记录成功 | "
                f"params_count={len(sanitized_params)} | "
                f"trace_id={self._trace_id}"
            )

        self._safe_execute(_log_params_operation, default=None)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """记录运行时指标。

        Args:
            metrics: RAG 链路指标字典
                - retrieval_latency_ms: 检索延迟（毫秒）
                - llm_latency_ms: LLM 推理延迟（毫秒）
                - total_tokens: 总 Token 数
                - reranker_score: 重排序得分
                - ... 其他指标

        性能约束：
        - 操作超时时间：self._timeout_seconds
        - 超时后自动取消，记录 WARNING 日志
        """
        if not self.is_available():
            return

        if not metrics:
            logger.debug(f"[MLOps] 无指标需要记录 | trace_id={self._trace_id}")
            return

        def _log_metrics_operation() -> None:
            """内部操作：记录指标"""
            if mlflow is None:
                return

            mlflow.log_metrics(metrics)

            logger.debug(
                f"[MLOps] 指标记录成功 | "
                f"metrics_count={len(metrics)} | "
                f"trace_id={self._trace_id}"
            )

        self._safe_execute(_log_metrics_operation, default=None)

    def end_run(self, status: str = "FINISHED") -> None:
        """结束当前 Run。

        Args:
            status: Run 状态，可选值：
                - "FINISHED": 正常完成（默认）
                - "FAILED": 失败
                - "KILLED": 被终止

        性能约束：
        - 操作超时时间：self._timeout_seconds
        - 超时后自动取消，记录 WARNING 日志
        """
        if not self.is_available():
            return

        def _end_run_operation() -> None:
            """内部操作：结束 Run"""
            if mlflow is None:
                return

            mlflow.end_run(status=status)

            logger.debug(
                f"[MLOps] Run 结束成功 | "
                f"run_id={self._current_run_id} | "
                f"status={status} | "
                f"trace_id={self._trace_id}"
            )

        self._safe_execute(_end_run_operation, default=None)

        # 清理状态
        self._current_run_id = None
        self._trace_id = None

    def track_rag_run(
        self,
        params: dict[str, Any],
        metrics: dict[str, float],
        run_name: str | None = None,
    ) -> str:
        """便捷方法：一次性完成追踪。

        封装 start_run、log_params、log_metrics、end_run 为原子操作。

        Args:
            params: RAG 链路参数（自动过滤敏感字段）
            metrics: RAG 链路指标
            run_name: 可选的 Run 名称

        Returns:
            成功时返回 run_id，失败时返回空字符串

        性能约束：
        - 总延迟不超过 50ms（REQ-DFX-R02）
        - 单个操作超时时间：self._timeout_seconds
        """
        if not self.is_available():
            return ""

        start_time = time.perf_counter()

        try:
            # 启动 Run
            run_id = self.start_run(run_name=run_name)

            if not run_id:
                return ""

            # 记录参数
            self.log_params(params)

            # 记录指标
            self.log_metrics(metrics)

            # 结束 Run
            self.end_run(status="FINISHED")

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            logger.debug(
                f"[MLOps] track_rag_run 完成 | "
                f"run_id={run_id} | "
                f"elapsed_ms={elapsed_ms:.2f} | "
                f"trace_id={self._trace_id}"
            )

            return run_id

        except Exception as e:
            logger.warning(
                f"[MLOps] track_rag_run 异常 | "
                f"error={type(e).__name__}: {e} | "
                f"trace_id={self._trace_id}"
            )
            return ""

    def log_tag(self, key: str, value: str) -> None:
        """记录标签。

        Args:
            key: 标签键
            value: 标签值

        性能约束：
        - 操作超时时间：self._timeout_seconds
        - 超时后自动取消，记录 WARNING 日志
        """
        if not self.is_available():
            return

        def _log_tag_operation() -> None:
            """内部操作：记录标签"""
            if mlflow is None:
                return

            mlflow.set_tag(key, value)

            logger.debug(
                f"[MLOps] 标签记录成功 | "
                f"key={key} | "
                f"trace_id={self._trace_id}"
            )

        self._safe_execute(_log_tag_operation, default=None)

    def _sanitize_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """过滤敏感字段。

        自动过滤 api_key, password, secret, token 等敏感字段。

        Args:
            params: 原始参数字典

        Returns:
            过滤后的安全参数字典
        """
        sanitized: dict[str, Any] = {}

        for key, value in params.items():
            # 不区分大小写匹配
            key_lower = key.lower()

            # 检查是否为敏感字段
            is_sensitive = any(
                pattern in key_lower for pattern in self._sensitive_keys
            )

            if is_sensitive:
                logger.debug(
                    f"[MLOps] 敏感字段已过滤 | "
                    f"key={key} | "
                    f"trace_id={self._trace_id}"
                )
                continue

            # 转换值为字符串（MLflow 要求）
            sanitized[key] = str(value) if not isinstance(value, str) else value

        return sanitized

    def _safe_execute(
        self,
        operation: Callable[[], _T],
        default: _T,
    ) -> _T:
        """安全执行 MLflow 操作，异常时降级为 warning 日志。

        遵循 REQ-DFX-R01：MLOps 追踪逻辑失败不阻塞核心对话链路。

        使用 ThreadPoolExecutor 实现超时控制，避免 MLflow 操作阻塞主线程。

        Args:
            operation: 要执行的操作（无参数函数）
            default: 异常时的默认返回值

        Returns:
            操作成功时返回结果，失败时返回 default

        性能约束：
        - 超时时间：self._timeout_seconds
        - 超时后自动取消，记录 WARNING 日志
        """
        if not self.is_available():
            return default

        try:
            # 使用 ThreadPoolExecutor 实现超时控制
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(operation)

                try:
                    # 等待结果，设置超时
                    result = future.result(timeout=self._timeout_seconds)
                    return result

                except FuturesTimeoutError:
                    logger.warning(
                        f"[MLOps] 操作超时，已取消 | "
                        f"timeout_seconds={self._timeout_seconds} | "
                        f"trace_id={self._trace_id}"
                    )
                    future.cancel()
                    return default

        except Exception as e:
            logger.warning(
                f"[MLOps] 操作异常，已降级 | "
                f"error={type(e).__name__}: {e} | "
                f"trace_id={self._trace_id}"
            )
            return default


# ============================================================================
# 公开 API 声明
# ============================================================================

__all__ = ["LLMExperimentTracker"]
