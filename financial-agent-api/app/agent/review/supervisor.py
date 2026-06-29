"""Supervisor Agent 主控协调者模块

实现 Supervisor 模式的多Agent协作调度，负责：
- 解析审查类型，确定需要调度的 Worker Agent 列表
- 使用 concurrent.futures.ThreadPoolExecutor 并行调度多个 Worker Agent
- 收集所有 Worker 结果（含失败项和超时项）
- 调用 SummaryAgent 汇总结果
- 降级策略：Supervisor 调度失败时降级为单 Agent 模式
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable
from uuid import uuid4

from langchain_core.language_models import BaseChatModel

from app.agent.review.constants import REVIEW_TYPE_WORKERS
from app.agent.review.workers.architecture_agent import ArchitectureAgent
from app.agent.review.workers.performance_agent import PerformanceAgent
from app.agent.review.workers.security_agent import SecurityAgent
from app.agent.review.workers.style_agent import StyleAgent
from app.agent.review.workers.summary_agent import SummaryAgent
from app.core.enums import ReviewStatus, ReviewType
from app.models.schemas import ReviewResponse, ReviewResultItem, WorkerResult

logger = logging.getLogger(__name__)

__all__ = ["SupervisorAgent"]


# 降级模式下的单 Agent 系统提示词
_FALLBACK_SYSTEM_PROMPT = """你是一位全面的代码审查专家，请对以下代码进行综合审查，
涵盖安全性、架构设计、性能优化和代码风格等方面。

请以 Markdown 格式输出审查报告，包含以下部分：
1. 审查概要
2. 发现的问题（按严重程度排序）
3. 改进建议

如果没有发现问题，请说明代码质量良好。"""


class SupervisorAgent:
    """Supervisor Agent 主控协调者

    负责任务分发、Worker Agent 调度、结果汇总和最终输出。
    使用 concurrent.futures.ThreadPoolExecutor 并行调度多个 Worker Agent，
    因为现有 LLM 调用是同步的。

    Attributes:
        llm: LangChain BaseChatModel 实例
        worker_timeout_seconds: Worker 执行超时时间（秒）
        max_workers: 线程池最大工作线程数
    """

    def __init__(
        self,
        llm: BaseChatModel,
        worker_timeout_seconds: int = 60,
        max_workers: int = 4,
    ) -> None:
        """初始化 Supervisor Agent

        Args:
            llm: LangChain BaseChatModel 实例
            worker_timeout_seconds: Worker 执行超时时间（秒），默认 60
            max_workers: 线程池最大工作线程数，默认 4
        """
        self._llm = llm
        self._worker_timeout_seconds = worker_timeout_seconds
        self._max_workers = max_workers

        # 初始化所有 Worker Agent
        self._workers: dict[str, Any] = {
            "security": SecurityAgent(llm=llm, worker_timeout_seconds=worker_timeout_seconds),
            "architecture": ArchitectureAgent(
                llm=llm, worker_timeout_seconds=worker_timeout_seconds
            ),
            "performance": PerformanceAgent(llm=llm, worker_timeout_seconds=worker_timeout_seconds),
            "style": StyleAgent(llm=llm, worker_timeout_seconds=worker_timeout_seconds),
        }
        self._summary_agent = SummaryAgent(llm=llm, worker_timeout_seconds=worker_timeout_seconds)

        logger.info(
            "SupervisorAgent 初始化完成，Worker 维度：%s，超时：%ds",
            list(self._workers.keys()),
            worker_timeout_seconds,
        )

    def dispatch(
        self,
        review_request: Any,
        on_worker_start: Callable[[str], None] | None = None,
        on_worker_result: Callable[[WorkerResult], None] | None = None,
        on_summary_start: Callable[[], None] | None = None,
    ) -> ReviewResponse:
        """调度审查任务并返回最终审查报告

        核心方法，实现 Supervisor 模式的完整审查流程：
        1. 解析 review_type，确定需要调度的 Worker Agent 列表
        2. 使用 ThreadPoolExecutor 并行调度多个 Worker Agent
        3. 每个 Worker 独立调用 LLM，设置超时控制
        4. 收集所有 Worker 结果（含失败项和超时项）
        5. 调用 SummaryAgent 汇总结果
        6. 降级策略：Supervisor 调度失败时降级为单 Agent 模式

        Args:
            review_request: 审查请求对象，需包含 code_content、review_type 等字段
            on_worker_start: Worker 开始执行时的回调，参数为维度名称
            on_worker_result: Worker 完成时的回调，参数为 WorkerResult
            on_summary_start: Summary Agent 开始汇总时的回调

        Returns:
            ReviewResponse 包含审查结果和汇总报告
        """
        total_start = time.monotonic()

        # 生成会话 ID 和追踪 ID
        session_id = getattr(review_request, "session_id", None) or str(uuid4())
        trace_id = str(uuid4())
        review_type = getattr(review_request, "review_type", ReviewType.FULL)
        review_type_str = review_type.value if hasattr(review_type, "value") else str(review_type)
        code_content = getattr(review_request, "code_content", "") or ""
        _code_url = getattr(review_request, "code_url", "") or ""

        logger.info(
            "Supervisor 接收审查请求：session_id=%s, trace_id=%s, review_type=%s",
            session_id,
            trace_id,
            review_type_str,
        )

        try:
            # Step 1: 确定需要调度的 Worker 列表
            worker_dimensions = REVIEW_TYPE_WORKERS.get(
                review_type_str,
                REVIEW_TYPE_WORKERS["full"],
            )
            logger.info(
                "审查类型 [%s]，将调度以下 Worker：%s",
                review_type_str,
                worker_dimensions,
            )

            # Step 2: 并行调度 Worker Agent
            worker_results = self._dispatch_workers(
                code_content=code_content,
                worker_dimensions=worker_dimensions,
                trace_id=trace_id,
                on_worker_start=on_worker_start,
                on_worker_result=on_worker_result,
            )

            # Step 3: 调用 SummaryAgent 汇总结果

            logger.info("所有 Worker 执行完毕，调用 SummaryAgent 汇总结果")
            if on_summary_start is not None:
                try:
                    on_summary_start()
                except Exception as cb_exc:
                    logger.warning("on_summary_start 回调异常：%s", cb_exc)
            summary_report = self._summary_agent.generate_summary(
                code_content=code_content,
                worker_results=worker_results,
            )

            # Step 4: 组装 ReviewResponse
            total_duration_ms = int((time.monotonic() - total_start) * 1000)
            result_items = self._build_result_items(worker_results)

            logger.info(
                "审查完成：session_id=%s, 耗时=%dms, 维度数=%d",
                session_id,
                total_duration_ms,
                len(result_items),
            )

            return ReviewResponse(
                session_id=session_id,
                review_type=review_type
                if isinstance(review_type, ReviewType)
                else ReviewType(review_type_str),
                results=result_items,
                summary=summary_report,
                total_duration_ms=total_duration_ms,
                is_fallback=False,
                fallback_message="",
            )

        except Exception as exc:
            # 降级策略：Supervisor 调度失败时降级为单 Agent 模式
            logger.error(
                "Supervisor 调度失败，降级为单 Agent 模式：%s",
                exc,
                exc_info=True,
            )
            return self._fallback_review(
                code_content=code_content,
                session_id=session_id,
                review_type=review_type
                if isinstance(review_type, ReviewType)
                else ReviewType(review_type_str),
                total_start=total_start,
                error_message=str(exc),
            )

    def _dispatch_workers(
        self,
        code_content: str,
        worker_dimensions: list[str],
        trace_id: str,
        on_worker_start: Callable[[str], None] | None = None,
        on_worker_result: Callable[[WorkerResult], None] | None = None,
    ) -> list[WorkerResult]:
        """并行调度多个 Worker Agent

        使用 ThreadPoolExecutor 并行调度，因为现有 LLM 调用是同步的。
        每个 Worker 独立调用 LLM，设置超时控制。
        单个 Worker 失败不影响其余 Worker。

        Args:
            code_content: 待审查的代码内容
            worker_dimensions: 需要调度的 Worker 维度列表
            trace_id: 全链路追踪标识
            on_worker_start: Worker 开始执行时的回调，参数为维度名称
            on_worker_result: Worker 完成时的回调，参数为 WorkerResult

        Returns:
            所有 Worker 的执行结果列表（含失败项和超时项）
        """
        worker_results: list[WorkerResult] = []

        # 如果只有一个 Worker，直接同步执行
        if len(worker_dimensions) == 1:
            dimension = worker_dimensions[0]
            worker = self._workers.get(dimension)
            if worker is None:
                logger.error("未找到维度 [%s] 对应的 Worker", dimension)
                return worker_results

            # 调用 on_worker_start 回调
            if on_worker_start is not None:
                try:
                    on_worker_start(dimension)
                except Exception as cb_exc:
                    logger.warning(
                        "on_worker_start 回调异常（dimension=%s）：%s", dimension, cb_exc
                    )

            context = {"trace_id": trace_id}
            result = worker.safe_execute(code_content=code_content, context=context)
            worker_results.append(result)

            # 调用 on_worker_result 回调
            if on_worker_result is not None:
                try:
                    on_worker_result(result)
                except Exception as cb_exc:
                    logger.warning(
                        "on_worker_result 回调异常（dimension=%s）：%s", dimension, cb_exc
                    )

            return worker_results

        # 多个 Worker 并行执行
        with ThreadPoolExecutor(
            max_workers=min(len(worker_dimensions), self._max_workers)
        ) as executor:
            future_to_dimension: dict[Any, str] = {}

            for dimension in worker_dimensions:
                worker = self._workers.get(dimension)
                if worker is None:
                    logger.error("未找到维度 [%s] 对应的 Worker", dimension)
                    continue

                # 调用 on_worker_start 回调
                if on_worker_start is not None:
                    try:
                        on_worker_start(dimension)
                    except Exception as cb_exc:
                        logger.warning(
                            "on_worker_start 回调异常（dimension=%s）：%s", dimension, cb_exc
                        )

                context = {"trace_id": trace_id}
                future = executor.submit(
                    worker.safe_execute,
                    code_content=code_content,
                    context=context,
                )
                future_to_dimension[future] = dimension

            # 收集结果
            for future in as_completed(
                future_to_dimension, timeout=self._worker_timeout_seconds + 10
            ):
                dimension = future_to_dimension[future]
                try:
                    result = future.result(timeout=self._worker_timeout_seconds)
                    worker_results.append(result)
                    logger.info(
                        "Worker [%s] 完成，状态：%s",
                        dimension,
                        result.status,
                    )

                    # 调用 on_worker_result 回调
                    if on_worker_result is not None:
                        try:
                            on_worker_result(result)
                        except Exception as cb_exc:
                            logger.warning(
                                "on_worker_result 回调异常（dimension=%s）：%s", dimension, cb_exc
                            )

                except TimeoutError:
                    logger.warning("Worker [%s] 超时", dimension)
                    timeout_result = WorkerResult(
                        dimension=ReviewType(dimension),
                        status=ReviewStatus.TIMEOUT,
                        findings=[],
                        duration_ms=self._worker_timeout_seconds * 1000,
                        error_message=f"审查超时（{self._worker_timeout_seconds}s）",
                    )
                    worker_results.append(timeout_result)

                    # 调用 on_worker_result 回调（超时也通知）
                    if on_worker_result is not None:
                        try:
                            on_worker_result(timeout_result)
                        except Exception as cb_exc:
                            logger.warning(
                                "on_worker_result 回调异常（dimension=%s）：%s", dimension, cb_exc
                            )

                except Exception as exc:
                    logger.error("Worker [%s] 执行异常：%s", dimension, exc)
                    error_result = WorkerResult(
                        dimension=ReviewType(dimension),
                        status=ReviewStatus.FAILED,
                        findings=[],
                        duration_ms=0,
                        error_message=f"审查失败：{exc!s}",
                    )
                    worker_results.append(error_result)

                    # 调用 on_worker_result 回调（异常也通知）
                    if on_worker_result is not None:
                        try:
                            on_worker_result(error_result)
                        except Exception as cb_exc:
                            logger.warning(
                                "on_worker_result 回调异常（dimension=%s）：%s", dimension, cb_exc
                            )

        return worker_results

    def _fallback_review(
        self,
        code_content: str,
        session_id: str,
        review_type: ReviewType,
        total_start: float,
        error_message: str,
    ) -> ReviewResponse:
        """降级为单 Agent 模式执行审查

        当 Supervisor 调度失败时，使用单个 LLM 调用执行综合审查。

        Args:
            code_content: 待审查的代码内容
            session_id: 审查会话 ID
            review_type: 审查类型
            total_start: 审查开始时间
            error_message: 降级原因

        Returns:
            ReviewResponse 包含降级审查结果
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            messages = [
                SystemMessage(content=_FALLBACK_SYSTEM_PROMPT),
                HumanMessage(content=f"请对以下代码进行综合审查：\n\n```\n{code_content}\n```"),
            ]
            response = self._llm.invoke(messages)
            summary = str(response.content) if response.content else "审查完成，但未生成报告"
        except Exception as exc:
            logger.error("降级审查也失败：%s", exc)
            summary = f"代码审查服务暂时不可用，请稍后重试。错误信息：{exc!s}"

        total_duration_ms = int((time.monotonic() - total_start) * 1000)

        return ReviewResponse(
            session_id=session_id,
            review_type=review_type,
            results=[],
            summary=summary,
            total_duration_ms=total_duration_ms,
            is_fallback=True,
            fallback_message=f"多Agent协作审查失败，已降级为单Agent模式。原因：{error_message}",
        )

    @staticmethod
    def _build_result_items(worker_results: list[WorkerResult]) -> list[ReviewResultItem]:
        """将 WorkerResult 列表转换为 ReviewResultItem 列表

        Args:
            worker_results: Worker 执行结果列表

        Returns:
            ReviewResultItem 列表
        """
        result_items: list[ReviewResultItem] = []
        for wr in worker_results:
            result_items.append(
                ReviewResultItem(
                    dimension=wr.dimension,
                    status=wr.status,
                    findings=wr.findings,
                    duration_ms=wr.duration_ms,
                )
            )
        return result_items
