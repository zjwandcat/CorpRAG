"""Worker Agent 抽象基类模块

定义所有审查 Worker Agent 的公共接口和通用逻辑，
包括超时控制、异常捕获与 WorkerResult 状态标记、
LLM 返回结果的 JSON 解析。
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel

from app.core.enums import ReviewStatus, ReviewType, Severity
from app.models.schemas import ReviewFinding, WorkerResult

logger = logging.getLogger(__name__)

__all__ = ["BaseWorkerAgent"]


class BaseWorkerAgent(ABC):
    """Worker Agent 抽象基类

    所有审查 Worker Agent（Security、Architecture、Performance、Style、Summary）
    的公共基类，定义统一的接口协议和通用执行逻辑。

    子类必须实现：
        - dimension: 审查维度标识
        - system_prompt: Worker 专属系统提示词
        - execute(): 执行审查任务

    通用逻辑：
        - 超时控制（使用 worker_timeout_seconds 配置）
        - 异常捕获与 WorkerResult 状态标记
        - Worker 间禁止直接通信

    Attributes:
        llm: LangChain BaseChatModel 实例，用于调用 LLM
        worker_timeout_seconds: Worker 执行超时时间（秒）
    """

    def __init__(
        self,
        llm: BaseChatModel,
        worker_timeout_seconds: int = 60,
    ) -> None:
        """初始化 Worker Agent

        Args:
            llm: LangChain BaseChatModel 实例
            worker_timeout_seconds: Worker 执行超时时间（秒），默认 60
        """
        self._llm = llm
        self._worker_timeout_seconds = worker_timeout_seconds

    @property
    @abstractmethod
    def dimension(self) -> str:
        """审查维度标识

        Returns:
            维度名称字符串，如 "security"、"architecture" 等
        """

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Worker 专属系统提示词

        Returns:
            系统提示词字符串，定义该 Worker 的审查聚焦领域和输出格式
        """

    @abstractmethod
    def execute(self, code_content: str, context: dict[str, Any] | None = None) -> WorkerResult:
        """执行审查任务

        子类实现具体的审查逻辑，调用 LLM 执行审查并解析返回结果。

        Args:
            code_content: 待审查的代码内容
            context: 可选的上下文信息（如审查类型、会话 ID 等）

        Returns:
            WorkerResult 包含审查维度、状态、发现问题和耗时
        """

    def _invoke_llm(self, user_content: str) -> str:
        """调用 LLM 执行推理

        复用现有 tools.py 中 _invoke_llm 的模式，
        使用 SystemMessage + HumanMessage 结构化调用。

        Args:
            user_content: 用户消息内容（通常包含待审查代码）

        Returns:
            LLM 返回的文本内容
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_content),
        ]
        response = self._llm.invoke(messages)
        content = response.content
        return str(content) if content else ""

    def _parse_findings(self, raw_response: str) -> list[ReviewFinding]:
        """解析 LLM 返回的审查结果

        尝试从 LLM 返回的文本中提取 JSON 格式的审查结果，
        解析失败时返回空列表并记录警告日志。

        Args:
            raw_response: LLM 返回的原始文本

        Returns:
            审查问题列表
        """
        try:
            # 尝试提取 JSON 块
            json_str = raw_response
            if "```json" in raw_response:
                json_str = raw_response.split("```json")[1].split("```")[0]
            elif "```" in raw_response:
                json_str = raw_response.split("```")[1].split("```")[0]

            parsed = json.loads(json_str.strip())
            raw_findings = parsed.get("findings", [])

            findings: list[ReviewFinding] = []
            for item in raw_findings:
                severity_str = item.get("severity", "info")
                try:
                    severity = Severity(severity_str)
                except ValueError:
                    logger.warning("未知的严重程度：%s，降级为 info", severity_str)
                    severity = Severity.INFO

                findings.append(
                    ReviewFinding(
                        severity=severity,
                        description=item.get("description", ""),
                        location=item.get("location", ""),
                        suggestion=item.get("suggestion"),
                    )
                )

            return findings

        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            logger.warning("审查结果解析失败：%s，返回空列表", exc)
            return []

    def safe_execute(
        self, code_content: str, context: dict[str, Any] | None = None
    ) -> WorkerResult:
        """安全执行审查任务，含超时控制和异常捕获

        包装 execute() 方法，提供统一的超时控制和异常处理。
        当 execute() 抛出异常或超时时，返回标记为 failed/timeout 的 WorkerResult。

        Args:
            code_content: 待审查的代码内容
            context: 可选的上下文信息

        Returns:
            WorkerResult 包含审查结果或错误状态
        """
        start_time = time.monotonic()
        dimension_value = self.dimension

        try:
            logger.info(
                "Worker [%s] 开始执行审查，超时设置：%ds",
                dimension_value,
                self._worker_timeout_seconds,
            )

            result = self.execute(code_content, context)

            duration_ms = int((time.monotonic() - start_time) * 1000)
            # 确保耗时被正确记录
            if result.duration_ms == 0:
                result = result.model_copy(update={"duration_ms": duration_ms})

            logger.info(
                "Worker [%s] 审查完成，状态：%s，耗时：%dms",
                dimension_value,
                result.status,
                result.duration_ms,
            )
            return result

        except TimeoutError:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            logger.warning(
                "Worker [%s] 执行超时（%ds），标记为 timeout",
                dimension_value,
                self._worker_timeout_seconds,
            )
            return WorkerResult(
                dimension=ReviewType(dimension_value),
                status=ReviewStatus.TIMEOUT,
                findings=[],
                duration_ms=duration_ms,
                error_message=f"审查超时（{self._worker_timeout_seconds}s）",
            )

        except Exception as exc:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            logger.error(
                "Worker [%s] 执行失败：%s",
                dimension_value,
                exc,
                exc_info=True,
            )
            return WorkerResult(
                dimension=ReviewType(dimension_value),
                status=ReviewStatus.FAILED,
                findings=[],
                duration_ms=duration_ms,
                error_message=f"审查失败：{exc!s}",
            )
