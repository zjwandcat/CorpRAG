"""结果汇总 Worker Agent

负责汇总所有 Worker Agent 的审查结果，按严重程度排序，
生成最终的 Markdown 格式审查报告。
"""

import logging
import time
from typing import Any

from langchain_core.language_models import BaseChatModel

from app.agent.review.base_worker import BaseWorkerAgent
from app.core.enums import ReviewStatus, ReviewType, Severity
from app.models.schemas import ReviewFinding, WorkerResult

logger = logging.getLogger(__name__)

__all__ = ["SummaryAgent"]

_SUMMARY_SYSTEM_PROMPT = """\
你是一位资深的技术审查报告撰写专家，\
负责汇总多个维度的代码审查结果并生成最终报告。

## 任务

你将收到来自多个审查维度的结果，请完成以下工作：

1. **汇总所有审查结果**：整合安全、架构、性能、风格等维度的发现
2. **按严重程度排序**：critical → high → medium → low → info
3. **生成最终 Markdown 报告**：结构清晰、内容完整

## 报告格式

请生成以下格式的 Markdown 报告：

```markdown
# 代码审查报告

## 审查概要

- **审查类型**：[full/security/architecture/performance/style]
- **审查维度**：[列出参与的审查维度]
- **总体评估**：[通过/需修改/需重写]

## 严重问题（Critical）

### [维度名] - [问题标题]
- **位置**：[代码位置]
- **描述**：[问题描述]
- **建议**：[修复建议]

## 重要问题（High）

### [维度名] - [问题标题]
...

## 中等问题（Medium）
...

## 轻微问题（Low）
...

## 建议改进（Info）
...

## 总结

[对代码整体质量的评价和改进建议]
```

## 注意事项

- 如果某个维度审查失败或超时，在报告中标注该维度状态
- 严重程度排序优先级：critical > high > medium > low > info
- 同一严重程度内按维度顺序排列
- 报告语言使用中文
- 只输出 Markdown 报告，不要输出其他内容"""


class SummaryAgent(BaseWorkerAgent):
    """结果汇总 Worker Agent

    负责汇总所有 Worker Agent 的审查结果，按严重程度排序，
    生成最终的 Markdown 格式审查报告。
    """

    @property
    def dimension(self) -> str:
        """审查维度标识"""
        return "summary"

    @property
    def system_prompt(self) -> str:
        """汇总审查系统提示词"""
        return _SUMMARY_SYSTEM_PROMPT

    def __init__(
        self,
        llm: BaseChatModel,
        worker_timeout_seconds: int = 60,
    ) -> None:
        """初始化汇总 Agent

        Args:
            llm: LangChain BaseChatModel 实例
            worker_timeout_seconds: Worker 执行超时时间（秒）
        """
        super().__init__(llm=llm, worker_timeout_seconds=worker_timeout_seconds)

    def execute(self, code_content: str, context: dict[str, Any] | None = None) -> WorkerResult:
        """执行结果汇总

        接收所有 Worker 的 WorkerResult，调用 LLM 生成汇总报告。

        Args:
            code_content: 原始代码内容（用于上下文引用）
            context: 必须包含 "worker_results" 键，值为各维度的 WorkerResult 列表

        Returns:
            WorkerResult 包含汇总报告
        """
        start_time = time.monotonic()

        worker_results: list[WorkerResult] = []
        if context and "worker_results" in context:
            worker_results = context["worker_results"]

        # 构建汇总输入
        user_content = self._build_summary_input(code_content, worker_results)
        raw_response = self._invoke_llm(user_content)

        duration_ms = int((time.monotonic() - start_time) * 1000)

        # 将汇总报告文本作为 ReviewFinding 添加到 findings 中
        summary_finding = ReviewFinding(
            severity=Severity.INFO,
            description=raw_response,
            location="汇总报告",
            suggestion=None,
        )

        return WorkerResult(
            dimension=ReviewType.FULL,
            status=ReviewStatus.COMPLETED,
            findings=[summary_finding],
            duration_ms=duration_ms,
            error_message=None,
        )

    def generate_summary(
        self,
        code_content: str,
        worker_results: list[WorkerResult],
    ) -> str:
        """生成汇总报告

        公开接口，供 SupervisorAgent 直接调用获取汇总报告文本。

        Args:
            code_content: 原始代码内容
            worker_results: 各维度的 WorkerResult 列表

        Returns:
            Markdown 格式的汇总报告
        """
        user_content = self._build_summary_input(code_content, worker_results)
        return self._invoke_llm(user_content)

    def _build_summary_input(
        self,
        code_content: str,
        worker_results: list[WorkerResult],
    ) -> str:
        """构建汇总输入文本

        将各维度的审查结果格式化为 LLM 可理解的输入文本。

        Args:
            code_content: 原始代码内容
            worker_results: 各维度的 WorkerResult 列表

        Returns:
            格式化的汇总输入文本
        """
        parts: list[str] = []

        # 添加代码概要（避免传入完整代码导致 token 过多）
        code_lines = code_content.split("\n")
        code_summary = (
            f"代码共 {len(code_lines)} 行\n"
            f"前 20 行预览：\n```\n" + "\n".join(code_lines[:20]) + "\n```"
        )
        parts.append(f"## 待审查代码\n{code_summary}")

        # 添加各维度审查结果
        parts.append("\n## 各维度审查结果\n")
        for result in worker_results:
            dimension_name = (
                result.dimension.value
                if hasattr(result.dimension, "value")
                else str(result.dimension)
            )
            _status_text = (
                result.status.value if hasattr(result.status, "value") else str(result.status)
            )

            if result.status == ReviewStatus.COMPLETED:
                parts.append(f"### {dimension_name} 维度（耗时 {result.duration_ms}ms）")
                if result.findings:
                    for finding in result.findings:
                        severity = (
                            finding.severity.value
                            if hasattr(finding.severity, "value")
                            else str(finding.severity)
                        )
                        parts.append(
                            f"- **[{severity}]** {finding.description}\n"
                            f"  - 位置：{finding.location or '未指定'}\n"
                            f"  - 建议：{finding.suggestion or '无'}"
                        )
                else:
                    parts.append(f"### {dimension_name} 维度 — 未发现问题 ✓")
            elif result.status == ReviewStatus.TIMEOUT:
                parts.append(f"### {dimension_name} 维度 — 审查超时 ⏱")
                if result.error_message:
                    parts.append(f"  - 原因：{result.error_message}")
            elif result.status == ReviewStatus.FAILED:
                parts.append(f"### {dimension_name} 维度 — 审查失败 ✗")
                if result.error_message:
                    parts.append(f"  - 原因：{result.error_message}")

        parts.append("\n请根据以上各维度的审查结果，生成最终的代码审查报告。")

        return "\n".join(parts)
