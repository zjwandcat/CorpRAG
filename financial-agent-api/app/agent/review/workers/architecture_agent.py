"""架构审查 Worker Agent

聚焦架构合规性检查和设计模式评估，包括 SOLID 原则、
DRY 原则、耦合度、内聚性等架构层面的问题。
"""

import logging
import time
from typing import Any

from langchain_core.language_models import BaseChatModel

from app.agent.review.base_worker import BaseWorkerAgent
from app.core.enums import ReviewStatus, ReviewType
from app.models.schemas import WorkerResult

logger = logging.getLogger(__name__)

__all__ = ["ArchitectureAgent"]

_ARCHITECTURE_SYSTEM_PROMPT = """你是一位资深的软件架构师，专注于架构合规性检查和设计模式评估。

## 审查范围

请对以下代码进行架构审查，重点关注：

1. **SOLID 原则**
   - 单一职责原则（SRP）：类/函数是否只有一个变更原因
   - 开闭原则（OCP）：是否通过扩展而非修改来添加功能
   - 里氏替换原则（LSP）：子类是否可以替换父类
   - 接口隔离原则（ISP）：接口是否精简专一
   - 依赖倒置原则（DIP）：是否依赖抽象而非具体实现

2. **DRY 原则**：是否存在重复代码、重复逻辑

3. **耦合度**：模块之间是否存在过度耦合、循环依赖

4. **内聚性**：模块内部是否高度内聚、职责清晰

5. **设计模式**：是否合理使用设计模式，是否存在反模式

6. **分层架构**：是否遵循分层架构原则（表现层、业务层、数据层）

7. **异常处理**：异常处理是否合理，是否吞没异常或过度捕获

8. **接口设计**：API 接口是否遵循 RESTful 规范，是否合理设计

9. **可扩展性**：代码是否易于扩展新功能

10. **可测试性**：代码是否易于编写单元测试

## 输出格式

请以 JSON 格式输出审查结果，格式如下：
```json
{
  "findings": [
    {
      "severity": "critical|high|medium|low|info",
      "description": "问题描述",
      "location": "问题所在代码位置（类名或函数名）",
      "suggestion": "修复建议"
    }
  ]
}
```

## 严重程度定义

- **critical**: 严重的架构缺陷，如循环依赖导致系统不可维护
- **high**: 重要的架构问题，如违反 SOLID 原则导致扩展困难
- **medium**: 中等架构问题，如代码重复、耦合度偏高
- **low**: 轻微架构问题，如命名不够清晰
- **info**: 架构改进建议，如推荐使用特定设计模式

如果没有发现架构问题，返回空的 findings 列表。只输出 JSON，不要输出其他内容。"""


class ArchitectureAgent(BaseWorkerAgent):
    """架构审查 Worker Agent

    聚焦架构合规性检查和设计模式评估，包括 SOLID 原则、
    DRY 原则、耦合度、内聚性等架构层面的问题。
    """

    @property
    def dimension(self) -> str:
        """审查维度标识"""
        return "architecture"

    @property
    def system_prompt(self) -> str:
        """架构审查系统提示词"""
        return _ARCHITECTURE_SYSTEM_PROMPT

    def __init__(
        self,
        llm: BaseChatModel,
        worker_timeout_seconds: int = 60,
    ) -> None:
        """初始化架构审查 Agent

        Args:
            llm: LangChain BaseChatModel 实例
            worker_timeout_seconds: Worker 执行超时时间（秒）
        """
        super().__init__(llm=llm, worker_timeout_seconds=worker_timeout_seconds)

    def execute(self, code_content: str, context: dict[str, Any] | None = None) -> WorkerResult:
        """执行架构审查

        调用 LLM 对代码进行架构合规性检查，解析返回结果为 WorkerResult。

        Args:
            code_content: 待审查的代码内容
            context: 可选的上下文信息

        Returns:
            WorkerResult 包含架构审查结果
        """
        start_time = time.monotonic()

        user_content = f"请对以下代码进行架构审查：\n\n```\n{code_content}\n```"
        raw_response = self._invoke_llm(user_content)

        findings = self._parse_findings(raw_response)
        duration_ms = int((time.monotonic() - start_time) * 1000)

        return WorkerResult(
            dimension=ReviewType.ARCHITECTURE,
            status=ReviewStatus.COMPLETED,
            findings=findings,
            duration_ms=duration_ms,
        )
