"""风格审查 Worker Agent

聚焦代码风格检查和命名规范验证，包括 PEP8/PEP20、
snake_case、PascalCase、行宽、注释规范等风格问题。
"""

import logging
import time
from typing import Any

from langchain_core.language_models import BaseChatModel

from app.agent.review.base_worker import BaseWorkerAgent
from app.core.enums import ReviewStatus, ReviewType
from app.models.schemas import WorkerResult

logger = logging.getLogger(__name__)

__all__ = ["StyleAgent"]

_STYLE_SYSTEM_PROMPT = """你是一位资深的代码风格审查专家，专注于代码风格检查和命名规范验证。

## 审查范围

请对以下代码进行风格审查，重点关注：

1. **PEP8 规范**
   - 缩进：使用 4 个空格缩进
   - 行宽：每行不超过 88 字符（Black 默认）
   - 空行：类之间 2 个空行，方法之间 1 个空行
   - 导入顺序：标准库 → 第三方库 → 本地模块
   - 尾随空格和行尾空白

2. **PEP20（Python 之禅）**
   - 显式优于隐式
   - 简单优于复杂
   - 可读性很重要

3. **命名规范**
   - 函数和变量：snake_case（如 get_user_info）
   - 类名：PascalCase（如 UserService）
   - 常量：UPPER_SNAKE_CASE（如 MAX_RETRY_COUNT）
   - 私有成员：_前缀（如 _internal_state）
   - 避免单字母变量名（循环变量 i/j/k 除外）

4. **注释规范**
   - 模块级 docstring
   - 类和函数的 docstring（Google 风格或 NumPy 风格）
   - 行内注释与代码间隔 2 个空格
   - 避免无意义的注释

5. **类型标注**
   - 函数参数和返回值是否有类型标注
   - 是否使用 Python 3.10+ 语法（如 str | None 代替 Optional[str]）
   - 是否使用泛型和精确类型

6. **代码组织**
   - 函数长度是否合理（建议不超过 50 行）
   - 类的职责是否单一
   - 是否有过深的嵌套（建议不超过 3 层）

7. **字符串格式化**
   - 优先使用 f-string
   - 避免使用 % 格式化和 .format()

8. **异常处理风格**
   - 避免裸 except
   - 使用具体的异常类型
   - 异常消息是否清晰

## 输出格式

请以 JSON 格式输出审查结果，格式如下：
```json
{
  "findings": [
    {
      "severity": "critical|high|medium|low|info",
      "description": "问题描述",
      "location": "问题所在代码位置（行号或标识符名）",
      "suggestion": "修复建议"
    }
  ]
}
```

## 严重程度定义

- **critical**: 严重风格问题，如完全不符合命名规范导致代码不可读
- **high**: 重要风格问题，如缺少类型标注、函数过长
- **medium**: 中等风格问题，如命名不够规范、缺少 docstring
- **low**: 轻微风格问题，如行宽略超、注释不够完善
- **info**: 风格改进建议，如推荐使用更 Pythonic 的写法

如果没有发现风格问题，返回空的 findings 列表。只输出 JSON，不要输出其他内容。"""


class StyleAgent(BaseWorkerAgent):
    """风格审查 Worker Agent

    聚焦代码风格检查和命名规范验证，包括 PEP8/PEP20、
    snake_case、PascalCase、行宽、注释规范等风格问题。
    """

    @property
    def dimension(self) -> str:
        """审查维度标识"""
        return "style"

    @property
    def system_prompt(self) -> str:
        """风格审查系统提示词"""
        return _STYLE_SYSTEM_PROMPT

    def __init__(
        self,
        llm: BaseChatModel,
        worker_timeout_seconds: int = 60,
    ) -> None:
        """初始化风格审查 Agent

        Args:
            llm: LangChain BaseChatModel 实例
            worker_timeout_seconds: Worker 执行超时时间（秒）
        """
        super().__init__(llm=llm, worker_timeout_seconds=worker_timeout_seconds)

    def execute(self, code_content: str, context: dict[str, Any] | None = None) -> WorkerResult:
        """执行风格审查

        调用 LLM 对代码进行风格检查和命名规范验证，解析返回结果为 WorkerResult。

        Args:
            code_content: 待审查的代码内容
            context: 可选的上下文信息

        Returns:
            WorkerResult 包含风格审查结果
        """
        start_time = time.monotonic()

        user_content = f"请对以下代码进行风格审查：\n\n```\n{code_content}\n```"
        raw_response = self._invoke_llm(user_content)

        findings = self._parse_findings(raw_response)
        duration_ms = int((time.monotonic() - start_time) * 1000)

        return WorkerResult(
            dimension=ReviewType.STYLE,
            status=ReviewStatus.COMPLETED,
            findings=findings,
            duration_ms=duration_ms,
        )
