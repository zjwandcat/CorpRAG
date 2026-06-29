"""安全审查 Worker Agent

聚焦安全漏洞扫描和代码注入检测，包括 SQL 注入、XSS、
命令注入、不安全的反序列化等常见安全问题。
"""

import logging
import time
from typing import Any

from langchain_core.language_models import BaseChatModel

from app.agent.review.base_worker import BaseWorkerAgent
from app.core.enums import ReviewStatus, ReviewType
from app.models.schemas import WorkerResult

logger = logging.getLogger(__name__)

__all__ = ["SecurityAgent"]

_SECURITY_SYSTEM_PROMPT = """你是一位资深的安全审查专家，专注于代码安全漏洞扫描和代码注入检测。

## 审查范围

请对以下代码进行安全审查，重点关注：

1. **SQL 注入**：检查是否存在未参数化的 SQL 查询、字符串拼接 SQL 语句
2. **XSS（跨站脚本攻击）**：检查是否存在未转义的用户输入直接输出到 HTML
3. **命令注入**：检查是否存在未过滤的用户输入传入 os.system、subprocess 等命令执行函数
4. **不安全的反序列化**：检查是否使用 pickle、yaml.load 等不安全的反序列化方法
5. **硬编码凭据**：检查是否存在硬编码的 API Key、密码、Token
6. **不安全的加密**：检查是否使用弱加密算法（MD5、SHA1）或不安全的随机数生成器
7. **路径遍历**：检查是否存在未验证的文件路径操作
8. **敏感信息泄露**：检查是否在日志或响应中暴露敏感信息
9. **不安全的依赖**：检查是否引入已知存在漏洞的第三方库
10. **认证与授权缺陷**：检查是否存在权限绕过、越权访问等问题

## 输出格式

请以 JSON 格式输出审查结果，格式如下：
```json
{
  "findings": [
    {
      "severity": "critical|high|medium|low|info",
      "description": "问题描述",
      "location": "问题所在代码位置（行号或函数名）",
      "suggestion": "修复建议"
    }
  ]
}
```

## 严重程度定义

- **critical**: 可被直接利用的安全漏洞，如远程代码执行、SQL 注入
- **high**: 严重的安全风险，如硬编码密码、认证绕过
- **medium**: 中等安全风险，如不安全的随机数、信息泄露
- **low**: 低安全风险，如缺少安全头配置
- **info**: 安全建议，如推荐使用更安全的替代方案

如果没有发现安全问题，返回空的 findings 列表。只输出 JSON，不要输出其他内容。"""


class SecurityAgent(BaseWorkerAgent):
    """安全审查 Worker Agent

    聚焦安全漏洞扫描和代码注入检测，包括 SQL 注入、XSS、
    命令注入、不安全的反序列化等常见安全问题。
    """

    @property
    def dimension(self) -> str:
        """审查维度标识"""
        return "security"

    @property
    def system_prompt(self) -> str:
        """安全审查系统提示词"""
        return _SECURITY_SYSTEM_PROMPT

    def __init__(
        self,
        llm: BaseChatModel,
        worker_timeout_seconds: int = 60,
    ) -> None:
        """初始化安全审查 Agent

        Args:
            llm: LangChain BaseChatModel 实例
            worker_timeout_seconds: Worker 执行超时时间（秒）
        """
        super().__init__(llm=llm, worker_timeout_seconds=worker_timeout_seconds)

    def execute(self, code_content: str, context: dict[str, Any] | None = None) -> WorkerResult:
        """执行安全审查

        调用 LLM 对代码进行安全漏洞扫描，解析返回结果为 WorkerResult。

        Args:
            code_content: 待审查的代码内容
            context: 可选的上下文信息

        Returns:
            WorkerResult 包含安全审查结果
        """
        start_time = time.monotonic()

        user_content = f"请对以下代码进行安全审查：\n\n```\n{code_content}\n```"
        raw_response = self._invoke_llm(user_content)

        findings = self._parse_findings(raw_response)
        duration_ms = int((time.monotonic() - start_time) * 1000)

        return WorkerResult(
            dimension=ReviewType.SECURITY,
            status=ReviewStatus.COMPLETED,
            findings=findings,
            duration_ms=duration_ms,
        )
