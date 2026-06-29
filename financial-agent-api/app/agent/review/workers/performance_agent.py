"""性能审查 Worker Agent

聚焦性能瓶颈分析和资源占用检测，包括算法复杂度、
内存泄漏、N+1 查询、不必要的循环等性能问题。
"""

import logging
import time
from typing import Any

from langchain_core.language_models import BaseChatModel

from app.agent.review.base_worker import BaseWorkerAgent
from app.core.enums import ReviewStatus, ReviewType
from app.models.schemas import WorkerResult

logger = logging.getLogger(__name__)

__all__ = ["PerformanceAgent"]

_PERFORMANCE_SYSTEM_PROMPT = """你是一位资深的性能优化专家，专注于性能瓶颈分析和资源占用检测。

## 审查范围

请对以下代码进行性能审查，重点关注：

1. **算法复杂度**：是否存在时间复杂度或空间复杂度过高的算法
   - O(n²) 及以上的嵌套循环
   - 不必要的递归调用
   - 可用哈希表优化但使用了线性查找的场景

2. **内存泄漏**：是否存在资源未释放、循环引用等问题
   - 文件句柄未关闭
   - 数据库连接未释放
   - 大对象未及时回收

3. **N+1 查询**：数据库操作中是否存在 N+1 查询问题
   - 循环中执行数据库查询
   - 未使用批量查询或预加载

4. **不必要的循环**：是否存在可以用内置函数替代的循环
   - 可用 map/filter/列表推导式替代的 for 循环
   - 可用集合操作替代的逐元素比较

5. **I/O 瓶颈**：是否存在同步阻塞 I/O、频繁的小文件读写
   - 可批量处理的单条 I/O 操作
   - 可异步化的同步 I/O 调用

6. **缓存缺失**：是否存在频繁计算相同结果但未缓存的场景

7. **序列化开销**：是否存在不必要的序列化/反序列化操作

8. **并发问题**：是否存在线程安全问题、锁竞争、死锁风险

9. **资源占用**：是否存在大对象频繁创建、字符串拼接效率低等问题

10. **懒加载缺失**：是否存在启动时加载全部数据但实际只使用部分的场景

## 输出格式

请以 JSON 格式输出审查结果，格式如下：
```json
{
  "findings": [
    {
      "severity": "critical|high|medium|low|info",
      "description": "问题描述",
      "location": "问题所在代码位置（函数名或行号）",
      "suggestion": "修复建议"
    }
  ]
}
```

## 严重程度定义

- **critical**: 严重性能问题，如内存泄漏、死锁风险
- **high**: 重要性能瓶颈，如 N+1 查询、O(n³) 算法
- **medium**: 中等性能问题，如不必要的循环、缺少缓存
- **low**: 轻微性能问题，如字符串拼接效率低
- **info**: 性能优化建议，如推荐使用更高效的数据结构

如果没有发现性能问题，返回空的 findings 列表。只输出 JSON，不要输出其他内容。"""


class PerformanceAgent(BaseWorkerAgent):
    """性能审查 Worker Agent

    聚焦性能瓶颈分析和资源占用检测，包括算法复杂度、
    内存泄漏、N+1 查询、不必要的循环等性能问题。
    """

    @property
    def dimension(self) -> str:
        """审查维度标识"""
        return "performance"

    @property
    def system_prompt(self) -> str:
        """性能审查系统提示词"""
        return _PERFORMANCE_SYSTEM_PROMPT

    def __init__(
        self,
        llm: BaseChatModel,
        worker_timeout_seconds: int = 60,
    ) -> None:
        """初始化性能审查 Agent

        Args:
            llm: LangChain BaseChatModel 实例
            worker_timeout_seconds: Worker 执行超时时间（秒）
        """
        super().__init__(llm=llm, worker_timeout_seconds=worker_timeout_seconds)

    def execute(self, code_content: str, context: dict[str, Any] | None = None) -> WorkerResult:
        """执行性能审查

        调用 LLM 对代码进行性能瓶颈分析，解析返回结果为 WorkerResult。

        Args:
            code_content: 待审查的代码内容
            context: 可选的上下文信息

        Returns:
            WorkerResult 包含性能审查结果
        """
        start_time = time.monotonic()

        user_content = f"请对以下代码进行性能审查：\n\n```\n{code_content}\n```"
        raw_response = self._invoke_llm(user_content)

        findings = self._parse_findings(raw_response)
        duration_ms = int((time.monotonic() - start_time) * 1000)

        return WorkerResult(
            dimension=ReviewType.PERFORMANCE,
            status=ReviewStatus.COMPLETED,
            findings=findings,
            duration_ms=duration_ms,
        )
