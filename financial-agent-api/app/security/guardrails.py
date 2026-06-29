"""Dead-loop Guardrails 模块

实现工具调用死循环检测与防护：
- ToolRepetitionDetector：滑动窗口检测同一工具的连续重复调用
- 超过阈值时返回 block 动作，触发 InfiniteLoopDetectedError
- 与 LangGraph recursion_limit 双层协同防御
"""

import json
from collections import deque
from typing import Any

from app.core.enums import GuardrailAction
from app.core.logging_config import get_logger
from app.observability.metrics import GUARDRAIL_INTERVENTION_COUNT

logger = get_logger(__name__)

__all__ = ["ToolRepetitionDetector"]


class ToolRepetitionDetector:
    """工具重复调用检测器

    使用滑动窗口维护最近的工具调用记录，检测同一工具 + 相同参数的
    连续重复调用。超过阈值时返回 block 动作。

    检测逻辑：
    - 窗口内相同调用次数 < max_repetition: ALLOW
    - 窗口内相同调用次数 == max_repetition - 1: WARN
    - 窗口内相同调用次数 >= max_repetition: BLOCK

    Attributes:
        _max_repetition: 同一工具连续调用上限
        _window_size: 滑动窗口大小
        _call_history: 最近工具调用记录的滑动窗口
    """

    def __init__(self, max_repetition: int = 3, window_size: int = 5) -> None:
        """初始化工具重复调用检测器

        Args:
            max_repetition: 同一工具连续调用上限，默认 3
            window_size: 滑动窗口大小，默认 5
        """
        self._max_repetition: int = max_repetition
        self._window_size: int = window_size
        self._call_history: deque[dict[str, Any]] = deque(maxlen=window_size)
        logger.info(
            "ToolRepetitionDetector 初始化完成，max_repetition=%d, window_size=%d",
            max_repetition,
            window_size,
        )

    def check(self, tool_name: str, tool_args: dict[str, Any]) -> GuardrailAction:
        """检测工具调用是否重复

        在滑动窗口内统计同一工具 + 相同参数的出现次数，根据阈值
        返回对应的护栏动作。

        Args:
            tool_name: 工具名称
            tool_args: 工具参数

        Returns:
            GuardrailAction 枚举值：
            - ALLOW: 允许调用
            - WARN: 接近阈值，发出警告
            - BLOCK: 超过阈值，阻止调用
        """
        call_record = {
            "tool_name": tool_name,
            "tool_args": json.dumps(tool_args, sort_keys=True, default=str),
        }

        # 统计窗口内相同调用的次数
        same_call_count = sum(
            1
            for record in self._call_history
            if record["tool_name"] == tool_name
            and record["tool_args"] == call_record["tool_args"]
        )

        # 添加当前调用到窗口
        self._call_history.append(call_record)

        # 判断动作
        if same_call_count >= self._max_repetition:
            action = GuardrailAction.BLOCK
            GUARDRAIL_INTERVENTION_COUNT.labels(
                tool_name=tool_name, action="block"
            ).inc()
            logger.warning(
                "检测到工具调用死循环：%s（重复 %d 次），动作：BLOCK",
                tool_name,
                same_call_count + 1,
            )
        elif same_call_count == self._max_repetition - 1:
            action = GuardrailAction.WARN
            GUARDRAIL_INTERVENTION_COUNT.labels(
                tool_name=tool_name, action="warn"
            ).inc()
            logger.warning(
                "工具调用接近死循环阈值：%s（重复 %d 次），动作：WARN",
                tool_name,
                same_call_count + 1,
            )
        else:
            action = GuardrailAction.ALLOW

        return action

    def reset(self) -> None:
        """重置检测器状态

        清空滑动窗口中的所有调用记录，通常在新的会话或线程开始时调用。
        """
        self._call_history.clear()
        logger.info("ToolRepetitionDetector 状态已重置")
