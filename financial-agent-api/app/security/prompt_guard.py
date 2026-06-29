"""Prompt Injection 防御模块

实现用户输入的 Prompt 注入风险评估和输入清洗。
通过模式匹配检测常见注入手法，并根据风险等级进行定界符包裹处理。
"""

import logging
import re
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 注入风险等级
# ---------------------------------------------------------------------------


class InjectionRisk(Enum):
    """Prompt 注入风险等级枚举。

    Attributes:
        LOW: 低风险，未检测到注入模式。
        MEDIUM: 中风险，检测到部分可疑模式。
        HIGH: 高风险，检测到明确的注入攻击模式。
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# 注入模式定义
# ---------------------------------------------------------------------------

INJECTION_PATTERNS: list[dict[str, Any]] = [
    {
        "name": "system_prompt_override",
        "pattern": re.compile(
            r"(?i)(ignore\s+(previous|above|earlier|all)\s+instructions?)",
        ),
        "risk": InjectionRisk.HIGH,
        "description": "尝试覆盖系统提示词",
    },
    {
        "name": "role_switch",
        "pattern": re.compile(
            r"(?i)(you\s+are\s+now|act\s+as\s+(?:a\s+)?(?:DAN|jailbreak|evil|malicious))",
        ),
        "risk": InjectionRisk.HIGH,
        "description": "尝试切换 AI 角色",
    },
    {
        "name": "data_exfiltration",
        "pattern": re.compile(
            r"(?i)(reveal\s+(?:your|the|system)\s+(?:initial|original|secret|hidden)\s+prompt)",
        ),
        "risk": InjectionRisk.HIGH,
        "description": "尝试窃取系统提示词",
    },
    {
        "name": "instruction_injection",
        "pattern": re.compile(
            r"(?i)(new\s+instruction\s*:|system\s*:|assistant\s*:)\s",
        ),
        "risk": InjectionRisk.HIGH,
        "description": "通过伪指令头注入",
    },
    {
        "name": "output_manipulation",
        "pattern": re.compile(
            r"(?i)(output\s+(?:only\s+)?(?:the|your)\s+(?:following|below|next))",
        ),
        "risk": InjectionRisk.MEDIUM,
        "description": "尝试操控输出内容",
    },
    {
        "name": "context_escape",
        "pattern": re.compile(
            r"(?i)(forget\s+(?:everything|all|previous)|disregard\s+(?:all|previous|safety))",
        ),
        "risk": InjectionRisk.HIGH,
        "description": "尝试逃离上下文约束",
    },
    {
        "name": "encoding_bypass",
        "pattern": re.compile(
            r"(?:\\u[0-9a-fA-F]{4}|\\x[0-9a-fA-F]{2}|&#\d{1,5};|&#x[0-9a-fA-F]{1,4};)",
        ),
        "risk": InjectionRisk.MEDIUM,
        "description": "通过编码绕过过滤",
    },
    {
        "name": "chain_of_thought_abuse",
        "pattern": re.compile(
            r"(?i)(let\'s\s+think\s+step\s+by\s+step\s+(?:about\s+how\s+to\s+(?:hack|exploit|bypass)))",
        ),
        "risk": InjectionRisk.MEDIUM,
        "description": "滥用思维链进行注入",
    },
]

# ---------------------------------------------------------------------------
# 定界符常量
# ---------------------------------------------------------------------------

UNTRUSTED_START = "[UNTRUSTED_USER_INPUT_START]"
UNTRUSTED_END = "[UNTRUSTED_USER_INPUT_END]"

# ---------------------------------------------------------------------------
# 核心函数
# ---------------------------------------------------------------------------


def assess_injection_risk(user_input: str) -> tuple[InjectionRisk, list[dict[str, Any]]]:
    """评估用户输入的 Prompt 注入风险。

    逐一匹配所有注入模式，返回最高风险等级和匹配到的模式详情。

    Args:
        user_input: 用户输入的文本。

    Returns:
        二元组：(最高风险等级, 匹配到的注入模式列表)。
        每个模式详情包含 name、risk、description、matched_text 字段。
    """
    matched_patterns: list[dict[str, Any]] = []
    max_risk = InjectionRisk.LOW

    # 风险等级优先级映射
    risk_priority = {
        InjectionRisk.LOW: 0,
        InjectionRisk.MEDIUM: 1,
        InjectionRisk.HIGH: 2,
    }

    for pattern_def in INJECTION_PATTERNS:
        match = pattern_def["pattern"].search(user_input)
        if match:
            matched_patterns.append(
                {
                    "name": pattern_def["name"],
                    "risk": pattern_def["risk"],
                    "description": pattern_def["description"],
                    "matched_text": match.group(),
                }
            )
            if risk_priority[pattern_def["risk"]] > risk_priority[max_risk]:
                max_risk = pattern_def["risk"]

    if matched_patterns:
        logger.warning(
            "检测到 Prompt 注入风险: risk=%s, patterns=%s",
            max_risk.value,
            [p["name"] for p in matched_patterns],
        )

    return max_risk, matched_patterns


def sanitize_input(user_input: str) -> tuple[str, InjectionRisk, list[dict[str, Any]]]:
    """清洗用户输入，根据注入风险等级进行定界符包裹。

    - **HIGH** 风险：用 ``[UNTRUSTED_USER_INPUT_START/END]`` 定界符包裹，
      并在前面添加安全警告。
    - **MEDIUM** 风险：同样用定界符包裹，但不添加额外警告。
    - **LOW** 风险：原样返回，不做修改。

    Args:
        user_input: 用户输入的文本。

    Returns:
        三元组：(清洗后的文本, 风险等级, 匹配到的注入模式列表)。
    """
    risk_level, matched_patterns = assess_injection_risk(user_input)

    if risk_level == InjectionRisk.HIGH:
        sanitized = (
            f"{UNTRUSTED_START}\n"
            f"[SECURITY WARNING: 高风险输入已检测到注入模式，请谨慎处理]\n"
            f"{user_input}\n"
            f"{UNTRUSTED_END}"
        )
        logger.warning("高危输入已包裹定界符: patterns=%s", [p["name"] for p in matched_patterns])
        return sanitized, risk_level, matched_patterns

    if risk_level == InjectionRisk.MEDIUM:
        sanitized = f"{UNTRUSTED_START}\n{user_input}\n{UNTRUSTED_END}"
        logger.info("中危输入已包裹定界符: patterns=%s", [p["name"] for p in matched_patterns])
        return sanitized, risk_level, matched_patterns

    # LOW 风险，原样返回
    return user_input, risk_level, matched_patterns


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

__all__ = [
    "INJECTION_PATTERNS",
    "InjectionRisk",
    "UNTRUSTED_END",
    "UNTRUSTED_START",
    "assess_injection_risk",
    "sanitize_input",
]
