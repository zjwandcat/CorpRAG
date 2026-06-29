"""PII 检测与脱敏模块

实现个人身份信息（PII）的检测、脱敏和 LLM 输入/输出扫描。
支持邮箱、手机号、身份证号、银行卡号、护照号等中国常见 PII 类型。
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PII 正则模式定义
# ---------------------------------------------------------------------------

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    ),
    "phone_cn": re.compile(
        r"(?<!\d)1[3-9]\d{9}(?!\d)",
    ),
    "id_card": re.compile(
        r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)",
    ),
    "bank_card": re.compile(
        r"(?<!\d)[1-9]\d{14,18}(?!\d)",
    ),
    "passport": re.compile(
        r"(?<![A-Za-z0-9])[EeKkGgDdSsPpHh]\d{8}(?![A-Za-z0-9])",
    ),
}

# ---------------------------------------------------------------------------
# 核心函数
# ---------------------------------------------------------------------------


def detect_pii(text: str) -> dict[str, list[dict[str, Any]]]:
    """检测文本中的个人身份信息（PII）。

    扫描文本中所有已定义的 PII 类型，返回每种类型的匹配结果。

    Args:
        text: 待检测的文本内容。

    Returns:
        字典，键为 PII 类型名称，值为匹配列表。
        每个匹配项包含 ``value``（匹配文本）和 ``start``/``end``（位置索引）。
        仅包含有匹配结果的 PII 类型。
    """
    results: dict[str, list[dict[str, Any]]] = {}

    for pii_type, pattern in PII_PATTERNS.items():
        matches: list[dict[str, Any]] = []
        for match in pattern.finditer(text):
            matches.append(
                {
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                }
            )
        if matches:
            results[pii_type] = matches

    return results


def mask_pii(text: str) -> str:
    """将文本中的 PII 替换为脱敏标记。

    每种 PII 类型会被替换为 ``***{type}***`` 格式的占位符。

    Args:
        text: 待脱敏的文本内容。

    Returns:
        脱敏后的文本，PII 已被替换为类型标记。
    """
    masked_text = text

    for pii_type, pattern in PII_PATTERNS.items():
        masked_text = pattern.sub(f"***{pii_type}***", masked_text)

    return masked_text


def scan_llm_input(user_query: str) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    """扫描 LLM 输入中的 PII。

    检测用户输入中的 PII 并记录日志，但 **不进行脱敏处理**，
    以保留原始输入的语义完整性。

    Args:
        user_query: 用户输入的查询文本。

    Returns:
        二元组：(原始查询文本, PII 检测报告)。
        查询文本保持不变，PII 报告供审计使用。
    """
    pii_report = detect_pii(user_query)

    if pii_report:
        total_count = sum(len(v) for v in pii_report.values())
        pii_types = list(pii_report.keys())
        logger.warning(
            "LLM 输入检测到 PII: types=%s, total_count=%d（输入不脱敏）",
            pii_types,
            total_count,
        )

    return user_query, pii_report


def scan_llm_output(llm_response: str) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    """扫描 LLM 输出中的 PII。

    检测 LLM 响应中的 PII 并进行脱敏处理，防止敏感信息泄露。

    Args:
        llm_response: LLM 生成的响应文本。

    Returns:
        二元组：(脱敏后的响应文本, PII 检测报告)。
        响应文本中的 PII 已被替换为脱敏标记。
    """
    pii_report = detect_pii(llm_response)

    if pii_report:
        total_count = sum(len(v) for v in pii_report.values())
        pii_types = list(pii_report.keys())
        logger.warning(
            "LLM 输出检测到 PII: types=%s, total_count=%d（输出已脱敏）",
            pii_types,
            total_count,
        )
        # 输出侧进行脱敏
        masked_response = mask_pii(llm_response)
        return masked_response, pii_report

    return llm_response, pii_report


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

__all__ = [
    "PII_PATTERNS",
    "detect_pii",
    "mask_pii",
    "scan_llm_input",
    "scan_llm_output",
]
