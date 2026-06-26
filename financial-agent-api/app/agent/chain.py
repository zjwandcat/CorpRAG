import json
import logging
import random
import re
import threading
import time
from typing import Any, ClassVar
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from app.core.enums import ErrorCode
from app.exceptions import (
    LLMInvocationError,
    RateLimitExceededError,
    ToolExecutionError,
)
from app.models.schemas import ChatResponse, SourceReference, ToolCallStep

logger = logging.getLogger(__name__)

__all__ = ["AgentChain", "RateLimiter", "get_rate_limiter"]

_SYSTEM_PROMPT = (
    "你是一个企业内部办公效能助手，不仅支持检索内部文档和通讯录，还具备产品经理效能辅助能力。\n\n"
    "工具使用规则：\n"
    "1. 查询内部文档、规章制度、流程规范时，使用 search_internal_documents 工具\n"
    "2. 查询员工联系方式、职位、部门时，使用 get_employee_info 工具\n"
    "3. 需要查询互联网上的最新信息时，使用 search_web 工具\n"
    "4. 需要通知或提醒其他员工时，使用 send_email_notification 工具\n"
    "5. 当用户要求写需求文档、PRD 或功能说明时，必须调用 generate_prd_document 工具\n"
    "6. 当用户要求画流程图、业务流转图时，必须调用 generate_flowchart_code 工具\n"
    "7. 当用户要求画原型、设计页面布局时，必须调用 generate_html_prototype 工具\n\n"
    "重要原则：\n"
    "- 必须先调用对应工具获取信息，再基于工具返回结果给出自然语言回答\n"
    "- 不要在没有调用工具的情况下编造数据\n"
    "- 回答时注明信息来源（文档名或搜索结果）\n"
    "- 对于产品经理效能相关的请求，必须使用对应的生成工具，不要自行编写文档内容"
)


class RateLimiter:
    """请求频率与并发控制器

    特性：
    1. 固定窗口 RPM 限速（避免固定间隔，改用滑动窗口）
    2. 并发池控制（Semaphore，防止瞬时洪峰）
    3. 按 Provider 差异化限速参数
    """

    __slots__ = (
        "_lock",
        "_semaphore",
        "_timestamps",
        "max_concurrent",
        "max_rpm",
        "provider",
    )

    # 各 Provider 默认限速参数
    _PROVIDER_DEFAULTS: ClassVar[dict[str, dict[str, int]]] = {
        "zhipu": {"max_rpm": 30, "max_concurrent": 3},  # 智谱免费模型：30 RPM, 3 并发
        "xfyun": {"max_rpm": 20, "max_concurrent": 2},  # 讯飞星火：20 RPM, 2 并发
        "nim": {"max_rpm": 15, "max_concurrent": 5},  # NVIDIA NIM：15 RPM, 5 并发
    }

    def __init__(self, max_rpm: int = 15, max_concurrent: int = 5, provider: str = "nim") -> None:
        self.max_rpm = max_rpm
        self.max_concurrent = max_concurrent
        self.provider = provider
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(max_concurrent)
        self._timestamps: list[float] = []  # 滑动窗口：记录最近1分钟内的请求时间戳

    @classmethod
    def from_provider(cls, provider: str, max_rpm_override: int | None = None) -> RateLimiter:
        """根据 Provider 创建限速器"""
        defaults = cls._PROVIDER_DEFAULTS.get(provider, cls._PROVIDER_DEFAULTS["nim"])
        rpm = max_rpm_override or defaults["max_rpm"]
        concurrent = defaults["max_concurrent"]
        return cls(max_rpm=rpm, max_concurrent=concurrent, provider=provider)

    def acquire(self) -> None:
        """获取并发许可（进入并发池）"""
        self._semaphore.acquire()

    def release(self) -> None:
        """释放并发许可（离开并发池）"""
        self._semaphore.release()

    def wait(self) -> None:
        """滑动窗口限速：确保 RPM 不超限

        与旧版固定间隔不同，滑动窗口允许短时突发但严格限制1分钟内总请求数。
        """
        wait_sec = 0.0
        with self._lock:
            now = time.monotonic()
            # 清理1分钟前的时间戳
            cutoff = now - 60.0
            self._timestamps = [t for t in self._timestamps if t > cutoff]

            if len(self._timestamps) >= self.max_rpm:
                # 已达 RPM 上限，需等待最早的时间戳过期
                oldest = self._timestamps[0]
                wait_sec = oldest + 60.0 - now + 0.1  # 多等0.1秒避免边界问题

            self._timestamps.append(max(now, now + wait_sec))

        if wait_sec > 0:
            logger.debug(
                "限速等待 %.1f 秒（%s RPM=%d，当前窗口已 %d 请求）",
                wait_sec,
                self.provider,
                self.max_rpm,
                len(self._timestamps),
            )
            time.sleep(wait_sec)


_rate_limiter_instance: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    global _rate_limiter_instance
    if _rate_limiter_instance is None:
        from app.core.config import settings

        _rate_limiter_instance = RateLimiter.from_provider(
            provider=settings.PROVIDER,
            max_rpm_override=settings.RATE_LIMIT_RPM,
        )
        logger.info(
            "限速器初始化，Provider=%s, RPM=%d, 并发=%d",
            settings.PROVIDER,
            _rate_limiter_instance.max_rpm,
            _rate_limiter_instance.max_concurrent,
        )
    return _rate_limiter_instance


def _infer_result_type(tool_name: str) -> str:
    match tool_name:
        case "generate_flowchart_code":
            return "mermaid"
        case "generate_html_prototype":
            return "html"
        case "generate_prd_document":
            return "markdown"
        case "search_internal_documents" | "search_web":
            return "search_results"
        case _:
            return "text"


def _extract_sources(result: str, tool_name: str) -> list[SourceReference]:
    if tool_name not in ("search_internal_documents", "search_web"):
        return []

    # 尝试解析 RAG_METADATA 注释块（仅 search_internal_documents）
    rag_metadata: dict[str, list[dict[str, Any]]] | None = None
    if tool_name == "search_internal_documents":
        metadata_match = re.search(r"<!--RAG_METADATA:(.+?)-->", result, re.DOTALL)
        if metadata_match:
            try:
                rag_metadata = json.loads(metadata_match.group(1))
            except json.JSONDecodeError, ValueError:
                logger.warning("RAG_METADATA 解析失败，使用默认提取")

    sources: list[SourceReference] = []
    pattern = r"【来源：(.+?)】\n(.+?)(?=【来源：|$)"
    matches = re.findall(pattern, result, re.DOTALL)

    # 构建 source -> metadata 的映射
    metadata_map: dict[str, dict[str, Any]] = {}
    if rag_metadata and "sources" in rag_metadata:
        for meta in rag_metadata["sources"]:
            source_name = meta.get("source", "")
            metadata_map[source_name] = meta

    for source_name, snippet in matches:
        snippet_clean = snippet.strip()[:200]
        # 从 RAG_METADATA 中提取精确的 score 和 rerank_score
        meta = metadata_map.get(source_name, {})
        score = float(meta.get("score", 0.0))
        rerank_score = meta.get("rerank_score")
        if rerank_score is not None:
            rerank_score = float(rerank_score)

        sources.append(
            SourceReference(
                source=source_name,
                department="通用",
                score=score,
                snippet=snippet_clean,
                rerank_score=rerank_score,
            )
        )

    return sources


class AgentChain:
    __slots__ = ("llm_with_tools", "sessions", "tools_by_name")

    def __init__(self, llm_with_tools: BaseChatModel, tools_by_name: dict[str, BaseTool]) -> None:
        self.llm_with_tools = llm_with_tools
        self.tools_by_name = tools_by_name
        self.sessions: dict[str, list[Any]] = {}
        logger.info("AgentChain 初始化完成，可用工具：%s", list(tools_by_name.keys()))

    def _invoke_with_retry(self, messages: list[Any], max_retries: int | None = None) -> AIMessage:
        if max_retries is None:
            from app.core.config import settings

            max_retries = settings.RETRY_MAX_ATTEMPTS

        limiter = get_rate_limiter()
        limiter.wait()
        limiter.acquire()

        try:
            wait = 0.0
            for attempt in range(max_retries):
                try:
                    return self.llm_with_tools.invoke(messages)
                except Exception as exc:
                    err_str = str(exc)
                    if ErrorCode.RATE_LIMIT in err_str and attempt < max_retries - 1:
                        # 指数退避 + 随机抖动，避免多实例同步重试（雷群效应）
                        base_wait = min(10 * (2**attempt), 120)  # 10s, 20s, 40s, 80s, 120s
                        jitter = random.uniform(
                            1, base_wait * 0.5
                        )  # 1~5s, 1~10s, 1~20s, 1~40s, 1~60s
                        wait = base_wait + jitter
                        logger.warning("429 限流，%.1f 秒后重试（第 %d 次）", wait, attempt + 1)
                        time.sleep(wait)
                        limiter.wait()
                    elif ErrorCode.RATE_LIMIT in err_str:
                        raise RateLimitExceededError(
                            message="LLM 调用频率超限，重试次数已耗尽",
                            retry_after=int(wait) if wait > 0 else None,
                            details=str(exc),
                        ) from exc
                    else:
                        raise LLMInvocationError(message="LLM 调用失败", details=str(exc)) from exc
        finally:
            limiter.release()

        raise LLMInvocationError(message="LLM 调用失败：未知错误")

    def _execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        messages: list[Any],
    ) -> tuple[list[str], list[ToolCallStep]]:
        tools_used: list[str] = []
        intermediate_steps: list[ToolCallStep] = []

        for tool_call in tool_calls:
            match tool_call:
                case {
                    "name": str(tool_name),
                    "args": dict(tool_args),
                    "id": str(tool_call_id),
                }:
                    logger.info("工具调用：%s", tool_name)
                    logger.debug("工具参数：%s", json.dumps(tool_args, ensure_ascii=False))

                    step_start = time.monotonic()
                    result_type = _infer_result_type(tool_name)

                    tool_func = self.tools_by_name.get(tool_name)
                    if tool_func is None:
                        result = f"错误：找不到名为 {tool_name} 的工具。"
                        logger.error("工具 %s 不存在", tool_name)
                        step_status = "error"
                    else:
                        try:
                            result = tool_func.invoke(tool_args)
                            logger.info("工具执行成功")
                            step_status = "success"
                        except Exception as exc:
                            result = f"工具执行失败：{exc!s}"
                            logger.error("工具执行失败：%s", exc)
                            step_status = "error"

                    duration_ms = int((time.monotonic() - step_start) * 1000)
                    tools_used.append(tool_name)
                    sources = _extract_sources(result, tool_name)

                    step = ToolCallStep(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        tool_result=str(result),
                        tool_result_type=result_type,
                        sources=sources,
                        duration_ms=duration_ms,
                        status=step_status,
                    )
                    intermediate_steps.append(step)
                    messages.append(ToolMessage(content=str(result), tool_call_id=tool_call_id))
                case _:
                    logger.warning("未知的工具调用格式：%s", tool_call)
                    raise ToolExecutionError(message="工具调用格式无效", details=str(tool_call))

        return tools_used, intermediate_steps

    def run(self, query: str, session_id: str | None = None) -> ChatResponse:
        from app.core.config import settings

        total_start = time.monotonic()
        system_message = SystemMessage(content=_SYSTEM_PROMPT)

        if session_id is None or session_id not in self.sessions:
            if session_id is None:
                session_id = str(uuid4())
            logger.info("创建新会话：%s", session_id)
            self.sessions[session_id] = [system_message]
        else:
            logger.info("使用已有会话：%s", session_id)

        messages = self.sessions[session_id]
        messages.append(HumanMessage(content=query))
        logger.info("用户输入：%s", query)

        all_tools_used: list[str] = []
        all_intermediate_steps: list[ToolCallStep] = []

        for _round in range(settings.MAX_TOOL_ROUNDS):
            logger.info("调用 LLM（第 %d 轮）", _round + 1)
            ai_message: AIMessage = self._invoke_with_retry(messages)
            messages.append(ai_message)

            if not ai_message.tool_calls:
                logger.info("LLM 未调用工具，直接返回回答")
                break

            logger.info("LLM 请求调用 %d 个工具", len(ai_message.tool_calls))
            tools_used, steps = self._execute_tool_calls(ai_message.tool_calls, messages)
            all_tools_used.extend(tools_used)
            all_intermediate_steps.extend(steps)
        else:
            logger.warning("达到最大工具调用轮数 %d，强制生成最终回答", settings.MAX_TOOL_ROUNDS)
            ai_message = self._invoke_with_retry(messages)
            messages.append(ai_message)

        content = ai_message.content
        answer = str(content) if content else ""
        logger.info("最终回答生成完成，长度：%d 字符", len(answer))

        if len(messages) > settings.SESSION_MAX_MESSAGES:
            messages[:] = [system_message] + messages[-settings.SESSION_MAX_MESSAGES :]

        total_duration_ms = int((time.monotonic() - total_start) * 1000)

        return ChatResponse(
            answer=answer,
            answer_format="markdown",
            tools_used=all_tools_used,
            intermediate_steps=all_intermediate_steps,
            total_duration_ms=total_duration_ms,
            session_id=session_id,
        )

    def clear_session(self, session_id: str) -> None:
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info("已清除会话：%s", session_id)
        else:
            logger.warning("会话 %s 不存在", session_id)
