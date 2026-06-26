"""LangGraph 状态图核心模块

将线性 AgentChain 执行逻辑重构为基于 LangGraph 的状态机拓扑，
支持多轮对话记忆（MemorySaver）、条件路由和 SSE 流式状态推送。
"""

import json
import logging
import random
import time
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, TypedDict

from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from app.agent.chain import _extract_sources, _infer_result_type, get_rate_limiter
from app.core.enums import ErrorCode, SSEEventType
from app.exceptions import LLMInvocationError, RateLimitExceededError
from app.models.schemas import ChatResponse, SourceReference, ToolCallStep

logger = logging.getLogger(__name__)

__all__ = ["AgentGraph", "AgentState", "build_agent_graph"]

# 复用 AgentChain 的系统提示词
_SYSTEM_PROMPT = (
    "你是一个企业内部办公效能助手，不仅支持检索内部文档和通讯录，还具备产品经理效能辅助能力。\n\n"
    "工具选择决策规则（按优先级排序）：\n"
    "1. 查询员工联系方式、职位、部门 → 使用 get_employee_info 工具\n"
    "2. 查询公司内部文档、规章制度、流程规范 → 使用 search_internal_documents 工具\n"
    "3. 查询互联网上的最新信息、行业动态、外部知识 → 使用 search_web 工具\n"
    "4. 需要通知或提醒其他员工 → 使用 send_email_notification 工具\n"
    "5. 写需求文档、PRD 或功能说明 → 必须调用 generate_prd_document 工具\n"
    "6. 画流程图、业务流转图 → 必须调用 generate_flowchart_code 工具\n"
    "7. 画原型、设计页面布局 → 必须调用 generate_html_prototype 工具\n\n"
    "重要：联网搜索 vs 内部文档的判断标准：\n"
    "- 如果用户问题明确涉及公司内部制度、流程、规范 → search_internal_documents\n"
    "- 如果用户问题涉及外部信息"
    "（如行业趋势、市场数据、新闻资讯、技术博客、公开知识）→ search_web\n"
    "- 如果 search_internal_documents 返回的结果"
    "与用户问题不相关 → 应该再尝试 search_web\n"
    "- 如果不确定信息来源 → 先尝试 search_internal_documents，"
    "结果不相关时再用 search_web\n\n"
    "重要原则：\n"
    "- 必须先调用对应工具获取信息，再基于工具返回结果给出自然语言回答\n"
    "- 不要在没有调用工具的情况下编造数据\n"
    "- 回答时注明信息来源（文档名或搜索结果）\n"
    "- 对于产品经理效能相关的请求，必须使用对应的生成工具，不要自行编写文档内容"
)

_FORCE_END_PROMPT = (
    "\n\n[系统提示] 已达到最大推理轮数，请基于已有信息生成最终回答，不要再调用任何工具。"
)


# =====================================================================
# AgentState TypedDict
# =====================================================================


class AgentState(TypedDict):
    """LangGraph 状态图全局状态对象"""

    messages: Annotated[list[BaseMessage], add_messages]
    """对话消息列表，使用 add_messages reducer 自动追加"""

    retrieval_context: list[SourceReference]
    """检索上下文对象列表，默认空列表"""

    tool_outputs: list[ToolCallStep]
    """工具输出汇总列表，默认空列表"""

    current_status: Literal[
        "idle",
        "agent_processing",
        "tool_executing",
        "retrieving",
        "completed",
        "error",
    ]
    """当前执行状态标识，默认 "idle" """

    iteration_count: int
    """Agent 循环计数器，默认 0，每次 agent_node 执行递增"""


# =====================================================================
# 节点函数
# =====================================================================


def agent_node(
    state: AgentState,
    llm_with_tools: BaseChatModel,
) -> dict:
    """LLM 决策节点：调用 ChatNVIDIA，返回 AIMessage 更新

    1. 检查 iteration_count，若 >= MAX_TOOL_ROUNDS 则追加强制结束提示
    2. 调用 llm_with_tools.invoke(state["messages"])
    3. 更新 current_status = "agent_processing"
    4. 递增 iteration_count
    5. 返回状态更新片段
    """
    from app.core.config import settings

    messages = state["messages"]
    iteration_count = state.get("iteration_count", 0)

    # 循环溢出保护：追加强制结束提示
    if iteration_count >= settings.MAX_TOOL_ROUNDS:
        logger.warning(
            "循环次数 %d 已达上限 %d，追加强制结束提示",
            iteration_count,
            settings.MAX_TOOL_ROUNDS,
        )
        # 在消息列表末尾追加一条系统提示
        messages = list(messages) + [SystemMessage(content=_FORCE_END_PROMPT)]

    # 调用 LLM（含重试逻辑）
    ai_message = _invoke_with_retry(llm_with_tools, messages)

    new_iteration = iteration_count + 1
    logger.info(
        "agent_node 完成，iteration=%d，tool_calls=%s",
        new_iteration,
        [tc.get("name", "") for tc in (ai_message.tool_calls or [])],
    )

    return {
        "messages": [ai_message],
        "current_status": "agent_processing",
        "iteration_count": new_iteration,
    }


def _invoke_with_retry(
    llm_with_tools: BaseChatModel,
    messages: list[Any],
    max_retries: int | None = None,
) -> AIMessage:
    """调用 LLM 并在 429 限流时指数退避重试

    使用并发池控制 + 滑动窗口限速，避免洪峰式请求。
    """
    from app.core.config import settings

    if max_retries is None:
        max_retries = settings.RETRY_MAX_ATTEMPTS

    limiter = get_rate_limiter()
    limiter.wait()
    limiter.acquire()

    try:
        wait = 0.0
        for attempt in range(max_retries):
            try:
                return llm_with_tools.invoke(messages)
            except Exception as exc:
                err_str = str(exc)
                if ErrorCode.RATE_LIMIT in err_str and attempt < max_retries - 1:
                    # 指数退避 + 随机抖动
                    base_wait = min(10 * (2**attempt), 120)
                    jitter = random.uniform(1, base_wait * 0.5)
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


def tool_node(
    state: AgentState,
    tools_by_name: dict[str, BaseTool],
) -> dict:
    """通用工具执行节点：执行非检索类 tool_calls

    1. 从最后一条 AIMessage 提取非检索类 tool_calls
    2. 顺序执行每个工具调用
    3. 每个工具执行结果封装为 ToolMessage 追加至 messages
    4. 同时构建 ToolCallStep 追加至 tool_outputs
    5. 更新 current_status = "tool_executing"
    """
    messages = state["messages"]
    last_message = messages[-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"current_status": "tool_executing"}

    # 分离非检索类 tool_calls
    non_retrieval_calls = [
        tc for tc in last_message.tool_calls if tc.get("name") != "search_internal_documents"
    ]

    if not non_retrieval_calls:
        return {"current_status": "tool_executing"}

    new_messages: list[ToolMessage] = []
    new_tool_outputs: list[ToolCallStep] = []

    for tool_call in non_retrieval_calls:
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id", "")

        logger.info("工具调用：%s", tool_name)
        logger.debug("工具参数：%s", json.dumps(tool_args, ensure_ascii=False))

        step_start = time.monotonic()
        result_type = _infer_result_type(tool_name)

        tool_func = tools_by_name.get(tool_name)
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
        new_tool_outputs.append(step)
        new_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call_id))

    return {
        "messages": new_messages,
        "tool_outputs": new_tool_outputs,
        "current_status": "tool_executing",
    }


def retrieval_node(
    state: AgentState,
    vectorstore: Chroma,
    reranker: Any,
    tools_by_name: dict[str, BaseTool],
) -> dict:
    """知识库检索节点：执行 hybrid_search + Reranker 精排

    1. 从最后一条 AIMessage 提取 search_internal_documents 的 tool_call
    2. 调用 hybrid_search 执行混合检索
    3. 调用 Reranker 精排（如可用）
    4. 格式化检索结果 + RAG_METADATA
    5. 将检索结果封装为 ToolMessage 追加至 messages
    6. 更新 retrieval_context 字段
    7. 处理同一 AIMessage 中的非检索类 tool_calls（如有）
    8. 更新 current_status = "retrieving"
    """
    from app.rag.vectorstore import hybrid_search

    messages = state["messages"]
    last_message = messages[-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"current_status": "retrieving"}

    new_messages: list[ToolMessage] = []
    new_tool_outputs: list[ToolCallStep] = []
    all_sources: list[SourceReference] = []

    # 获取所有文档内容（用于 BM25 检索）
    all_docs_data = vectorstore.get()
    all_documents: list[str] = [
        doc for doc in (all_docs_data.get("documents") or []) if doc is not None
    ]

    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id", "")

        if tool_name == "search_internal_documents":
            # 执行检索
            query = tool_args.get("query", "")
            department = tool_args.get("department", "通用")

            logger.info("检索工具调用：query=%s, department=%s", query, department)
            step_start = time.monotonic()

            try:
                # Step 1: hybrid_search（向量 + BM25 + RRF 融合）
                docs = hybrid_search(
                    query=query,
                    department=department,
                    vectorstore=vectorstore,
                    all_documents=all_documents,
                    top_k=3,
                )

                # Step 2: Reranker 精排
                if docs and reranker is not None:
                    try:
                        rerank_results = reranker.rerank(query=query, documents=docs, top_n=3)
                        if rerank_results:
                            docs = [r.document for r in rerank_results]
                            for r in rerank_results:
                                r.document.metadata["rerank_score"] = r.relevance_score
                            logger.info("Reranker 精排完成，top %d 结果", len(rerank_results))
                        else:
                            logger.warning("Reranker 降级，使用 RRF 原始排序")
                    except Exception as exc:
                        logger.warning("Reranker 调用失败，使用原始排序：%s", exc)

                # 格式化检索结果
                if not docs:
                    result = f"未找到关于「{query}」的内部文档（部门：{department}）。"
                    logger.info("检索无结果：%s", query)
                else:
                    formatted_parts: list[str] = []
                    rerank_metadata: list[dict[str, Any]] = []

                    for d in docs:
                        source = d.metadata.get("source", "未知来源")
                        formatted_parts.append(f"【来源：{source}】\n{d.page_content}")

                        rerank_score = d.metadata.get("rerank_score")
                        rerank_metadata.append(
                            {
                                "source": source,
                                "score": float(rerank_score) if rerank_score is not None else 0.0,
                                "rerank_score": float(rerank_score)
                                if rerank_score is not None
                                else None,
                            }
                        )

                    result = "\n\n".join(formatted_parts)

                    # 构建 RAG_METADATA 注释块
                    if rerank_metadata:
                        metadata_json = json.dumps({"sources": rerank_metadata}, ensure_ascii=False)
                        result += f"\n\n<!--RAG_METADATA:{metadata_json}-->"

                    logger.info("检索到 %d 个相关文本块（混合检索 + Reranker）", len(docs))

                duration_ms = int((time.monotonic() - step_start) * 1000)
                sources = _extract_sources(result, tool_name)
                all_sources.extend(sources)

                step = ToolCallStep(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=result,
                    tool_result_type="search_results",
                    sources=sources,
                    duration_ms=duration_ms,
                    status="success",
                )
                new_tool_outputs.append(step)
                new_messages.append(ToolMessage(content=result, tool_call_id=tool_call_id))

            except Exception as exc:
                error_result = f"检索失败：{exc!s}"
                logger.error("检索执行失败：%s", exc)
                duration_ms = int((time.monotonic() - step_start) * 1000)
                step = ToolCallStep(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=error_result,
                    tool_result_type="text",
                    sources=[],
                    duration_ms=duration_ms,
                    status="error",
                )
                new_tool_outputs.append(step)
                new_messages.append(ToolMessage(content=error_result, tool_call_id=tool_call_id))

        else:
            # 非检索类工具调用（当 AIMessage 同时包含检索和非检索调用时）
            logger.info("retrieval_node 中处理非检索工具：%s", tool_name)
            step_start = time.monotonic()
            result_type = _infer_result_type(tool_name)

            # 通过函数参数 tools_by_name 查找工具（闭包注入，线程安全）
            tool_func = tools_by_name.get(tool_name)
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
            new_tool_outputs.append(step)
            new_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call_id))

    return {
        "messages": new_messages,
        "retrieval_context": all_sources,
        "tool_outputs": new_tool_outputs,
        "current_status": "retrieving",
    }


# =====================================================================
# 条件路由
# =====================================================================


def should_continue(state: AgentState) -> str:
    """根据 agent_node 输出决定路由方向

    1. 最后一条消息无 tool_calls → "end"
    2. tool_calls 包含 search_internal_documents → "retrieval"
    3. tool_calls 仅包含非检索工具 → "tools"
    """
    messages = state["messages"]
    last_message = messages[-1]

    # 无 tool_calls → 结束
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return "end"

    # 有 tool_calls → 判断是否包含检索工具
    tool_names = [tc.get("name", "") for tc in last_message.tool_calls]

    if "search_internal_documents" in tool_names:
        # 检索工具路由到 retrieval_node
        # 非检索工具也由 retrieval_node 内部分离处理
        return "retrieval"

    return "tools"


# =====================================================================
# 图构建
# =====================================================================


def build_agent_graph(
    llm_with_tools: BaseChatModel,
    tools_by_name: dict[str, BaseTool],
    vectorstore: Chroma,
    reranker: Any,
    memory_saver: MemorySaver | None = None,
) -> Any:
    """构建并编译 LangGraph 状态图

    Args:
        llm_with_tools: 绑定了工具的 LLM 实例
        tools_by_name: 工具名称到工具实例的映射
        vectorstore: ChromaDB 向量库实例
        reranker: Reranker 实例
        memory_saver: 可选的 MemorySaver 实例，未传入时自动创建

    Returns:
        编译后的 CompiledStateGraph
    """
    # 创建状态图
    graph = StateGraph(AgentState)

    # 添加节点（使用 lambda 闭包注入依赖）
    graph.add_node(
        "agent",
        lambda state: agent_node(state, llm_with_tools),
    )
    graph.add_node(
        "tools",
        lambda state: tool_node(state, tools_by_name),
    )
    graph.add_node(
        "retrieval",
        lambda state: retrieval_node(state, vectorstore, reranker, tools_by_name),
    )

    # 设置入口
    graph.set_entry_point("agent")

    # 添加条件边：agent → (tools | retrieval | END)
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "retrieval": "retrieval",
            "end": END,
        },
    )

    # 添加回环边：tools → agent, retrieval → agent
    graph.add_edge("tools", "agent")
    graph.add_edge("retrieval", "agent")

    # 编译图（注入 MemorySaver 检查点）
    if memory_saver is None:
        memory_saver = MemorySaver()
    compiled = graph.compile(checkpointer=memory_saver)

    logger.info("LangGraph 状态图构建完成，节点：agent, tools, retrieval")
    return compiled


# =====================================================================
# AgentGraph 执行器类
# =====================================================================


class AgentGraph:
    """LangGraph 状态图执行器（替代 AgentChain）

    提供同步执行、异步流式执行和会话清除三种核心能力。
    """

    def __init__(self, compiled_graph: Any, memory_saver: MemorySaver) -> None:
        self._compiled_graph = compiled_graph
        self._memory_saver = memory_saver
        logger.info("AgentGraph 初始化完成")

    def run(self, query: str, session_id: str | None = None) -> ChatResponse:
        """同步执行状态图，返回 ChatResponse

        Args:
            query: 用户问题
            session_id: 会话 ID（同时作为 thread_id）

        Returns:
            ChatResponse 包含回答、工具调用步骤、耗时等
        """
        from uuid import uuid4

        total_start = time.monotonic()

        if session_id is None:
            session_id = str(uuid4())

        config = {"configurable": {"thread_id": session_id}}

        # 构造输入：LangGraph + MemorySaver 会自动加载历史状态，
        # 因此只需传入新增的消息，避免 add_messages reducer 重复追加。
        # 新会话需包含 SystemMessage，已有会话只传 HumanMessage。
        # 注意：非 reducer 字段（如 iteration_count）传入后会覆盖检查点值，
        # 因此已有会话需从检查点读取当前值。
        has_existing_state = False
        saved_iteration_count = 0
        try:
            existing_state = self._compiled_graph.get_state(config)
            if existing_state and existing_state.values:
                existing_messages = existing_state.values.get("messages", [])
                if existing_messages:
                    has_existing_state = True
                    saved_iteration_count = existing_state.values.get("iteration_count", 0)
                    logger.info(
                        "从 MemorySaver 恢复会话：%s，历史消息数：%d",
                        session_id,
                        len(existing_messages),
                    )
        except Exception as exc:
            logger.warning("MemorySaver 读取失败，降级为无记忆模式：%s", exc)

        if has_existing_state:
            # 已有会话：只传入新的 HumanMessage，历史由检查点自动加载
            input_state: AgentState = {
                "messages": [HumanMessage(content=query)],
                "retrieval_context": [],
                "tool_outputs": [],
                "current_status": "idle",
                "iteration_count": saved_iteration_count,
            }
        else:
            # 新会话：需包含 SystemMessage + HumanMessage
            logger.info("创建新会话：%s", session_id)
            input_state = {
                "messages": [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=query),
                ],
                "retrieval_context": [],
                "tool_outputs": [],
                "current_status": "idle",
                "iteration_count": 0,
            }

        # 执行状态图
        try:
            result = self._compiled_graph.invoke(input_state, config)
        except Exception as exc:
            logger.error("状态图执行失败：%s", exc)
            total_duration_ms = int((time.monotonic() - total_start) * 1000)
            return ChatResponse(
                answer=f"对话处理失败：{exc!s}",
                answer_format="text",
                tools_used=[],
                intermediate_steps=[],
                total_duration_ms=total_duration_ms,
                session_id=session_id,
            )

        # 从最终状态组装 ChatResponse
        final_messages = result.get("messages", [])
        tool_outputs = result.get("tool_outputs", [])
        _ = result.get("retrieval_context", [])

        # 提取最终回答（最后一条 AIMessage 的 content）
        answer = ""
        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage) and msg.content:
                answer = str(msg.content)
                break

        # 提取使用的工具列表
        tools_used = [step.tool_name for step in tool_outputs]

        # 消息窗口截断
        self._truncate_messages(final_messages, session_id, config)

        total_duration_ms = int((time.monotonic() - total_start) * 1000)
        logger.info("状态图执行完成，回答长度：%d，耗时：%dms", len(answer), total_duration_ms)

        return ChatResponse(
            answer=answer,
            answer_format="markdown",
            tools_used=tools_used,
            intermediate_steps=tool_outputs,
            total_duration_ms=total_duration_ms,
            session_id=session_id,
        )

    async def run_stream(self, query: str, thread_id: str) -> AsyncGenerator[dict]:
        """异步流式执行，返回 SSE 事件生成器

        通过 compiled_graph.astream_events() 监听节点执行事件，
        转换为 SSE 事件格式推送给前端。

        Args:
            query: 用户问题
            thread_id: 会话线程 ID

        Yields:
            SSE 事件字典，包含 event, data, id 字段
        """
        from app.core.config import settings

        total_start = time.monotonic()
        event_seq = 0

        def _make_event(event_type: SSEEventType, data: dict) -> dict:
            """构建 SSE 事件"""
            nonlocal event_seq
            event_seq += 1
            return {
                "event": event_type.value,
                "data": {
                    **data,
                    "thread_id": thread_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
                "id": f"{thread_id}-{event_seq}",
            }

        # 推送 stream_start 事件
        yield _make_event(SSEEventType.STREAM_START, {})

        config = {"configurable": {"thread_id": thread_id}}

        # 构造输入：LangGraph + MemorySaver 会自动加载历史状态，
        # 因此只需传入新增的消息，避免 add_messages reducer 重复追加。
        # 非 reducer 字段（如 iteration_count）需从检查点读取当前值。
        has_existing_state = False
        saved_iteration_count = 0
        try:
            existing_state = self._compiled_graph.get_state(config)
            if existing_state and existing_state.values:
                existing_messages = existing_state.values.get("messages", [])
                if existing_messages:
                    has_existing_state = True
                    saved_iteration_count = existing_state.values.get("iteration_count", 0)
                    logger.info("流式执行：从 MemorySaver 恢复会话：%s", thread_id)
        except Exception as exc:
            logger.warning("MemorySaver 读取失败，降级为无记忆模式：%s", exc)
            yield _make_event(SSEEventType.MEMORY_LOAD_FAILED, {})

        if has_existing_state:
            # 已有会话：只传入新的 HumanMessage，历史由检查点自动加载
            input_state: AgentState = {
                "messages": [HumanMessage(content=query)],
                "retrieval_context": [],
                "tool_outputs": [],
                "current_status": "idle",
                "iteration_count": saved_iteration_count,
            }
        else:
            # 新会话：需包含 SystemMessage + HumanMessage
            logger.info("流式执行：创建新会话：%s", thread_id)
            input_state = {
                "messages": [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=query),
                ],
                "retrieval_context": [],
                "tool_outputs": [],
                "current_status": "idle",
                "iteration_count": 0,
            }

        # 收集执行结果
        all_tool_outputs: list[ToolCallStep] = []
        all_retrieval_context: list[SourceReference] = []
        final_answer = ""
        final_messages: list[BaseMessage] = []

        try:
            # 使用 astream_events 监听节点执行
            async for event in self._compiled_graph.astream_events(
                input_state,
                config,
                version="v2",
            ):
                kind = event.get("event", "")

                # 节点开始
                if kind == "on_chain_start":
                    _ = event.get("tags", [])
                    node_name = event.get("name", "")
                    if node_name == "agent":
                        # iteration_count 从 agent_node 输出中获取更准确，
                        # 此处先用 0 占位，实际值在 on_chain_end 中更新
                        yield _make_event(SSEEventType.AGENT_START, {"iteration": 0})

                # 节点结束
                elif kind == "on_chain_end":
                    node_name = event.get("name", "")
                    output = event.get("data", {}).get("output", {})

                    if node_name == "agent":
                        # agent_node 完成
                        messages_from_node = output.get("messages", [])
                        iteration_count = output.get("iteration_count", 0)

                        # 检查是否有 tool_calls
                        has_tool_calls = False
                        tool_calls_list = None
                        for msg in messages_from_node:
                            # 兼容 AIMessage 对象和序列化后的字典
                            if isinstance(msg, AIMessage):
                                if msg.tool_calls:
                                    has_tool_calls = True
                                    tool_calls_list = [
                                        {
                                            "name": tc.get("name", ""),
                                            "args": tc.get("args", {}),
                                        }
                                        for tc in msg.tool_calls
                                    ]
                            elif isinstance(msg, dict) and msg.get("tool_calls"):
                                has_tool_calls = True
                                tool_calls_list = [
                                    {
                                        "name": tc.get("name", ""),
                                        "args": tc.get("args", {}),
                                    }
                                    for tc in msg["tool_calls"]
                                ]

                        # 检查是否达到最大迭代次数
                        if iteration_count >= settings.MAX_TOOL_ROUNDS:
                            yield _make_event(
                                SSEEventType.AGENT_FORCE_END,
                                {
                                    "iteration": iteration_count,
                                    "max_iterations": settings.MAX_TOOL_ROUNDS,
                                },
                            )

                        yield _make_event(
                            SSEEventType.AGENT_END,
                            {
                                "iteration": iteration_count,
                                "tool_calls": tool_calls_list,
                                "has_final_answer": not has_tool_calls,
                            },
                        )

                    elif node_name == "tools":
                        # tool_node 完成
                        node_tool_outputs = output.get("tool_outputs", [])
                        all_tool_outputs.extend(node_tool_outputs)

                        for step in node_tool_outputs:
                            yield _make_event(
                                SSEEventType.TOOL_RESULT,
                                {
                                    "tool_name": step.tool_name,
                                    "result_summary": step.tool_result[:200]
                                    if step.tool_result
                                    else "",
                                    "duration_ms": step.duration_ms,
                                    "status": step.status,
                                },
                            )

                    elif node_name == "retrieval":
                        # retrieval_node 完成
                        node_tool_outputs = output.get("tool_outputs", [])
                        node_retrieval_context = output.get("retrieval_context", [])
                        all_tool_outputs.extend(node_tool_outputs)
                        all_retrieval_context.extend(node_retrieval_context)

                        # 推送检索结果事件
                        if node_retrieval_context:
                            yield _make_event(
                                SSEEventType.RETRIEVAL_RESULT,
                                {
                                    "sources": [s.model_dump() for s in node_retrieval_context],
                                    "total_count": len(node_retrieval_context),
                                },
                            )
                        else:
                            # 检索无结果
                            for step in node_tool_outputs:
                                if step.tool_name == "search_internal_documents":
                                    yield _make_event(
                                        SSEEventType.RETRIEVAL_EMPTY,
                                        {
                                            "query": step.tool_args.get("query", ""),
                                        },
                                    )
                                    break

                        # 推送工具结果事件
                        for step in node_tool_outputs:
                            yield _make_event(
                                SSEEventType.TOOL_RESULT,
                                {
                                    "tool_name": step.tool_name,
                                    "result_summary": step.tool_result[:200]
                                    if step.tool_result
                                    else "",
                                    "duration_ms": step.duration_ms,
                                    "status": step.status,
                                },
                            )

                # 工具调用开始（从 LLM 输出中检测）
                elif kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                        for tc_chunk in chunk.tool_call_chunks:
                            # 兼容 ToolCallChunk 对象和字典
                            tc_name = (
                                tc_chunk.get("name")
                                if isinstance(tc_chunk, dict)
                                else getattr(tc_chunk, "name", None)
                            )
                            tc_args = (
                                tc_chunk.get("args")
                                if isinstance(tc_chunk, dict)
                                else getattr(tc_chunk, "args", {})
                            )
                            if tc_name:
                                yield _make_event(
                                    SSEEventType.TOOL_CALL,
                                    {
                                        "tool_name": tc_name,
                                        "tool_args": tc_args or {},
                                    },
                                )

        except Exception as exc:
            logger.error("流式执行失败：%s", exc)
            yield _make_event(
                SSEEventType.STREAM_ERROR,
                {
                    "error_type": type(exc).__name__,
                    "error_message": "对话处理失败，请稍后重试",
                },
            )
            return

        # 从最终状态提取回答
        try:
            final_state = self._compiled_graph.get_state(config)
            if final_state and final_state.values:
                final_messages = final_state.values.get("messages", [])
                all_tool_outputs = final_state.values.get("tool_outputs", all_tool_outputs)
                all_retrieval_context = final_state.values.get(
                    "retrieval_context", all_retrieval_context
                )

                for msg in reversed(final_messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        final_answer = str(msg.content)
                        break
        except Exception as exc:
            logger.warning("获取最终状态失败：%s", exc)

        # 消息窗口截断
        self._truncate_messages(final_messages, thread_id, config)

        # 推送 stream_end 事件
        total_duration_ms = int((time.monotonic() - total_start) * 1000)
        tools_used = [step.tool_name for step in all_tool_outputs]

        chat_response = ChatResponse(
            answer=final_answer,
            answer_format="markdown",
            tools_used=tools_used,
            intermediate_steps=all_tool_outputs,
            total_duration_ms=total_duration_ms,
            session_id=thread_id,
        )

        yield _make_event(
            SSEEventType.STREAM_END,
            {
                "chat_response": chat_response.model_dump(),
            },
        )

    def clear_session(self, session_id: str) -> None:
        """清除 MemorySaver 中对应 thread_id 的检查点

        MemorySaver（内存版）暂不支持直接删除检查点。
        采用"空状态覆盖"策略：对指定 thread_id 写入仅含 SystemMessage 的初始状态，
        等效于清除历史。

        Args:
            session_id: 要清除的会话 ID
        """
        config = {"configurable": {"thread_id": session_id}}

        try:
            # 使用空状态覆盖，等效于清除
            empty_state: AgentState = {
                "messages": [SystemMessage(content=_SYSTEM_PROMPT)],
                "retrieval_context": [],
                "tool_outputs": [],
                "current_status": "idle",
                "iteration_count": 0,
            }
            # 通过 update_state 覆盖当前状态
            self._compiled_graph.update_state(config, empty_state, as_node="agent")
            logger.info("已清除会话检查点：%s", session_id)
        except Exception as exc:
            logger.warning("清除会话检查点失败：%s", exc)

    def _truncate_messages(
        self,
        messages: list[BaseMessage],
        session_id: str,
        config: dict,
    ) -> None:
        """消息窗口截断：超过 SESSION_MAX_MESSAGES 时保留 SystemMessage + 最近 N 条

        Args:
            messages: 当前消息列表
            session_id: 会话 ID
            config: LangGraph 配置（含 thread_id）
        """
        from app.core.config import settings

        if not messages or len(messages) <= settings.SESSION_MAX_MESSAGES:
            return

        # 保留 SystemMessage + 最近 N 条
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        non_system_messages = [m for m in messages if not isinstance(m, SystemMessage)]

        truncated = system_messages + non_system_messages[-settings.SESSION_MAX_MESSAGES :]

        logger.info(
            "消息窗口截断：%d → %d（session_id=%s）",
            len(messages),
            len(truncated),
            session_id,
        )

        # 更新 MemorySaver 中的状态
        try:
            # 保留当前 iteration_count，避免截断后循环计数重置
            current_state = self._compiled_graph.get_state(config)
            current_iteration = 0
            if current_state and current_state.values:
                current_iteration = current_state.values.get("iteration_count", 0)

            truncated_state: AgentState = {
                "messages": truncated,
                "retrieval_context": [],
                "tool_outputs": [],
                "current_status": "idle",
                "iteration_count": current_iteration,
            }
            self._compiled_graph.update_state(config, truncated_state, as_node="agent")
        except Exception as exc:
            logger.warning("消息窗口截断后更新状态失败：%s", exc)
