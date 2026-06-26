import asyncio
import logging
import re
import time
from datetime import datetime
from typing import Any, Final
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from llama_index.core.agent import AgentWorkflow
from llama_index.core.agent.workflow import ToolCallResult
from starlette.requests import Request

from app.agent.workflow import clear_session
from app.core.config import settings
from app.core.dependencies import get_agent
from app.core.enums import ErrorCode, ModelProvider
from app.core.limiter import limiter
from app.exceptions import LLMInvocationError, RateLimitExceededError
from app.models.schemas import ChatRequest, ChatResponse, SourceReference, ToolCallStep

RATE_LIMIT_WINDOW: Final[int] = 60

business_audit_logger = logging.getLogger("audit.business")
business_audit_logger.setLevel(logging.INFO)
if not business_audit_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [BIZ_AUDIT] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    business_audit_logger.addHandler(handler)

__all__ = ["router"]

router = APIRouter()


def _infer_result_type(tool_name: str) -> str:
    """推断工具结果的类型。"""
    match tool_name:
        case "generate_flowchart_code":
            return "mermaid"
        case "generate_html_prototype":
            return "html"
        case "generate_prd_document":
            return "markdown"
        case "search_internal_documents" | "search_web" | "search_csv_data":
            return "search_results"
        case _:
            return "text"


def _extract_sources(result: str, tool_name: str) -> list[SourceReference]:
    """从检索结果中提取来源引用。"""
    if tool_name not in ("search_internal_documents", "search_web", "search_csv_data"):
        return []

    sources: list[SourceReference] = []
    pattern = r"【来源：(.+?)】\n(.+?)(?=【来源：|$)"
    matches = re.findall(pattern, result, re.DOTALL)

    # 也匹配 CSV 数据源格式
    csv_pattern = r"【数据源：(.+?)】\n(.+?)(?=【数据源：|$)"
    csv_matches = re.findall(csv_pattern, result, re.DOTALL)
    matches.extend(csv_matches)

    for source_name, snippet in matches:
        snippet_clean = snippet.strip()[:200]
        sources.append(
            SourceReference(
                source=source_name,
                department="通用",
                score=0.0,
                snippet=snippet_clean,
            )
        )

    return sources


def _extract_intermediate_steps_from_tool_calls(
    tool_call_results: list[ToolCallResult],
) -> tuple[list[str], list[ToolCallStep]]:
    """从 LlamaIndex Agent 工具调用结果中提取中间步骤。

    LlamaIndex 0.14.x 新 API 中，工具调用结果通过事件流收集，
    每个 ToolCallResult 包含工具名、参数和输出。
    """
    tools_used: list[str] = []
    intermediate_steps: list[ToolCallStep] = []

    for tc in tool_call_results:
        tool_name = tc.tool_name
        tool_args = tc.tool_kwargs
        # ToolOutput.content 是文本内容
        tool_output = tc.tool_output
        tool_result_str = (
            str(tool_output.content)
            if hasattr(tool_output, "content")
            else str(tool_output.raw_output)
        )

        result_type = _infer_result_type(tool_name)
        sources = _extract_sources(tool_result_str, tool_name)

        tools_used.append(tool_name)
        intermediate_steps.append(
            ToolCallStep(
                tool_name=tool_name,
                tool_args=tool_args if isinstance(tool_args, dict) else {},
                tool_result=tool_result_str,
                tool_result_type=result_type,
                sources=sources,
                duration_ms=0,
                status=(
                    "error"
                    if (hasattr(tool_output, "is_error") and tool_output.is_error)
                    else "success"
                ),
            )
        )

    return tools_used, intermediate_steps


@router.post("/chat", summary="与 Agent 对话", response_model=None)
@limiter.limit(f"{settings.RATE_LIMIT_RPM}/minute")
async def chat(
    request: Request, body: ChatRequest, agent: AgentWorkflow = Depends(get_agent)
):
    try:
        logger = logging.getLogger(__name__)
        logger.info("收到对话请求：%s...", body.query[:50])

        total_start = time.monotonic()

        # 使用限速器控制请求频率
        from app.core.dependencies import get_rate_limiter

        limiter = get_rate_limiter()

        # 异步等待限速（在事件循环中运行阻塞的 limiter.wait()）
        await asyncio.to_thread(limiter.wait)

        # LlamaIndex Agent 调用（新 API 使用 agent.run()，返回 WorkflowHandler）
        # 通过事件流收集工具调用结果
        handler = agent.run(
            body.query,
            max_iterations=settings.MAX_TOOL_ROUNDS,
            early_stopping_method="generate",
        )

        tool_call_results: list[ToolCallResult] = []
        final_response_text = ""

        async for event in handler.stream_events():
            if isinstance(event, ToolCallResult):
                tool_call_results.append(event)
                # 工具调用间加入智能延迟：根据 Provider 差异化
                if settings.PROVIDER in (ModelProvider.ZHIPU, ModelProvider.XFYUN):
                    await asyncio.sleep(1.0)  # 国内 API 限流更严格，间隔更长
                else:
                    await asyncio.sleep(0.5)

        # 等待 workflow 完成，获取最终结果
        # await handler 直接返回 AgentOutput 对象
        agent_output = await handler
        # agent_output.response 是 ChatMessage，取其文本内容
        response_msg = agent_output.response
        final_response_text = (
            response_msg.content
            if hasattr(response_msg, "content")
            else str(response_msg)
        )

        total_duration_ms = int((time.monotonic() - total_start) * 1000)

        # 提取工具调用步骤
        tools_used, intermediate_steps = _extract_intermediate_steps_from_tool_calls(
            tool_call_results
        )

        # 生成 session_id
        session_id = body.session_id or str(uuid4())

        # 构建 ChatResponse
        chat_response = ChatResponse(
            answer=final_response_text,
            answer_format="markdown",
            tools_used=tools_used,
            intermediate_steps=intermediate_steps,
            total_duration_ms=total_duration_ms,
            session_id=session_id,
        )

        business_audit_logger.info(
            {
                "session_id": session_id,
                "query": body.query[:100],
                "tools_used": tools_used,
                "timestamp": datetime.now().isoformat(),
            }
        )

        logger.info("对话处理完成，使用工具：%s", tools_used)
        return chat_response

    except Exception as exc:
        err_str = str(exc)
        logging.getLogger(__name__).error("对话处理错误：%s", exc)

        match True:
            case _ if "429" in err_str or "Too Many Requests" in err_str:
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": (
                            "请求频率超限，请稍等 1-2 分钟后再试。"
                            "如需更高配额请联系管理员。"
                        ),
                    },
                    headers={"Retry-After": "120"},
                )
            case _ if ErrorCode.RATE_LIMIT in err_str:
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": RateLimitExceededError().message,
                        "error": err_str,
                    },
                    headers={"Retry-After": str(RATE_LIMIT_WINDOW)},
                )
            case _:
                raise HTTPException(
                    status_code=500,
                    detail=LLMInvocationError(f"对话处理失败：{err_str}").message,
                )


@router.delete("/session/{session_id}", summary="清除会话历史")
async def delete_session(
    session_id: str, agent: AgentWorkflow = Depends(get_agent)
) -> dict[str, Any]:
    try:
        logging.getLogger(__name__).info("清除会话：%s", session_id)
        clear_session(session_id)
        return {"message": f"会话 {session_id} 已清除"}
    except Exception as exc:
        logging.getLogger(__name__).error("清除会话错误：%s", exc)
        raise HTTPException(status_code=500, detail=f"清除会话失败：{exc!s}")
