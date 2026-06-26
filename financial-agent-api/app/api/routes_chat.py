import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Final
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.requests import Request

from app.agent.graph import AgentGraph
from app.core.config import settings
from app.core.dependencies import (
    get_agent_graph,
    get_blueprint_client,
    get_engine_router,
)
from app.core.enums import ErrorCode, RAGEngine
from app.core.limiter import limiter
from app.exceptions import LLMInvocationError, RateLimitExceededError
from app.models.schemas import ChatRequest, StreamErrorData
from app.rag.engine_router import EngineRouter

RATE_LIMIT_WINDOW: Final[int] = 60
SSE_KEEPALIVE_INTERVAL: Final[int] = 15  # 秒

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


# =============================================================================
# 任务 5.1: POST /chat/stream — SSE 流式对话
# =============================================================================


@router.post("/chat/stream", summary="SSE 流式对话")
@limiter.limit(f"{settings.RATE_LIMIT_RPM}/minute")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    agent_graph: AgentGraph = Depends(get_agent_graph),
):
    """SSE 流式对话接口

    通过 POST 方法建立 SSE 连接，实时推送 Agent 推理状态事件。
    前端需使用 fetch + ReadableStream 方案（非 EventSource，因为 POST 请求）。

    SSE 事件格式：event: {type}\\ndata: {json}\\nid: {event_id}\\n\\n
    心跳：每 15 秒发送 : keepalive\\n\\n 注释行
    """
    # 校验 thread_id（即 session_id）合法性
    thread_id = body.session_id
    if thread_id is not None:
        thread_id = thread_id.strip()
        if not thread_id:
            raise HTTPException(status_code=400, detail="session_id 不能为空字符串")
    else:
        thread_id = str(uuid4())

    logger = logging.getLogger(__name__)
    logger.info("收到 SSE 流式请求：%s... (thread_id=%s)", body.query[:50], thread_id)

    async def event_generator():
        """SSE 事件生成器

        逐个推送 SSE 事件，包含心跳机制和异常处理。
        使用 asyncio.Queue 合并事件流和心跳信号。
        """
        event_queue: asyncio.Queue[str | None] = asyncio.Queue()
        stream_finished = False

        async def _produce_events():
            """从 agent_graph.run_stream() 生产事件到队列"""
            nonlocal stream_finished
            try:
                async for event in agent_graph.run_stream(
                    query=body.query,
                    thread_id=thread_id,
                ):
                    # 将事件转换为 SSE 格式
                    event_type = event.get("event", "message")
                    event_data = event.get("data", {})
                    event_id = event.get("id", "")

                    # 序列化 data 为 JSON
                    try:
                        data_json = json.dumps(event_data, ensure_ascii=False)
                    except TypeError, ValueError:
                        data_json = json.dumps({"error": "数据序列化失败"}, ensure_ascii=False)

                    sse_frame = f"event: {event_type}\ndata: {data_json}\nid: {event_id}\n\n"
                    await event_queue.put(sse_frame)

            except Exception as exc:
                # 推送 stream_error 事件
                logger.error("SSE 流式执行异常（thread_id=%s）：%s", thread_id, exc)

                error_type = type(exc).__name__
                error_message = "对话处理失败，请稍后重试"

                if isinstance(exc, RateLimitExceededError):
                    error_message = "请求频率超限，请稍后重试"
                elif isinstance(exc, LLMInvocationError):
                    error_message = "AI 服务暂时不可用，请稍后重试"

                error_data = StreamErrorData(
                    thread_id=thread_id,
                    timestamp=datetime.now().isoformat(),
                    error_type=error_type,
                    error_message=error_message,
                )

                try:
                    error_json = error_data.model_dump_json()
                except Exception:
                    error_json = json.dumps(
                        {
                            "thread_id": thread_id,
                            "error_type": error_type,
                            "error_message": error_message,
                        },
                        ensure_ascii=False,
                    )

                sse_frame = f"event: stream_error\ndata: {error_json}\nid: {thread_id}-error\n\n"
                await event_queue.put(sse_frame)

            finally:
                stream_finished = True
                await event_queue.put(None)  # 哨兵值，通知消费结束

        # 启动事件生产者协程
        producer_task = asyncio.create_task(_produce_events())

        try:
            while True:
                try:
                    # 等待事件，超时后发送心跳
                    item = await asyncio.wait_for(event_queue.get(), timeout=SSE_KEEPALIVE_INTERVAL)
                except TimeoutError:
                    # 超时，发送心跳注释行
                    yield ": keepalive\n\n"
                    continue

                if item is None:
                    # 哨兵值，流结束
                    break

                yield item

        except asyncio.CancelledError:
            # 客户端断开连接
            logger.info("SSE 连接被客户端断开（thread_id=%s）", thread_id)

        finally:
            # 确保生产者任务被取消
            if not producer_task.done():
                producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =============================================================================
# 任务 5.2: POST /chat — 同步对话（保持向后兼容）
# =============================================================================


@router.post("/chat", summary="与 Agent 对话", response_model=None)
@limiter.limit(f"{settings.RATE_LIMIT_RPM}/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    engine_router: EngineRouter = Depends(get_engine_router),
):
    """同步对话接口（保持向后兼容）

    内部通过 EngineRouter 调用 AgentGraph.run()，
    返回格式与重构前完全一致（ChatResponse JSON）。
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info("收到对话请求：%s... (rag_engine=%s)", body.query[:50], body.rag_engine)

        # 将 rag_engine 字符串转换为枚举
        rag_engine = (
            RAGEngine(body.rag_engine)
            if body.rag_engine in ("builtin", "blueprint")
            else RAGEngine.BUILTIN
        )

        response = await engine_router.route(
            query=body.query,
            session_id=body.session_id,
            rag_engine=rag_engine,
        )

        business_audit_logger.info(
            {
                "session_id": response.session_id,
                "query": body.query[:100],
                "tools_used": response.tools_used,
                "rag_engine": response.rag_engine,
                "is_fallback": response.is_fallback,
                "timestamp": datetime.now().isoformat(),
            }
        )

        logger.info(
            "对话处理完成，使用工具：%s, RAG引擎：%s, 降级：%s",
            response.tools_used,
            response.rag_engine,
            response.is_fallback,
        )
        return response

    except Exception as exc:
        err_str = str(exc)
        logging.getLogger(__name__).error("对话处理错误：%s", exc)

        match True:
            case _ if ErrorCode.RATE_LIMIT in err_str:
                return JSONResponse(
                    status_code=429,
                    content={"detail": RateLimitExceededError().message},
                    headers={"Retry-After": str(RATE_LIMIT_WINDOW)},
                )
            case _:
                raise HTTPException(
                    status_code=500, detail=LLMInvocationError("对话处理失败").message
                )


# =============================================================================
# 任务 5.3: DELETE /session/{session_id} — 清除会话（同时清除 MemorySaver 检查点）
# =============================================================================


@router.delete("/session/{session_id}", summary="清除会话历史")
async def clear_session(
    session_id: str, agent_graph: AgentGraph = Depends(get_agent_graph)
) -> dict[str, Any]:
    """清除会话历史

    同时清除 AgentGraph（MemorySaver 检查点）中的会话数据。
    返回格式不变。
    """
    try:
        logging.getLogger(__name__).info("清除会话：%s", session_id)
        await asyncio.to_thread(agent_graph.clear_session, session_id)
        return {"message": f"会话 {session_id} 已清除"}
    except Exception as exc:
        logging.getLogger(__name__).error("清除会话错误：%s", exc)
        raise HTTPException(status_code=500, detail=f"清除会话失败：{exc!s}")


# =============================================================================
# 其他接口（保持不变）
# =============================================================================


@router.get("/rag/blueprint/status", summary="查询 Blueprint 可用性")
async def blueprint_status(
    client=Depends(get_blueprint_client),
) -> dict[str, bool | str]:
    """查询 NVIDIA RAG Blueprint 是否可用

    前端据此决定是否启用 Blueprint 复选框。
    """
    configured = client.is_configured()
    available = False
    if configured:
        available = await client.health_check()
    return {
        "available": available,
        "api_url": (
            settings.BLUEPRINT_API_URL[:30] + "..."
            if len(settings.BLUEPRINT_API_URL) > 30
            else settings.BLUEPRINT_API_URL
        ),
        "model_name": settings.BLUEPRINT_LLM_MODELNAME or "(默认)",
    }
