"""代码审查路由模块

实现代码审查相关的 API 端点，包括：
- POST /review/code：提交代码审查请求（支持 SSE 流式输出）
- GET /review/config：查询当前配置
- PUT /review/config：更新配置
- GET /review/mcp/tools：查询 MCP 工具列表
- POST /review/mcp/call：调用 MCP 工具

所有端点通过 FastAPI Depends 注入依赖，限流使用 @limiter.limit() 装饰器。
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from starlette.requests import Request

from app.core.config import settings
from app.core.dependencies import (
    get_feature_flags,
    get_mcp_client,
    get_review_settings,
    get_supervisor,
)
from app.core.enums import MCPServerName, ReviewEventType, ReviewStatus, ReviewType
from app.core.limiter import limiter
from app.exceptions import (
    ConfigPersistenceError,
    MCPToolCallError,
    MCPToolCallTimeoutError,
    MCPToolNotAllowedError,
    ReviewError,
)
from app.mcp.client import MCPClient
from app.agent.review.supervisor import SupervisorAgent
from app.models.schemas import (
    MCPCallRequest,
    MCPCallResponse,
    ReviewConfigResponse,
    ReviewConfigUpdateRequest,
    ReviewRequest,
    ReviewResponse,
    WorkerResult,
)
from app.review.features import FeatureFlags
from app.review.settings import ReviewSettings
from app.security import authenticate
from app.security.auth import get_user_rate_limit
from app.security.rate_limiter import user_rate_limiter

logger = logging.getLogger(__name__)

# 业务审计日志器，独立于业务日志，便于审计追踪和归档
audit_logger = logging.getLogger("audit.review")

__all__ = ["router"]

router = APIRouter(prefix="/review", tags=["代码审查"])

# SSE 心跳间隔（秒）
_SSE_KEEPALIVE_INTERVAL: int = 15


# =============================================================================
# POST /review/code — 提交代码审查请求
# =============================================================================


@router.post("/code", summary="提交代码审查请求", response_model=None)
@limiter.limit(f"{settings.RATE_LIMIT_RPM}/minute")
async def review_code(
    request: Request,
    body: ReviewRequest,
    supervisor: SupervisorAgent = Depends(get_supervisor),
    mcp_client: MCPClient = Depends(get_mcp_client),
    feature_flags: FeatureFlags = Depends(get_feature_flags),
    user: dict = Depends(authenticate),
) -> ReviewResponse | StreamingResponse:
    """提交代码审查请求

    支持同步和 SSE 流式两种响应模式：
    - stream=false（默认）：返回完整的 ReviewResponse JSON
    - stream=true：返回 SSE 事件流，逐个推送审查进度事件

    当 code_url 存在且 code_content 为空时，通过 GitHub MCP Server 获取代码差异。
    当 code_content 和 code_url 同时提供时，优先使用 code_content。
    """
    # 生成或复用 session_id
    session_id = body.session_id or str(uuid4())

    # 按用户限流
    if settings.ENABLE_RATE_LIMIT:
        user_limit = get_user_rate_limit(user)
        user_rate_limiter.check(key=user["hashed_key"], limit=user_limit)

    # 审计日志：审查请求接收
    audit_logger.info(
        "审查请求接收 | session_id=%s | review_type=%s | stream=%s | code_url=%s | code_length=%d",
        session_id,
        body.review_type,
        body.stream,
        body.code_url or "",
        len(body.code_content) if body.code_content else 0,
    )

    # 获取代码内容：优先使用 code_content，否则通过 MCP 获取
    code_content = body.code_content
    if not code_content and body.code_url:
        code_content = await _fetch_code_from_url(body.code_url, mcp_client, feature_flags)
        if not code_content:
            raise HTTPException(
                status_code=400,
                detail=f"无法通过 GitHub MCP Server 获取代码差异：{body.code_url}",
            )

    logger.info(
        "收到代码审查请求：session_id=%s, review_type=%s, stream=%s, code_length=%d",
        session_id,
        body.review_type,
        body.stream,
        len(code_content) if code_content else 0,
    )

    # 构建审查请求对象
    review_request = _build_review_request(
        code_content=code_content or "",
        code_url=body.code_url,
        review_type=body.review_type,
        session_id=session_id,
    )

    if body.stream:
        # 审计日志：SSE 流式审查开始
        audit_logger.info(
            "审查SSE流开始 | session_id=%s | review_type=%s",
            session_id,
            body.review_type,
        )

        # SSE 流式模式
        return StreamingResponse(
            _review_event_generator(
                supervisor=supervisor,
                review_request=review_request,
                session_id=session_id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # 同步模式
        review_start_time = time.monotonic()
        try:
            response = await asyncio.to_thread(supervisor.dispatch, review_request)
            review_duration_ms = int((time.monotonic() - review_start_time) * 1000)

            # 审计日志：审查完成
            audit_logger.info(
                "审查完成 | session_id=%s | review_type=%s"
                " | duration_ms=%d | is_fallback=%s | dimensions=%d",
                session_id,
                body.review_type,
                review_duration_ms,
                response.is_fallback,
                len(response.results),
            )

            return response
        except ReviewError as exc:
            review_duration_ms = int((time.monotonic() - review_start_time) * 1000)

            # 审计日志：审查业务异常
            audit_logger.error(
                "审查业务异常 | session_id=%s | review_type=%s | duration_ms=%d | error=%s",
                session_id,
                body.review_type,
                review_duration_ms,
                exc.message,
            )

            logger.error("代码审查失败：%s", exc)
            raise HTTPException(
                status_code=500,
                detail=exc.message,
            ) from exc
        except Exception as exc:
            review_duration_ms = int((time.monotonic() - review_start_time) * 1000)

            # 审计日志：审查系统异常
            audit_logger.error(
                "审查系统异常 | session_id=%s | review_type=%s | duration_ms=%d | error=%s",
                session_id,
                body.review_type,
                review_duration_ms,
                str(exc),
            )

            logger.error("代码审查异常：%s", exc, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="代码审查服务暂时不可用，请稍后重试",
            ) from exc


# =============================================================================
# GET /review/config — 查询当前配置
# =============================================================================


@router.get("/config", summary="查询当前配置", response_model=ReviewConfigResponse)
async def get_review_config(
    review_settings: ReviewSettings = Depends(get_review_settings),
    user: dict = Depends(authenticate),
) -> ReviewConfigResponse:
    """查询当前审查相关配置

    返回所有审查配置项的当前状态，包括多Agent协作开关、
    MCP 总开关、各 MCP Server 启用状态、审查类型等。
    """
    return ReviewConfigResponse(
        multi_agent_enabled=review_settings.multi_agent_enabled,
        mcp_enabled=review_settings.mcp_enabled,
        mcp_servers=review_settings.mcp_servers,
        review_types=[ReviewType(rt) for rt in review_settings.review_types],
        worker_timeout_seconds=review_settings.worker_timeout_seconds,
        max_concurrent_reviews=review_settings.max_concurrent_reviews,
    )


# =============================================================================
# PUT /review/config — 更新配置
# =============================================================================


@router.put("/config", summary="更新配置", response_model=ReviewConfigResponse)
async def update_review_config(
    body: ReviewConfigUpdateRequest,
    feature_flags: FeatureFlags = Depends(get_feature_flags),
    review_settings: ReviewSettings = Depends(get_review_settings),
    user: dict = Depends(authenticate),
) -> ReviewConfigResponse:
    """更新审查相关配置

    所有字段均为可选，仅更新提供的字段。
    配置更新后立即生效（热更新），无需重启服务。
    已创建的审查会话使用创建时的配置快照，不受影响。

    如果配置持久化失败，响应中会包含 warning 字段。

    权限要求：仅 admin 角色可更新配置。
    """
    # 权限检查：仅 admin 角色可更新配置
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="仅管理员可更新审查配置",
        )
    # 构建更新参数（仅包含非 None 的字段）
    update_kwargs: dict[str, Any] = {}
    if body.multi_agent_enabled is not None:
        update_kwargs["multi_agent_enabled"] = body.multi_agent_enabled
    if body.mcp_enabled is not None:
        update_kwargs["mcp_enabled"] = body.mcp_enabled
    if body.mcp_servers is not None:
        update_kwargs["mcp_servers"] = body.mcp_servers
    if body.review_types is not None:
        update_kwargs["review_types"] = [str(rt) for rt in body.review_types]
    if body.worker_timeout_seconds is not None:
        update_kwargs["worker_timeout_seconds"] = body.worker_timeout_seconds
    if body.max_concurrent_reviews is not None:
        update_kwargs["max_concurrent_reviews"] = body.max_concurrent_reviews

    if not update_kwargs:
        # 无更新字段，直接返回当前配置
        return ReviewConfigResponse(
            multi_agent_enabled=review_settings.multi_agent_enabled,
            mcp_enabled=review_settings.mcp_enabled,
            mcp_servers=review_settings.mcp_servers,
            review_types=[ReviewType(rt) for rt in review_settings.review_types],
            worker_timeout_seconds=review_settings.worker_timeout_seconds,
            max_concurrent_reviews=review_settings.max_concurrent_reviews,
        )

    # 执行热更新
    try:
        feature_flags.update(**update_kwargs)
        logger.info("配置已热更新：%s", list(update_kwargs.keys()))
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    # 尝试持久化
    warning: str | None = None
    try:
        feature_flags.persist()
    except ConfigPersistenceError as exc:
        warning = "配置未持久化，服务重启后将恢复原配置"
        logger.warning("配置持久化失败：%s", exc)

    # 重新读取更新后的配置
    updated_settings = feature_flags.settings
    return ReviewConfigResponse(
        multi_agent_enabled=updated_settings.multi_agent_enabled,
        mcp_enabled=updated_settings.mcp_enabled,
        mcp_servers=updated_settings.mcp_servers,
        review_types=[ReviewType(rt) for rt in updated_settings.review_types],
        worker_timeout_seconds=updated_settings.worker_timeout_seconds,
        max_concurrent_reviews=updated_settings.max_concurrent_reviews,
        warning=warning,
    )


# =============================================================================
# GET /review/mcp/tools — 查询 MCP 工具列表
# =============================================================================


@router.get("/mcp/tools", summary="查询 MCP 工具列表")
async def get_mcp_tools(
    mcp_client: MCPClient = Depends(get_mcp_client),
    feature_flags: FeatureFlags = Depends(get_feature_flags),
    user: dict = Depends(authenticate),
) -> dict[str, Any]:
    """查询可用 MCP 工具列表

    前置条件：mcp_enabled=true，否则返回 HTTP 503。

    返回所有已连接 MCP Server 提供的工具列表，
    包含工具名称、描述、参数定义和所属 Server 名称。
    """
    if not feature_flags.is_mcp_enabled():
        raise HTTPException(
            status_code=503,
            detail="MCP 功能未启用，请在配置中启用 MCP 总开关",
        )

    tools = mcp_client.list_tools()
    servers_status = mcp_client.health_check()

    # 将 MCPToolInfo 对象序列化为字典
    tools_data: list[dict[str, Any]] = []
    for tool in tools:
        if hasattr(tool, "model_dump"):
            tools_data.append(tool.model_dump())
        elif hasattr(tool, "__dict__"):
            tools_data.append(
                {
                    "name": getattr(tool, "name", ""),
                    "description": getattr(tool, "description", ""),
                    "parameters": getattr(tool, "parameters", {}),
                    "server_name": getattr(tool, "server_name", ""),
                }
            )

    return {
        "tools": tools_data,
        "servers_status": servers_status,
    }


# =============================================================================
# POST /review/mcp/call — 调用 MCP 工具
# =============================================================================


@router.post("/mcp/call", summary="调用 MCP 工具", response_model=MCPCallResponse)
async def call_mcp_tool(
    body: MCPCallRequest,
    mcp_client: MCPClient = Depends(get_mcp_client),
    feature_flags: FeatureFlags = Depends(get_feature_flags),
    user: dict = Depends(authenticate),
) -> MCPCallResponse:
    """调用 MCP 工具

    前置条件：mcp_enabled=true，否则返回 HTTP 503。

    通过 MCP Client 路由工具调用到对应的适配器执行。
    支持指定目标 Server 名称，也可自动路由。
    """
    if not feature_flags.is_mcp_enabled():
        raise HTTPException(
            status_code=503,
            detail="MCP 功能未启用，请在配置中启用 MCP 总开关",
        )

    start_time = time.monotonic()

    try:
        result = await asyncio.to_thread(
            mcp_client.call_tool,
            tool_name=body.tool_name,
            arguments=body.arguments,
            server_name=body.server_name,
        )
        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        # 确定实际执行该工具的 Server 名称
        server_name = body.server_name or ""
        if not server_name:
            # 通过公开方法从注册表获取
            registry = mcp_client.registry
            server_name = registry.get_tool_server(body.tool_name)

        return MCPCallResponse(
            tool_name=body.tool_name,
            result=result,
            server_name=server_name,
            duration_ms=elapsed_ms,
        )

    except MCPToolCallTimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=f"工具调用超时，请稍后重试：{exc.message}",
        ) from exc

    except MCPToolNotAllowedError as exc:
        raise HTTPException(
            status_code=403,
            detail=f"无权调用该 MCP 工具：{exc.message}",
        ) from exc

    except MCPToolCallError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"MCP 工具调用失败：{exc.message}",
        ) from exc

    except Exception as exc:
        logger.error("MCP 工具调用异常：%s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"MCP 工具调用异常：{exc!s}",
        ) from exc


# =============================================================================
# 内部辅助函数
# =============================================================================


async def _fetch_code_from_url(
    code_url: str,
    mcp_client: MCPClient,
    feature_flags: FeatureFlags,
) -> str | None:
    """通过 GitHub MCP Server 获取代码差异

    当 code_url 存在且 code_content 为空时调用。
    如果 MCP 功能未启用或 GitHub Server 不可用，返回 None。

    Args:
        code_url: 代码仓库 PR 链接
        mcp_client: MCP 客户端实例
        feature_flags: 功能开关管理器实例

    Returns:
        代码差异内容，获取失败时返回 None
    """
    if not feature_flags.is_mcp_enabled():
        logger.warning("MCP 功能未启用，无法获取代码差异：%s", code_url)
        return None

    if not feature_flags.is_mcp_server_enabled(MCPServerName.GITHUB):
        logger.warning("GitHub MCP Server 未启用，无法获取代码差异：%s", code_url)
        return None

    try:
        result = await asyncio.to_thread(
            mcp_client.call_tool,
            tool_name="github_get_pr_diff",
            arguments={"pr_url": code_url},
            server_name=MCPServerName.GITHUB,
        )
        return str(result) if result else None
    except Exception as exc:
        logger.error("通过 GitHub MCP Server 获取代码差异失败：%s", exc)
        return None


def _build_review_request(
    code_content: str,
    code_url: str | None,
    review_type: ReviewType,
    session_id: str,
) -> Any:
    """构建审查请求对象

    创建一个简单的命名空间对象，供 SupervisorAgent.dispatch() 使用。

    Args:
        code_content: 代码内容
        code_url: 代码仓库 PR 链接
        review_type: 审查类型
        session_id: 会话 ID

    Returns:
        审查请求对象
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        code_content=code_content,
        code_url=code_url,
        review_type=review_type,
        session_id=session_id,
    )


async def _review_event_generator(
    supervisor: SupervisorAgent,
    review_request: Any,
    session_id: str,
) -> AsyncGenerator[str, None]:
    """SSE 事件生成器

    通过回调机制实现实时推送，每个 Worker 完成后立即推送事件，
    而非等待全部完成后再批量推送。

    事件类型：
    - review_start：审查开始
    - worker_start：Worker 开始执行
    - worker_result：Worker 完成审查
    - worker_timeout：Worker 执行超时
    - worker_error：Worker 执行失败
    - summary_start：Summary Agent 开始汇总
    - review_end：审查完成

    Args:
        supervisor: SupervisorAgent 实例
        review_request: 审查请求对象
        session_id: 审查会话 ID

    Yields:
        SSE 格式的事件字符串
    """
    event_queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _put_event(event_str: str) -> None:
        """线程安全地将事件放入 asyncio.Queue

        由于 supervisor.dispatch() 在 ThreadPoolExecutor 中执行，
        回调在同步线程中调用，需要通过 loop.call_soon_threadsafe
        将 put 操作调度到事件循环线程。
        """
        loop.call_soon_threadsafe(event_queue.put_nowait, event_str)

    async def _produce_events() -> None:
        """通过回调机制实时生产 SSE 事件到队列"""
        try:
            # 发送 review_start 事件
            review_start_data = {
                "session_id": session_id,
                "review_type": str(review_request.review_type),
                "timestamp": datetime.now().isoformat(),
            }
            await event_queue.put(
                _format_sse_event(ReviewEventType.REVIEW_START, review_start_data, session_id)
            )

            # 定义回调函数 —— 在同步线程中执行，通过 _put_event 线程安全地放入队列

            def on_worker_start(dimension: str) -> None:
                """Worker 开始执行回调"""
                worker_start_data = {
                    "session_id": session_id,
                    "dimension": dimension,
                    "timestamp": datetime.now().isoformat(),
                }
                _put_event(
                    _format_sse_event(ReviewEventType.WORKER_START, worker_start_data, session_id)
                )

            def on_worker_result(result: WorkerResult) -> None:
                """Worker 完成回调"""
                if result.status == ReviewStatus.COMPLETED:
                    event_type = ReviewEventType.WORKER_RESULT
                elif result.status == ReviewStatus.TIMEOUT:
                    event_type = ReviewEventType.WORKER_TIMEOUT
                else:
                    event_type = ReviewEventType.WORKER_ERROR

                result_data = {
                    "session_id": session_id,
                    "dimension": str(result.dimension),
                    "status": str(result.status),
                    "findings": [
                        f.model_dump() if hasattr(f, "model_dump") else str(f)
                        for f in result.findings
                    ],
                    "duration_ms": result.duration_ms,
                    "timestamp": datetime.now().isoformat(),
                }
                if result.error_message:
                    result_data["error_message"] = result.error_message
                _put_event(_format_sse_event(event_type, result_data, session_id))

            def on_summary_start() -> None:
                """Summary Agent 开始汇总回调"""
                summary_start_data = {
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                }
                _put_event(
                    _format_sse_event(ReviewEventType.SUMMARY_START, summary_start_data, session_id)
                )

            # 在线程池中执行审查（因为 dispatch 是同步的），传入回调实现实时推送
            response = await asyncio.to_thread(
                supervisor.dispatch,
                review_request,
                on_worker_start,
                on_worker_result,
                on_summary_start,
            )

            # 发送 review_end 事件（包含完整 ReviewResponse）
            review_end_data = response.model_dump() if hasattr(response, "model_dump") else {}
            review_end_data["timestamp"] = datetime.now().isoformat()
            await event_queue.put(
                _format_sse_event(ReviewEventType.REVIEW_END, review_end_data, session_id)
            )

            # 审计日志：SSE 审查完成
            audit_logger.info(
                "审查SSE流完成 | session_id=%s | review_type=%s"
                " | duration_ms=%d | is_fallback=%s | dimensions=%d",
                session_id,
                str(review_request.review_type),
                response.total_duration_ms,
                response.is_fallback,
                len(response.results),
            )

        except Exception as exc:
            logger.error("SSE 审查流式执行异常（session_id=%s）：%s", session_id, exc)

            # 审计日志：SSE 审查异常
            audit_logger.error(
                "审查SSE流异常 | session_id=%s | error=%s",
                session_id,
                str(exc),
            )

            # 发送 worker_error 事件
            error_data = {
                "session_id": session_id,
                "error": str(exc),
                "timestamp": datetime.now().isoformat(),
            }
            await event_queue.put(
                _format_sse_event(ReviewEventType.WORKER_ERROR, error_data, session_id)
            )

        finally:
            await event_queue.put(None)  # 哨兵值，通知消费结束

    # 启动事件生产者协程
    producer_task = asyncio.create_task(_produce_events())

    try:
        while True:
            try:
                # 等待事件，超时后发送心跳
                item = await asyncio.wait_for(event_queue.get(), timeout=_SSE_KEEPALIVE_INTERVAL)
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
        logger.info("SSE 审查连接被客户端断开（session_id=%s）", session_id)
        audit_logger.warning(
            "审查SSE流客户端断开 | session_id=%s",
            session_id,
        )

    finally:
        # 确保生产者任务被取消
        if not producer_task.done():
            producer_task.cancel()
            try:
                await producer_task
            except asyncio.CancelledError:
                pass


def _format_sse_event(
    event_type: ReviewEventType,
    data: dict[str, Any],
    session_id: str,
) -> str:
    """格式化 SSE 事件字符串

    SSE 事件格式：event: {type}\\ndata: {json}\\nid: {event_id}\\n\\n

    Args:
        event_type: 事件类型
        data: 事件数据
        session_id: 会话 ID

    Returns:
        格式化的 SSE 事件字符串
    """
    try:
        data_json = json.dumps(data, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        data_json = json.dumps({"error": "数据序列化失败"}, ensure_ascii=False)

    event_id = f"{session_id}-{event_type.value}"
    return f"event: {event_type.value}\ndata: {data_json}\nid: {event_id}\n\n"
