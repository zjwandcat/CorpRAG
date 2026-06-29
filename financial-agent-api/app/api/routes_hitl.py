"""HITL 审批 API 路由模块

提供 Human-in-the-Loop 审批管理的 API 端点：
- GET /hitl/approvals — 获取所有待审批项
- GET /hitl/approvals/{approval_id} — 获取指定审批项
- POST /hitl/approvals/{approval_id}/approve — 批准审批
- POST /hitl/approvals/{approval_id}/reject — 拒绝审批
- GET /hitl/status — 查询 HITL 功能状态

所有端点通过 FastAPI Depends 注入 authenticate 依赖，确保请求方已认证。
当 HITL 功能未启用时，相关端点返回 404 状态码，不中断核心链路。
"""

from fastapi import APIRouter, Depends, HTTPException, Request

from app.agent.hitl_manager import HITLManager
from app.core.config import settings
from app.core.dependencies import get_hitl_manager
from app.core.limiter import limiter
from app.core.logging_config import get_logger
from app.models.schemas import HITLApprovalResponse, HITLApprovalResult
from app.security import authenticate

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/hitl/approvals",
    summary="获取所有待审批项",
    response_model=list[HITLApprovalResponse],
)
@limiter.limit("10/minute")
async def list_approvals(
    request: Request,
    user: dict = Depends(authenticate),
) -> list[HITLApprovalResponse]:
    """获取所有状态为 PENDING 的审批项列表

    当 HITL 功能未启用时返回 404。
    """
    manager = get_hitl_manager()
    if manager is None:
        raise HTTPException(status_code=404, detail="HITL 功能未启用")
    try:
        approvals = manager.get_pending_approvals()
        logger.info(
            "HITL审批列表查询 | pending_count=%d | user=%s",
            len(approvals),
            user.get("role", "unknown"),
        )
        return approvals
    except Exception as exc:
        logger.warning("HITL审批列表查询异常，降级处理：%s", exc)
        return manager.get_pending_approvals()


@router.get(
    "/hitl/approvals/{approval_id}",
    summary="获取指定审批项",
    response_model=HITLApprovalResponse,
)
@limiter.limit("10/minute")
async def get_approval(
    request: Request,
    approval_id: str,
    user: dict = Depends(authenticate),
) -> HITLApprovalResponse:
    """根据 approval_id 获取指定审批项详情

    Args:
        approval_id: 审批唯一标识

    当审批项不存在时返回 404。
    """
    manager = get_hitl_manager()
    if manager is None:
        raise HTTPException(status_code=404, detail="HITL 功能未启用")
    approval = manager.get_approval(approval_id)
    if approval is None:
        raise HTTPException(status_code=404, detail=f"审批项不存在：{approval_id}")
    try:
        logger.info(
            "HITL审批详情查询 | approval_id=%s | user=%s",
            approval_id,
            user.get("role", "unknown"),
        )
    except Exception as exc:
        logger.warning("HITL审批详情日志异常，降级处理：%s", exc)
    return approval


@router.post(
    "/hitl/approvals/{approval_id}/approve",
    summary="批准审批",
    response_model=HITLApprovalResult,
)
@limiter.limit("10/minute")
async def approve_approval(
    request: Request,
    approval_id: str,
    reason: str | None = None,
    user: dict = Depends(authenticate),
) -> HITLApprovalResult:
    """批准指定的审批请求

    Args:
        approval_id: 审批唯一标识
        reason: 批准理由（可选）

    当审批项不存在时返回 404，审批已处理时返回 400。
    """
    manager = get_hitl_manager()
    if manager is None:
        raise HTTPException(status_code=404, detail="HITL 功能未启用")
    try:
        result = manager.resolve_approval(approval_id, action="approve", reason=reason)
        try:
            logger.info(
                "HITL审批批准 | approval_id=%s | user=%s",
                approval_id,
                user.get("role", "unknown"),
            )
        except Exception as exc:
            logger.warning("HITL审批批准日志异常，降级处理：%s", exc)
        return result
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/hitl/approvals/{approval_id}/reject",
    summary="拒绝审批",
    response_model=HITLApprovalResult,
)
@limiter.limit("10/minute")
async def reject_approval(
    request: Request,
    approval_id: str,
    reason: str | None = None,
    user: dict = Depends(authenticate),
) -> HITLApprovalResult:
    """拒绝指定的审批请求

    Args:
        approval_id: 审批唯一标识
        reason: 拒绝理由（可选）

    当审批项不存在时返回 404，审批已处理时返回 400。
    """
    manager = get_hitl_manager()
    if manager is None:
        raise HTTPException(status_code=404, detail="HITL 功能未启用")
    try:
        result = manager.resolve_approval(approval_id, action="reject", reason=reason)
        try:
            logger.info(
                "HITL审批拒绝 | approval_id=%s | reason=%s | user=%s",
                approval_id,
                reason or "未提供",
                user.get("role", "unknown"),
            )
        except Exception as exc:
            logger.warning("HITL审批拒绝日志异常，降级处理：%s", exc)
        return result
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/hitl/status", summary="查询 HITL 功能状态")
@limiter.limit("10/minute")
async def hitl_status(
    request: Request,
    user: dict = Depends(authenticate),
) -> dict[str, bool | list[str] | int]:
    """查询 HITL 功能的全局状态和配置信息

    返回 HITL 是否启用、高风险工具列表和审批超时时间。
    """
    return {
        "enabled": settings.HITL_ENABLED,
        "high_risk_tools": [
            t.strip() for t in settings.HITL_HIGH_RISK_TOOLS.split(",") if t.strip()
        ],
        "approval_timeout_seconds": settings.HITL_APPROVAL_TIMEOUT_SECONDS,
    }