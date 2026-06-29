"""MLOps API 路由模块

提供 RAG 评估、A/B 测试配置、漂移检测和健康检查等 MLOps 相关 API 端点。
"""

import logging

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.dependencies import (
    get_ab_router,
    get_drift_detector,
    get_evaluator,
    get_tracker,
)
from app.models.schemas import (
    ABConfigRequest,
    ABConfigResponse,
    ABMetricsResponse,
    DriftStatusResponse,
    EvalRequest,
    EvalResponse,
    MLOpsHealthResponse,
)
from app.security import authenticate

logger = logging.getLogger(__name__)

__all__ = ["router"]

router = APIRouter()


# =============================================================================
# 辅助函数
# =============================================================================


def require_admin_role(user: dict[str, Any]) -> None:
    """验证用户是否为管理员角色

    Args:
        user: 由 authenticate() 返回的用户信息字典

    Raises:
        HTTPException: 当用户不是管理员时返回 403 Forbidden
    """
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="此操作需要管理员权限",
        )


# =============================================================================
# POST /api/v1/mlops/eval — 触发 RAG 评估任务
# =============================================================================


@router.post(
    "/eval",
    response_model=EvalResponse,
    summary="触发 RAG 评估任务",
)
async def trigger_evaluation(
    body: EvalRequest,
    evaluator=Depends(get_evaluator),
    user: dict[str, Any] = Depends(authenticate),
) -> EvalResponse:
    """触发 RAG 评估任务，返回四维度评分 JSON 报告

    评估维度：
    - faithfulness_score: 忠实度得分
    - answer_relevancy_score: 答案相关性得分
    - context_precision_score: 上下文精确率
    - context_recall_score: 上下文召回率

    Args:
        body: 评估请求，包含数据集路径
        evaluator: RAGEvaluator 单例
        user: 已认证用户信息

    Returns:
        EvalResponse: 包含四维度评分的评估报告

    Raises:
        HTTPException: 评估执行失败时返回 500
    """
    try:
        logger.info("触发 RAG 评估任务 | dataset_path=%s | user=%s", body.dataset_path, user.get("name"))

        # 执行评估（evaluate 是同步方法，直接返回 EvalResponse）
        result = evaluator.evaluate(dataset_path=body.dataset_path)

        logger.info(
            "RAG 评估完成 | faithfulness=%.4f | relevancy=%.4f | precision=%.4f | recall=%.4f",
            result.faithfulness_score,
            result.answer_relevancy_score,
            result.context_precision_score,
            result.context_recall_score,
        )

        return result

    except FileNotFoundError as exc:
        logger.error("评估数据集不存在: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"评估数据集不存在: {body.dataset_path}",
        )
    except Exception as exc:
        logger.error("RAG 评估执行失败: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"评估执行失败: {exc!s}",
        )


# =============================================================================
# GET /api/v1/mlops/ab-config — 获取当前 A/B 测试配置
# =============================================================================


@router.get(
    "/ab-config",
    response_model=ABConfigResponse,
    summary="获取当前 A/B 测试配置",
)
async def get_ab_config(
    ab_router=Depends(get_ab_router),
    user: dict[str, Any] = Depends(authenticate),
) -> ABConfigResponse:
    """获取当前 A/B 测试策略配置

    返回当前 A/B 测试的流量分配和策略配置信息。

    Args:
        ab_router: ABTestRouter 单例
        user: 已认证用户信息

    Returns:
        ABConfigResponse: A/B 测试配置信息
    """
    logger.info("查询 A/B 测试配置 | user=%s", user.get("name"))

    config = ab_router.get_config()

    return ABConfigResponse(
        bucket_a_ratio=config.bucket_a_ratio,
        bucket_a_strategy=config.bucket_a_strategy,
        bucket_b_strategy=config.bucket_b_strategy,
        enabled=config.enabled,
    )


# =============================================================================
# PUT /api/v1/mlops/ab-config — 动态更新 A/B 测试策略配置
# =============================================================================


@router.put(
    "/ab-config",
    response_model=ABConfigResponse,
    summary="动态更新 A/B 测试策略配置（仅限管理员）",
)
async def update_ab_config(
    body: ABConfigRequest,
    ab_router=Depends(get_ab_router),
    user: dict[str, Any] = Depends(authenticate),
) -> ABConfigResponse:
    """动态更新 A/B 测试策略配置（仅限管理员）

    允许管理员动态调整 A/B 测试的流量分配和策略配置。
    更新后立即生效，无需重启服务。

    Args:
        body: A/B 测试配置请求
        ab_router: ABTestRouter 单例
        user: 已认证用户信息

    Returns:
        ABConfigResponse: 更新后的 A/B 测试配置

    Raises:
        HTTPException: 非管理员访问时返回 403
    """
    # 管理员角色校验
    require_admin_role(user)

    try:
        logger.info(
            "更新 A/B 测试配置 | user=%s | bucket_a_ratio=%.2f | enabled=%s",
            user.get("name"),
            body.bucket_a_ratio,
            body.enabled,
        )

        # 更新配置
        ab_router.update_config(body)

        # 返回更新后的配置
        config = ab_router.get_config()

        return ABConfigResponse(
            bucket_a_ratio=config.bucket_a_ratio,
            bucket_a_strategy=config.bucket_a_strategy,
            bucket_b_strategy=config.bucket_b_strategy,
            enabled=config.enabled,
        )

    except Exception as exc:
        logger.error("更新 A/B 测试配置失败: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新配置失败: {exc!s}",
        )


# =============================================================================
# GET /api/v1/mlops/ab-metrics — 获取各 Bucket 指标
# =============================================================================


@router.get(
    "/ab-metrics",
    response_model=ABMetricsResponse,
    summary="获取各 Bucket 指标",
)
async def get_ab_metrics(
    ab_router=Depends(get_ab_router),
    user: dict[str, Any] = Depends(authenticate),
) -> ABMetricsResponse:
    """获取 A/B 测试各 Bucket 的性能指标

    返回 Bucket A 和 Bucket B 的平均延迟和总请求数等指标。

    Args:
        ab_router: ABTestRouter 单例
        user: 已认证用户信息

    Returns:
        ABMetricsResponse: A/B 测试指标数据
    """
    logger.info("查询 A/B 测试指标 | user=%s", user.get("name"))

    metrics = ab_router.get_metrics()

    return ABMetricsResponse(
        bucket_a_avg_latency_ms=metrics.bucket_a_avg_latency_ms,
        bucket_b_avg_latency_ms=metrics.bucket_b_avg_latency_ms,
        bucket_a_total_requests=metrics.bucket_a_total_requests,
        bucket_b_total_requests=metrics.bucket_b_total_requests,
    )


# =============================================================================
# GET /api/v1/mlops/drift-status — 获取漂移检测状态
# =============================================================================


@router.get(
    "/drift-status",
    response_model=DriftStatusResponse,
    summary="获取漂移检测状态",
)
async def get_drift_status(
    drift_detector=Depends(get_drift_detector),
    user: dict[str, Any] = Depends(authenticate),
) -> DriftStatusResponse:
    """获取查询漂移检测状态

    返回漂移检测器的当前状态、最近检测时间和漂移分数等信息。

    Args:
        drift_detector: QueryDriftDetector 单例
        user: 已认证用户信息

    Returns:
        DriftStatusResponse: 漂移检测状态信息
    """
    logger.info("查询漂移检测状态 | user=%s", user.get("name"))

    status_info = drift_detector.get_status()

    return DriftStatusResponse(
        status=str(status_info.status.value),
        last_check_timestamp=None,
        last_drift_score=None,
        alert_count=status_info.drift_count,
    )


# =============================================================================
# GET /api/v1/mlops/health — MLOps 模块健康检查
# =============================================================================


@router.get(
    "/health",
    response_model=MLOpsHealthResponse,
    summary="MLOps 模块健康检查",
)
async def mlops_health_check(
    tracker=Depends(get_tracker),
    drift_detector=Depends(get_drift_detector),
    ab_router=Depends(get_ab_router),
    user: dict[str, Any] = Depends(authenticate),
) -> MLOpsHealthResponse:
    """MLOps 模块健康检查

    检查各 MLOps 组件的运行状态：
    - MLflow 连接状态
    - 漂移检测器状态
    - A/B 测试启用状态

    Args:
        tracker: LLMExperimentTracker 单例
        drift_detector: QueryDriftDetector 单例
        ab_router: ABTestRouter 单例
        user: 已认证用户信息

    Returns:
        MLOpsHealthResponse: MLOps 各模块健康状态
    """
    logger.info("MLOps 健康检查 | user=%s", user.get("name"))

    # 检查 MLflow 连接状态
    mlflow_connected = False
    try:
        mlflow_connected = tracker.is_available()
    except Exception as exc:
        logger.warning("MLflow 连接检查失败: %s", exc)

    # 获取漂移检测器状态
    drift_status = "unavailable"
    try:
        drift_status = str(drift_detector.get_status().status.value)
    except Exception as exc:
        logger.warning("漂移检测器状态检查失败: %s", exc)

    # 获取 A/B 测试启用状态
    ab_enabled = False
    try:
        ab_enabled = ab_router.get_config().enabled
    except Exception as exc:
        logger.warning("A/B 测试状态检查失败: %s", exc)

    return MLOpsHealthResponse(
        mlflow_connected=mlflow_connected,
        drift_detector_status=drift_status,
        ab_testing_enabled=ab_enabled,
    )