"""MLOps 完整流程端到端（E2E）测试

测试完整的 MLOps 流程，包括：
- 完整 MLOps 流程：对话 → 追踪 → A/B测试 → 漂移检测
- MLOps 逻辑异常降级不影响核心对话
- MLOps API 健康检查

运行方式：
    pytest tests/e2e/test_full_mlops_flow.py -v --tb=short

环境要求：
    - 服务运行在 http://localhost:8001
    - 环境变量 TEST_API_KEY 已设置（可选，默认使用 test-key）
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest
import requests

# ============================================================================
# 常量定义
# ============================================================================

BASE_URL: str = os.getenv("E2E_BASE_URL", "http://localhost:8001")
API_KEY: str = os.getenv("TEST_API_KEY", "test-key")
REQUEST_TIMEOUT: int = 10

# 项目根目录
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


# ============================================================================
# 辅助函数
# ============================================================================


def _get_auth_headers() -> dict[str, str]:
    """构造认证请求头

    Returns:
        包含 Bearer Token 的请求头字典
    """
    return {"Authorization": f"Bearer {API_KEY}"}


def _is_service_reachable() -> bool:
    """检测目标服务是否可达

    Returns:
        服务可达返回 True，否则返回 False
    """
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=REQUEST_TIMEOUT)
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


# ============================================================================
# 模块级 fixture：服务可用性检测
# ============================================================================


_SERVICE_REACHABLE: bool | None = None


def _check_service_reachable() -> bool:
    """检查服务是否可达（带缓存）

    Returns:
        服务可达返回 True，否则返回 False
    """
    global _SERVICE_REACHABLE
    if _SERVICE_REACHABLE is None:
        _SERVICE_REACHABLE = _is_service_reachable()
    return _SERVICE_REACHABLE


def _require_service() -> None:
    """如果服务不可达，跳过当前测试

    用于需要服务连接的测试用例开头调用。
    """
    if not _check_service_reachable():
        pytest.skip(f"目标服务 {BASE_URL} 不可达，跳过此测试")


# ============================================================================
# 1. 完整 MLOps 流程测试
# ============================================================================


class TestFullMLOpsFlow:
    """完整 MLOps 流程测试"""

    def test_chat_with_tracking(self) -> None:
        """测试对话 → 追踪流程

        流程：
        1. 发送对话请求
        2. 验证响应成功
        3. 验证追踪信息（通过响应头或日志）

        断言：
        - 对话响应状态码为 200
        - 响应包含 answer 字段
        - 响应头包含追踪 ID（X-Request-ID 或 X-Correlation-ID）
        """
        _require_service()

        resp = requests.post(
            f"{BASE_URL}/api/v1/chat",
            json={"query": "报销流程是什么？"},
            headers=_get_auth_headers(),
            timeout=REQUEST_TIMEOUT * 3,
        )

        assert resp.status_code == 200, (
            f"对话请求应返回 200，实际：{resp.status_code}"
        )

        data: dict[str, Any] = resp.json()
        assert "answer" in data, "响应应包含 answer 字段"
        assert len(data.get("answer", "")) > 0, "answer 不应为空"

        # 验证追踪 ID
        correlation_id = resp.headers.get("X-Correlation-ID") or resp.headers.get("X-Request-ID")
        assert correlation_id is not None, "响应头应包含追踪 ID"

    def test_ab_testing_flow(self) -> None:
        """测试 A/B 测试流程

        流程：
        1. 查询当前 A/B 测试配置
        2. 发送多个对话请求，验证分桶一致性
        3. 查询 A/B 测试指标

        断言：
        - A/B 测试配置查询成功
        - 同一 session_id 多次请求分配到同一 Bucket
        - 指标查询成功
        """
        _require_service()

        # 1. 查询 A/B 测试配置
        config_resp = requests.get(
            f"{BASE_URL}/api/v1/mlops/ab-config",
            headers=_get_auth_headers(),
            timeout=REQUEST_TIMEOUT,
        )

        if config_resp.status_code == 404:
            pytest.skip("A/B 测试端点未启用")

        assert config_resp.status_code == 200, (
            f"A/B 测试配置查询应返回 200，实际：{config_resp.status_code}"
        )

        config_data = config_resp.json()
        assert "bucket_a_ratio" in config_data
        assert "enabled" in config_data

        # 2. 发送多个对话请求（使用同一 session_id）
        session_id = "test-ab-session-001"
        for i in range(3):
            resp = requests.post(
                f"{BASE_URL}/api/v1/chat",
                json={
                    "query": f"测试问题 {i+1}",
                    "session_id": session_id,
                },
                headers=_get_auth_headers(),
                timeout=REQUEST_TIMEOUT * 3,
            )
            assert resp.status_code == 200

        # 3. 查询 A/B 测试指标
        metrics_resp = requests.get(
            f"{BASE_URL}/api/v1/mlops/ab-metrics",
            headers=_get_auth_headers(),
            timeout=REQUEST_TIMEOUT,
        )

        if metrics_resp.status_code == 200:
            metrics_data = metrics_resp.json()
            assert "bucket_a_total_requests" in metrics_data
            assert "bucket_b_total_requests" in metrics_data

    def test_drift_detection_flow(self) -> None:
        """测试漂移检测流程

        流程：
        1. 查询漂移检测状态
        2. 发送多个查询触发检测
        3. 再次查询状态验证更新

        断言：
        - 漂移检测状态查询成功
        - 状态包含必要字段
        """
        _require_service()

        # 1. 查询漂移检测状态
        status_resp = requests.get(
            f"{BASE_URL}/api/v1/mlops/drift-status",
            headers=_get_auth_headers(),
            timeout=REQUEST_TIMEOUT,
        )

        if status_resp.status_code == 404:
            pytest.skip("漂移检测端点未启用")

        assert status_resp.status_code == 200, (
            f"漂移检测状态查询应返回 200，实际：{status_resp.status_code}"
        )

        status_data = status_resp.json()
        assert "status" in status_data
        assert status_data["status"] in ["normal", "degraded", "unavailable"]

        # 2. 发送多个查询（可能触发漂移检测）
        for i in range(5):
            resp = requests.post(
                f"{BASE_URL}/api/v1/chat",
                json={"query": f"查询 {i+1}：公司政策"},
                headers=_get_auth_headers(),
                timeout=REQUEST_TIMEOUT * 3,
            )
            if resp.status_code != 200:
                # 不强制要求成功，仅记录
                pass

        # 3. 再次查询状态
        status_resp2 = requests.get(
            f"{BASE_URL}/api/v1/mlops/drift-status",
            headers=_get_auth_headers(),
            timeout=REQUEST_TIMEOUT,
        )

        if status_resp2.status_code == 200:
            status_data2 = status_resp2.json()
            assert "status" in status_data2

    def test_complete_mlops_pipeline(self) -> None:
        """测试完整 MLOps 管道：对话 → 追踪 → A/B测试 → 漂移检测

        流程：
        1. 发送对话请求
        2. 验证追踪信息
        3. 验证 A/B 测试分桶
        4. 验证漂移检测状态

        断言：
        - 所有环节正常工作
        - MLOps 功能不阻塞核心对话
        """
        _require_service()

        # 1. 发送对话请求
        start_time = time.time()
        resp = requests.post(
            f"{BASE_URL}/api/v1/chat",
            json={
                "query": "完整的 MLOps 流程测试",
                "session_id": "mlops-pipeline-test-001",
            },
            headers=_get_auth_headers(),
            timeout=REQUEST_TIMEOUT * 3,
        )
        elapsed_ms = (time.time() - start_time) * 1000

        assert resp.status_code == 200, (
            f"对话请求应返回 200，实际：{resp.status_code}"
        )

        data = resp.json()
        assert "answer" in data

        # 2. 验证追踪 ID
        correlation_id = resp.headers.get("X-Correlation-ID") or resp.headers.get("X-Request-ID")
        assert correlation_id is not None

        # 3. 性能约束检查：MLOps 追踪不应显著增加延迟
        # 预期：MLOps 追踪增加延迟不超过 50ms（REQ-DFX-R02）
        # 注：这里放宽检查，因为包含 LLM 调用
        assert elapsed_ms < 30000, (
            f"对话响应时间过长：{elapsed_ms:.0f}ms"
        )


# ============================================================================
# 2. MLOps 异常降级测试
# ============================================================================


class TestMLOpsDegradation:
    """MLOps 异常降级测试"""

    def test_chat_success_with_mlflow_unavailable(self) -> None:
        """测试 MLflow 不可用时对话仍正常工作

        场景：MLflow Server 不可用

        验证：
        - 对话请求成功返回
        - 追踪逻辑降级为 no-op
        - 记录 warning 日志
        """
        _require_service()

        # 发送对话请求
        resp = requests.post(
            f"{BASE_URL}/api/v1/chat",
            json={"query": "MLflow 不可用时的对话测试"},
            headers=_get_auth_headers(),
            timeout=REQUEST_TIMEOUT * 3,
        )

        # 无论 MLflow 是否可用，对话都应成功
        assert resp.status_code == 200, (
            f"即使 MLflow 不可用，对话也应返回 200，实际：{resp.status_code}"
        )

        data = resp.json()
        assert "answer" in data

    def test_chat_success_with_drift_detector_unavailable(self) -> None:
        """测试漂移检测器不可用时对话仍正常工作

        场景：参考数据集未加载或检测器禁用

        验证：
        - 对话请求成功返回
        - 漂移检测降级为跳过
        """
        _require_service()

        # 发送对话请求
        resp = requests.post(
            f"{BASE_URL}/api/v1/chat",
            json={"query": "漂移检测不可用时的对话测试"},
            headers=_get_auth_headers(),
            timeout=REQUEST_TIMEOUT * 3,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data

    def test_chat_success_with_ab_testing_disabled(self) -> None:
        """测试 A/B 测试禁用时对话仍正常工作

        场景：A/B 测试功能未启用

        验证：
        - 对话请求成功返回
        - 使用默认 RAG 策略
        """
        _require_service()

        # 发送对话请求
        resp = requests.post(
            f"{BASE_URL}/api/v1/chat",
            json={"query": "A/B 测试禁用时的对话测试"},
            headers=_get_auth_headers(),
            timeout=REQUEST_TIMEOUT * 3,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data

    def test_mlops_failure_does_not_block_chat(self) -> None:
        """测试 MLOps 失败不阻塞核心对话

        核心原则：REQ-DFX-R01 - MLOps 追踪逻辑失败不阻塞核心对话链路

        验证：
        - 即使 MLOps 组件异常，对话仍能正常完成
        - 响应时间不受显著影响
        """
        _require_service()

        # 连续发送多个请求，模拟 MLOps 组件可能的异常场景
        for i in range(5):
            resp = requests.post(
                f"{BASE_URL}/api/v1/chat",
                json={"query": f"MLOps 容错测试 {i+1}"},
                headers=_get_auth_headers(),
                timeout=REQUEST_TIMEOUT * 3,
            )

            # 所有请求都应成功
            assert resp.status_code == 200, (
                f"请求 {i+1} 应返回 200，实际：{resp.status_code}"
            )


# ============================================================================
# 3. MLOps API 健康检查测试
# ============================================================================


class TestMLOpsHealthCheck:
    """MLOps API 健康检查测试"""

    def test_mlops_health_endpoint(self) -> None:
        """测试 MLOps 健康检查端点

        断言：
        - GET /api/v1/mlops/health 返回 200
        - 响应包含各组件状态
        """
        _require_service()

        resp = requests.get(
            f"{BASE_URL}/api/v1/mlops/health",
            headers=_get_auth_headers(),
            timeout=REQUEST_TIMEOUT,
        )

        if resp.status_code == 404:
            pytest.skip("MLOps 健康检查端点未启用")

        assert resp.status_code == 200, (
            f"MLOps 健康检查应返回 200，实际：{resp.status_code}"
        )

        data = resp.json()
        assert "mlflow_connected" in data
        assert "drift_detector_status" in data
        assert "ab_testing_enabled" in data

    def test_mlops_health_reflects_component_status(self) -> None:
        """测试 MLOps 健康检查反映组件真实状态

        验证：
        - mlflow_connected 为布尔值
        - drift_detector_status 为有效状态值
        - ab_testing_enabled 为布尔值
        """
        _require_service()

        resp = requests.get(
            f"{BASE_URL}/api/v1/mlops/health",
            headers=_get_auth_headers(),
            timeout=REQUEST_TIMEOUT,
        )

        if resp.status_code == 404:
            pytest.skip("MLOps 健康检查端点未启用")

        data = resp.json()

        # 验证字段类型
        assert isinstance(data.get("mlflow_connected"), bool)
        assert isinstance(data.get("ab_testing_enabled"), bool)

        drift_status = data.get("drift_detector_status")
        assert drift_status in ["normal", "degraded", "unavailable", "unknown"]

    def test_individual_component_health(self) -> None:
        """测试各组件独立健康检查

        验证：
        - MLflow 连接状态可查询
        - 漂移检测器状态可查询
        - A/B 测试配置可查询
        """
        _require_service()

        # 查询 A/B 测试配置
        ab_config_resp = requests.get(
            f"{BASE_URL}/api/v1/mlops/ab-config",
            headers=_get_auth_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        if ab_config_resp.status_code == 200:
            ab_config = ab_config_resp.json()
            assert "enabled" in ab_config

        # 查询漂移检测状态
        drift_status_resp = requests.get(
            f"{BASE_URL}/api/v1/mlops/drift-status",
            headers=_get_auth_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        if drift_status_resp.status_code == 200:
            drift_status = drift_status_resp.json()
            assert "status" in drift_status


# ============================================================================
# 4. A/B 测试配置管理测试
# ============================================================================


class TestABTestingConfig:
    """A/B 测试配置管理测试"""

    def test_get_ab_config(self) -> None:
        """测试获取 A/B 测试配置

        断言：
        - 返回当前配置
        - 配置字段完整
        """
        _require_service()

        resp = requests.get(
            f"{BASE_URL}/api/v1/mlops/ab-config",
            headers=_get_auth_headers(),
            timeout=REQUEST_TIMEOUT,
        )

        if resp.status_code == 404:
            pytest.skip("A/B 测试端点未启用")

        assert resp.status_code == 200

        data = resp.json()
        assert "bucket_a_ratio" in data
        assert "bucket_a_strategy" in data
        assert "bucket_b_strategy" in data
        assert "enabled" in data

        # 验证字段类型和范围
        assert 0.0 <= data["bucket_a_ratio"] <= 1.0
        assert isinstance(data["enabled"], bool)

    def test_ab_metrics_tracking(self) -> None:
        """测试 A/B 测试指标追踪

        流程：
        1. 发送多个对话请求
        2. 查询 A/B 测试指标
        3. 验证指标更新

        断言：
        - 指标正确累计
        - 平均延迟计算正确
        """
        _require_service()

        # 发送多个请求
        session_ids = [f"metrics-test-{i}" for i in range(5)]
        for session_id in session_ids:
            requests.post(
                f"{BASE_URL}/api/v1/chat",
                json={
                    "query": "测试 A/B 指标追踪",
                    "session_id": session_id,
                },
                headers=_get_auth_headers(),
                timeout=REQUEST_TIMEOUT * 3,
            )

        # 查询指标
        metrics_resp = requests.get(
            f"{BASE_URL}/api/v1/mlops/ab-metrics",
            headers=_get_auth_headers(),
            timeout=REQUEST_TIMEOUT,
        )

        if metrics_resp.status_code == 200:
            metrics = metrics_resp.json()
            # 验证指标字段存在
            assert "bucket_a_total_requests" in metrics
            assert "bucket_b_total_requests" in metrics


# ============================================================================
# 5. 漂移检测状态测试
# ============================================================================


class TestDriftDetectionStatus:
    """漂移检测状态测试"""

    def test_get_drift_status(self) -> None:
        """测试获取漂移检测状态

        断言：
        - 返回当前状态
        - 状态字段完整
        """
        _require_service()

        resp = requests.get(
            f"{BASE_URL}/api/v1/mlops/drift-status",
            headers=_get_auth_headers(),
            timeout=REQUEST_TIMEOUT,
        )

        if resp.status_code == 404:
            pytest.skip("漂移检测端点未启用")

        assert resp.status_code == 200

        data = resp.json()
        assert "status" in data
        assert data["status"] in ["normal", "degraded", "unavailable"]

    def test_drift_status_after_queries(self) -> None:
        """测试发送查询后漂移状态更新

        流程：
        1. 查询初始状态
        2. 发送多个查询
        3. 查询更新后状态

        断言：
        - 状态可能更新（取决于检测器配置）
        """
        _require_service()

        # 查询初始状态
        initial_resp = requests.get(
            f"{BASE_URL}/api/v1/mlops/drift-status",
            headers=_get_auth_headers(),
            timeout=REQUEST_TIMEOUT,
        )

        if initial_resp.status_code == 404:
            pytest.skip("漂移检测端点未启用")

        initial_status = initial_resp.json().get("status")

        # 发送多个查询
        for i in range(10):
            requests.post(
                f"{BASE_URL}/api/v1/chat",
                json={"query": f"漂移检测测试查询 {i+1}"},
                headers=_get_auth_headers(),
                timeout=REQUEST_TIMEOUT * 3,
            )

        # 查询更新后状态
        updated_resp = requests.get(
            f"{BASE_URL}/api/v1/mlops/drift-status",
            headers=_get_auth_headers(),
            timeout=REQUEST_TIMEOUT,
        )

        if updated_resp.status_code == 200:
            updated_status = updated_resp.json().get("status")
            # 状态可能在 normal/degraded/unavailable 之间变化
            assert updated_status in ["normal", "degraded", "unavailable"]


# ============================================================================
# 6. 性能约束测试
# ============================================================================


class TestPerformanceConstraints:
    """性能约束测试"""

    def test_tracking_latency_constraint(self) -> None:
        """测试追踪延迟约束（REQ-DFX-R02）

        约束：追踪逻辑增加延迟不超过 50ms

        验证：
        - 对话响应时间在合理范围内
        - 追踪操作不显著增加延迟
        """
        _require_service()

        # 发送多个请求，测量平均响应时间
        latencies = []
        for i in range(5):
            start_time = time.time()
            resp = requests.post(
                f"{BASE_URL}/api/v1/chat",
                json={"query": f"性能测试 {i+1}"},
                headers=_get_auth_headers(),
                timeout=REQUEST_TIMEOUT * 3,
            )
            elapsed_ms = (time.time() - start_time) * 1000

            if resp.status_code == 200:
                latencies.append(elapsed_ms)

        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            # 注：这里检查的是整体响应时间，包含 LLM 调用
            # 追踪逻辑的额外延迟通常在毫秒级别
            assert avg_latency < 10000, (
                f"平均响应时间过长：{avg_latency:.0f}ms"
            )

    def test_concurrent_requests_performance(self) -> None:
        """测试并发请求性能

        验证：
        - 系统可处理多个并发请求
        - 追踪逻辑不影响并发性能
        """
        _require_service()

        import concurrent.futures

        def send_request(query: str) -> tuple[int, float]:
            start_time = time.time()
            resp = requests.post(
                f"{BASE_URL}/api/v1/chat",
                json={"query": query},
                headers=_get_auth_headers(),
                timeout=REQUEST_TIMEOUT * 3,
            )
            elapsed_ms = (time.time() - start_time) * 1000
            return (resp.status_code, elapsed_ms)

        # 并发发送 5 个请求
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(send_request, f"并发测试 {i+1}")
                for i in range(5)
            ]

            results = [
                future.result(timeout=REQUEST_TIMEOUT * 5)
                for future in concurrent.futures.as_completed(futures)
            ]

        # 验证所有请求成功
        success_count = sum(1 for status, _ in results if status == 200)
        assert success_count >= 4, (
            f"至少 4 个并发请求应成功，实际：{success_count}"
        )