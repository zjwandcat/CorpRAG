"""
服务健康检查脚本。

用途：在宿主机上快速检查 Docker Compose 编排的三个服务是否正常运行。
检查对象：
  - FastAPI 应用（端口 8001）
  - NVIDIA NIM 大模型服务（端口 8002）
  - ChromaDB 向量数据库（端口 8003）

使用方式：
  python scripts/check_services.py
"""

import subprocess
import sys
from enum import StrEnum
from typing import Final

from app.exceptions import ServiceConnectionError

# ============================================================================
# 常量定义（使用 Final 标注，符合 B-12）
# ============================================================================

# API 服务健康检查 URL
API_HEALTH_URL: Final[str] = "http://localhost:8001/health"
# NVIDIA NIM 大模型服务模型列表 URL
NIM_MODELS_URL: Final[str] = "http://localhost:8002/v1/models"
# ChromaDB 向量数据库心跳检查 URL
EMBEDDING_HEARTBEAT_URL: Final[str] = "http://localhost:8003/api/v1/heartbeat"
# 默认请求超时时间（秒）
DEFAULT_TIMEOUT: Final[int] = 5


# ============================================================================
# 枚举定义（使用 StrEnum，符合 B-08）
# ============================================================================


class ServiceType(StrEnum):
    """服务类型枚举。

    定义支持的检查服务类型，每个服务对应一个健康检查端点。
    继承 StrEnum 使得枚举值可以直接作为字符串使用。
    """

    API = "api"  # FastAPI 应用服务
    NIM = "nim"  # NVIDIA NIM 大模型服务
    EMBEDDING = "embedding"  # ChromaDB 向量数据库服务


class ServiceName(StrEnum):
    """服务显示名称枚举。

    用于在终端输出中显示友好的服务名称。
    """

    API = "FastAPI 应用"
    NIM = "NVIDIA NIM"
    EMBEDDING = "ChromaDB"


# ============================================================================
# 公开 API 声明（符合 B-13）
# ============================================================================

__all__ = ["ServiceName", "ServiceType", "check_service", "main"]


# ============================================================================
# 核心函数
# ============================================================================


def _ensure_requests() -> None:
    """确保 requests 库可用，若未安装则自动 pip install。

    这是运行时依赖检查函数，确保脚本可以在没有预先安装 requests 的环境中运行。
    """
    try:
        import requests  # noqa: F401
    except ImportError:
        print("[⏳] requests 库未安装，正在自动安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
        print("[✅] requests 库安装完成\n")


def _get_service_url(service_type: ServiceType) -> str:
    """根据服务类型获取对应的健康检查 URL。

    使用 match/case 模式匹配（符合 B-06），提供清晰的服务类型到 URL 的映射。

    Args:
        service_type: 服务类型枚举值

    Returns:
        对应服务的健康检查 URL

    Raises:
        ValueError: 如果服务类型不支持
    """
    match service_type:
        case ServiceType.API:
            return API_HEALTH_URL
        case ServiceType.NIM:
            return NIM_MODELS_URL
        case ServiceType.EMBEDDING:
            return EMBEDDING_HEARTBEAT_URL
        case _:
            # 理论上不会到达这里，因为枚举已限定类型
            raise ValueError(f"不支持的服务类型: {service_type}")


def _get_service_name(service_type: ServiceType) -> str:
    """根据服务类型获取显示名称。

    使用 match/case 模式匹配（符合 B-06），提供清晰的服务类型到显示名称的映射。

    Args:
        service_type: 服务类型枚举值

    Returns:
        对应服务的显示名称
    """
    match service_type:
        case ServiceType.API:
            return ServiceName.API
        case ServiceType.NIM:
            return ServiceName.NIM
        case ServiceType.EMBEDDING:
            return ServiceName.EMBEDDING
        case _:
            # 理论上不会到达这里，因为枚举已限定类型
            return str(service_type)


def check_service(
    service_type: ServiceType,
    url: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> bool:
    """检查单个服务的健康状态。

    使用自定义异常层级（符合 B-14），将 requests 异常转换为 ServiceConnectionError。

    Args:
        service_type: 服务类型枚举值
        url: 健康检查 URL（可选，如果不提供则根据服务类型自动获取）
        timeout: 请求超时时间（秒）

    Returns:
        True 表示服务正常，False 表示服务异常
    """
    import requests

    # 如果未提供 URL，则根据服务类型自动获取
    if url is None:
        url = _get_service_url(service_type)

    # 获取服务显示名称
    name = _get_service_name(service_type)

    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            print(f"[✅] {name} 正常（{url}，状态码 {response.status_code}）")
            return True
        else:
            print(f"[❌] {name} 异常（{url}，状态码 {response.status_code}）")
            return False
    except requests.exceptions.ConnectionError:
        # 使用自定义异常层级（符合 B-14）
        error = ServiceConnectionError(
            message=f"{name} 无法连接",
            service_name=name,
            endpoint=url,
            details="服务可能未启动",
        )
        print(f"[❌] {error}")
        return False
    except requests.exceptions.Timeout:
        # 使用自定义异常层级（符合 B-14）
        error = ServiceConnectionError(
            message=f"{name} 超时",
            service_name=name,
            endpoint=url,
            details=f"{timeout} 秒内未响应",
        )
        print(f"[❌] {error}")
        return False
    except Exception as exc:
        print(f"[❌] {name} 检查出错（{url}，错误：{exc}）")
        return False


def main() -> None:
    """主函数：依次检查三个服务的健康状态。

    使用枚举定义服务类型（符合 B-08），提高代码可读性和类型安全性。
    """
    _ensure_requests()

    print("=" * 60)
    print("  企业内部办公知识库智能体 - 服务健康检查")
    print("=" * 60)
    print()

    # 定义要检查的三个服务（使用枚举，符合 B-08）
    services: list[ServiceType] = [
        ServiceType.API,
        ServiceType.NIM,
        ServiceType.EMBEDDING,
    ]

    results: list[tuple[str, bool]] = []
    for service_type in services:
        ok = check_service(service_type, timeout=DEFAULT_TIMEOUT)
        name = _get_service_name(service_type)
        results.append((name, ok))

    # 汇总结果
    print()
    print("-" * 60)
    all_ok = all(ok for _, ok in results)
    if all_ok:
        print("[✅] 所有服务运行正常！")
    else:
        failed = [name for name, ok in results if not ok]
        print(f"[❌] 以下服务异常：{', '.join(failed)}")
        print("    请检查：1) 容器是否启动  2) 端口是否正确  3) 健康检查是否通过")
    print("-" * 60)

    # 异常时以非零退出码退出，方便 CI/CD 集成
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
