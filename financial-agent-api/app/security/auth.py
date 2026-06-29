"""API Key 身份认证模块

基于 API Key 的身份认证中间件，支持多角色权限控制。
提供 Key 注册、哈希、验证等核心能力，可作为 FastAPI 依赖注入使用。
"""

import hashlib
import logging
from typing import Any

from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 角色权限定义
# ---------------------------------------------------------------------------

ROLE_PERMISSIONS: dict[str, dict[str, Any]] = {
    "admin": {
        "endpoints": ["*"],
        "rate_limit_rpm": 100,
        "tools": ["*"],
        "review_config": True,
    },
    "developer": {
        "endpoints": [
            "/api/v1/chat",
            "/api/v1/chat/stream",
            "/api/v1/docs/*",
            "/api/v1/hardware/*",
            "/review/code",
        ],
        "rate_limit_rpm": 60,
        "tools": [
            "search_internal_documents",
            "search_web",
            "generate_prd_document",
            "generate_flowchart_code",
            "generate_html_prototype",
            "get_employee_info",
        ],
        "review_config": False,
    },
    "viewer": {
        "endpoints": ["/api/v1/chat", "/api/v1/docs/count", "/health"],
        "rate_limit_rpm": 20,
        "tools": ["search_internal_documents", "get_employee_info"],
        "review_config": False,
    },
}

# ---------------------------------------------------------------------------
# 内存 API Key 存储
# ---------------------------------------------------------------------------

API_KEY_STORE: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# 核心函数
# ---------------------------------------------------------------------------


def hash_api_key(raw_key: str) -> str:
    """对原始 API Key 进行 SHA-256 哈希。

    Args:
        raw_key: 原始 API Key 明文。

    Returns:
        SHA-256 哈希后的十六进制字符串。
    """
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def register_api_key(raw_key: str, role: str, name: str) -> None:
    """注册一个新的 API Key 到内存存储。

    Args:
        raw_key: 原始 API Key 明文，将自动进行哈希后存储。
        role: 角色名称，必须在 ROLE_PERMISSIONS 中定义。
        name: Key 的可读名称，用于标识用途。

    Raises:
        ValueError: 当角色不在 ROLE_PERMISSIONS 中时抛出。
    """
    if role not in ROLE_PERMISSIONS:
        raise ValueError(f"无效角色 '{role}'，允许的角色: {list(ROLE_PERMISSIONS.keys())}")

    hashed = hash_api_key(raw_key)
    API_KEY_STORE[hashed] = {
        "role": role,
        "name": name,
        "hashed_key": hashed,
    }
    logger.info("API Key 已注册: name=%s, role=%s", name, role)


async def authenticate(request: Request) -> dict[str, Any]:
    """FastAPI 依赖：从请求中提取并验证 API Key。

    支持两种 Header 格式：
    - ``Authorization: Bearer <api_key>``
    - ``X-API-Key: <api_key>``

    当 API_KEY_STORE 为空时（即未注册任何 API Key），自动跳过认证，
    返回匿名管理员身份，便于本地开发调试。

    Args:
        request: FastAPI 请求对象。

    Returns:
        验证通过后的用户信息字典，包含 role、name、hashed_key 等字段。

    Raises:
        HTTPException: 当 API Key 缺失或无效时返回 401。
    """
    # 如果没有注册任何 API Key，跳过认证（本地开发模式）
    if not API_KEY_STORE:
        return {"role": "admin", "name": "anonymous", "hashed_key": "anonymous"}

    raw_key: str | None = None

    # 优先从 Authorization: Bearer <key> 提取
    auth_header: str | None = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        raw_key = auth_header[7:].strip()

    # 回退到 X-API-Key header
    if raw_key is None:
        raw_key = request.headers.get("X-API-Key")

    if not raw_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少 API Key，请通过 Authorization: Bearer <key> 或 X-API-Key header 提供",
        )

    hashed = hash_api_key(raw_key)
    user_info = API_KEY_STORE.get(hashed)

    if user_info is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 API Key",
        )

    return user_info


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def has_endpoint_permission(user_info: dict[str, Any], endpoint: str) -> bool:
    """检查用户是否有权访问指定端点。

    Args:
        user_info: 由 authenticate() 返回的用户信息字典。
        endpoint: 请求的端点路径，如 ``/api/v1/chat``。

    Returns:
        有权限返回 True，否则返回 False。
    """
    role = user_info.get("role", "")
    permissions = ROLE_PERMISSIONS.get(role, {})
    allowed_endpoints: list[str] = permissions.get("endpoints", [])

    # 通配符表示全部端点
    if "*" in allowed_endpoints:
        return True

    for pattern in allowed_endpoints:
        if pattern.endswith("/*"):
            # 前缀匹配，如 /api/v1/docs/* 匹配 /api/v1/docs/upload
            prefix = pattern[:-1]  # 去掉末尾 *，保留 /
            if endpoint.startswith(prefix):
                return True
        elif endpoint == pattern:
            return True

    return False


def has_tool_permission(user_info: dict[str, Any], tool_name: str) -> bool:
    """检查用户是否有权使用指定工具。

    Args:
        user_info: 由 authenticate() 返回的用户信息字典。
        tool_name: 工具名称。

    Returns:
        有权限返回 True，否则返回 False。
    """
    role = user_info.get("role", "")
    permissions = ROLE_PERMISSIONS.get(role, {})
    allowed_tools: list[str] = permissions.get("tools", [])

    if "*" in allowed_tools:
        return True

    return tool_name in allowed_tools


def get_user_rate_limit(user_info: dict[str, Any]) -> int:
    """获取用户的速率限制（每分钟请求数）。

    Args:
        user_info: 由 authenticate() 返回的用户信息字典。

    Returns:
        每分钟允许的最大请求数。
    """
    role = user_info.get("role", "")
    permissions = ROLE_PERMISSIONS.get(role, {})
    return int(permissions.get("rate_limit_rpm", 20))


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

__all__ = [
    "API_KEY_STORE",
    "ROLE_PERMISSIONS",
    "authenticate",
    "get_user_rate_limit",
    "has_endpoint_permission",
    "has_tool_permission",
    "hash_api_key",
    "register_api_key",
]
