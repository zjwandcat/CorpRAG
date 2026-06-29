"""安全与合规模块

提供 API Key 认证、RBAC 权限控制、按用户限流、
PII 检测与脱敏、Prompt Injection 防御和企业级审计日志。
"""

from app.security.auth import ROLE_PERMISSIONS, authenticate, hash_api_key, register_api_key
from app.security.rate_limiter import SlidingWindowRateLimiter, user_rate_limiter
from app.security.pii_guard import detect_pii, mask_pii, scan_llm_input, scan_llm_output
from app.security.prompt_guard import InjectionRisk, assess_injection_risk, sanitize_input
from app.security.audit import AuditEvent, log_audit_event

__all__ = [
    "ROLE_PERMISSIONS",
    "SlidingWindowRateLimiter",
    "AuditEvent",
    "InjectionRisk",
    "authenticate",
    "assess_injection_risk",
    "detect_pii",
    "hash_api_key",
    "log_audit_event",
    "mask_pii",
    "register_api_key",
    "sanitize_input",
    "scan_llm_input",
    "scan_llm_output",
    "user_rate_limiter",
]
