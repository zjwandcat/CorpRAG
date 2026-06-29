"""滑动窗口限流器模块

基于用户维度的滑动窗口速率限制器，使用 defaultdict(deque) 和线程锁实现。
超限时抛出 HTTP 429 并附带 Retry-After header。
"""

import logging
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


class SlidingWindowRateLimiter:
    """滑动窗口速率限制器。

    使用滑动时间窗口算法，按用户（key）维度跟踪请求频率。
    线程安全，适用于多线程/多协程环境。

    Attributes:
        _windows: 每个用户的请求时间戳队列。
        _lock: 线程锁，保证并发安全。
    """

    def __init__(self) -> None:
        """初始化限流器。"""
        self._windows: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window_seconds: int = 60) -> None:
        """检查指定 key 的请求是否在速率限制内。

        使用滑动窗口算法：移除窗口外的旧时间戳，然后判断当前请求数是否超限。

        Args:
            key: 限流维度标识，通常为用户哈希 Key 或客户端 IP。
            limit: 窗口内允许的最大请求数。
            window_seconds: 滑动窗口时长（秒），默认 60 秒。

        Raises:
            HTTPException: 当请求超过限制时抛出 429 状态码，
                并在 header 中附带 Retry-After 值。
        """
        now = time.monotonic()
        window_start = now - window_seconds

        with self._lock:
            # 清除窗口外的过期时间戳
            timestamps = self._windows[key]
            while timestamps and timestamps[0] <= window_start:
                timestamps.popleft()

            current_count = len(timestamps)

            if current_count >= limit:
                # 计算最早请求的过期时间，作为 Retry-After 值
                oldest = timestamps[0] if timestamps else now
                retry_after = int(oldest + window_seconds - now) + 1
                retry_after = max(retry_after, 1)

                logger.warning(
                    "速率限制触发: key=%s, count=%d, limit=%d, retry_after=%ds",
                    key[:8] + "...",
                    current_count,
                    limit,
                    retry_after,
                )

                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"请求频率超限，每 {window_seconds} 秒最多 {limit} 次请求",
                    headers={"Retry-After": str(retry_after)},
                )

            # 记录当前请求时间戳
            timestamps.append(now)

    def reset(self, key: str | None = None) -> None:
        """重置限流状态。

        Args:
            key: 指定用户的 key 进行重置；若为 None 则清空所有用户的限流记录。
        """
        with self._lock:
            if key is None:
                self._windows.clear()
            else:
                self._windows.pop(key, None)

    def get_current_count(self, key: str, window_seconds: int = 60) -> int:
        """获取指定用户在当前窗口内的请求数。

        Args:
            key: 用户标识。
            window_seconds: 窗口时长（秒）。

        Returns:
            当前窗口内的请求数量。
        """
        now = time.monotonic()
        window_start = now - window_seconds

        with self._lock:
            timestamps = self._windows[key]
            # 清除过期时间戳
            while timestamps and timestamps[0] <= window_start:
                timestamps.popleft()
            return len(timestamps)


# ---------------------------------------------------------------------------
# 模块级单例
# ---------------------------------------------------------------------------

user_rate_limiter = SlidingWindowRateLimiter()

__all__ = [
    "SlidingWindowRateLimiter",
    "user_rate_limiter",
]
