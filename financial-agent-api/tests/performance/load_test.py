"""API 压测脚本

对 Agent Platform API 发送并发请求，测量延迟统计。
使用方法：python tests/performance/load_test.py [api_key]
"""

import asyncio
import time
import httpx
import statistics
import sys

BASE_URL = "http://localhost:8001"
TOTAL = 50


async def send(client: httpx.AsyncClient, key: str) -> dict:
    """发送单个请求并记录结果"""
    headers = {"Authorization": f"Bearer {key}"}
    payload = {"query": "公司的报销流程是怎样的？", "session_id": f"lt-{int(time.time())}"}
    start = time.time()
    try:
        r = await client.post(f"{BASE_URL}/api/v1/chat", json=payload, headers=headers, timeout=120)
        return {
            "code": r.status_code,
            "elapsed": round(time.time() - start, 3),
            "ok": r.status_code == 200,
        }
    except Exception as e:
        return {"code": 0, "elapsed": round(time.time() - start, 3), "ok": False, "err": str(e)}


async def main(key: str) -> None:
    """执行压测"""
    print(f"Load Test: {TOTAL} requests to {BASE_URL}/api/v1/chat")
    async with httpx.AsyncClient() as c:
        results = await asyncio.gather(*[send(c, key) for _ in range(TOTAL)])
    lats = [r["elapsed"] for r in results if r["ok"]]
    ok = sum(1 for r in results if r["ok"])
    print(f"Success: {ok}/{TOTAL} ({ok / TOTAL * 100:.1f}%)")
    if lats:
        print(
            f"Min: {min(lats):.3f}s | Max: {max(lats):.3f}s | Mean: {statistics.mean(lats):.3f}s | P95: {sorted(lats)[int(len(lats) * 0.95)]:.3f}s"
        )


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "test-key"))
