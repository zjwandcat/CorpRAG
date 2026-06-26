"""
uvicorn 启动脚本（LlamaIndex 版本）。

双击运行或在命令行执行：
    python run.py

端口：8002（避免与 LangChain 版本的 8001 冲突）
"""

from typing import Final

import uvicorn

APP_MODULE: Final[str] = "app.main:app"
HOST: Final[str] = "0.0.0.0"
PORT: Final[int] = 8002
RELOAD: Final[bool] = False

__all__ = ["main"]


def main() -> None:
    uvicorn.run(
        APP_MODULE,
        host=HOST,
        port=PORT,
        reload=RELOAD,
    )


if __name__ == "__main__":
    main()
