import logging

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.api.routes_chat import router as chat_router
from app.api.routes_docs import router as docs_router
from app.core.config import settings
from app.core.dependencies import get_index, update_api_key
from app.core.enums import APIPath, ModelProvider, ServiceStatus
from app.core.limiter import limiter
from app.exceptions import AgentError
from app.models.schemas import HealthResponse
from app.rag.index_store import configure_embed_model, get_index_count

logger = logging.getLogger(__name__)

audit_logger = logging.getLogger("audit")
audit_logger.setLevel(logging.INFO)
if not audit_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [AUDIT] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    audit_logger.addHandler(handler)

__all__ = ["app"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    provider_label = (
        "讯飞星辰"
        if settings.PROVIDER == ModelProvider.XFYUN
        else "智谱AI"
        if settings.PROVIDER == ModelProvider.ZHIPU
        else "NVIDIA NIM"
    )
    model_name = (
        settings.XFYUN_MODEL_NAME
        if settings.PROVIDER == ModelProvider.XFYUN
        else settings.ZHIPU_MODEL_NAME
        if settings.PROVIDER == ModelProvider.ZHIPU
        else settings.NIM_MODEL_NAME
    )
    logger.info("=" * 60)
    logger.info("  企业内部办公知识库智能体 API 已启动（LlamaIndex 版本）")
    logger.info("  Provider：%s", provider_label)
    logger.info("  模型：%s", model_name)
    logger.info("  NIM 地址：%s", settings.NIM_BASE_URL)
    _embed_model = (
        "BAAI/bge-small-zh-v1.5（本地降级）"
        if settings.PROVIDER in (ModelProvider.XFYUN, ModelProvider.ZHIPU)
        else settings.NIM_EMBEDDING_MODEL
    )
    logger.info("  Embedding 模型：%s", _embed_model)
    logger.info("  向量库目录：%s", settings.CHROMA_DB_DIR)
    logger.info("  知识库目录：%s", settings.KNOWLEDGE_DIR)
    logger.info("  CSV 数据目录：%s", settings.CSV_DIR)
    logger.info("  LLM 缓存类型：%s", settings.LLM_CACHE_TYPE)
    logger.info("  API 文档：http://localhost:8002/docs")
    logger.info("  前端界面：http://localhost:8002/")
    logger.info("=" * 60)

    # 在服务启动时配置 Embedding 模型
    try:
        configure_embed_model()
        logger.info("Embedding 模型配置成功")
    except Exception as exc:
        logger.error("Embedding 模型配置失败：%s", exc)

    yield


app = FastAPI(
    title="企业内部办公知识库智能体 API（LlamaIndex 版本）",
    description=(
        "## 企业内部办公知识库智能体 API（LlamaIndex 版本）\n\n"
        "基于 LlamaIndex + NVIDIA NIM 构建的企业内部办公知识库智能问答系统。\n\n"
        "### 快速开始\n\n"
        "| 接口 | 方法 | 路径 | 说明 |\n"
        "|------|------|------|------|\n"
        "| 对话 | POST | `/api/v1/chat` | 与 AI 助手对话 |\n"
        "| 上传文档 | POST | `/api/v1/docs/upload` | 上传 PDF/TXT/DOCX/MD/CSV |\n"
        "| 文档数量 | GET | `/api/v1/docs/count` | 查询向量库文档数 |\n"
        "| 清空文档 | DELETE | `/api/v1/docs/clear` | 清空向量库 |\n"
        "| 清除会话 | DELETE | `/api/v1/session/{id}` | 清除对话历史 |\n"
        "| 健康检查 | GET | `/health` | 服务状态探活 |\n\n"
        "### 技术栈\n\n"
        "- **Web 框架**：FastAPI（自动生成 OpenAPI 文档）\n"
        "- **LLM**：NVIDIA（DeepSeek V4 Pro）\n"
        "- **向量数据库**：ChromaDB（LlamaIndex 专用目录）\n"
        "- **Embedding**：NVIDIA nv-embedqa-e5-v5\n"
        "- **Agent 框架**：LlamaIndex ReActAgent\n"
        "- **RAG 优化**：NVIDIA Blueprints（QueryFusionRetriever + RRF）\n\n"
        "---\n\n"
        "> 访问 [首页](/) 查看完整使用指引"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> Response:
    start_time = time.time()
    response: Response = await call_next(request)
    process_time = time.time() - start_time

    audit_logger.info(
        {
            "path": request.url.path,
            "method": request.method,
            "client_ip": request.client.host if request.client else "unknown",
            "process_time_ms": round(process_time * 1000, 2),
            "status_code": response.status_code,
            "timestamp": datetime.now().isoformat(),
        }
    )

    return response


app.add_middleware(SlowAPIMiddleware)

app.include_router(chat_router, prefix=APIPath.API_V1, tags=["对话"])
app.include_router(docs_router, prefix=APIPath.API_V1, tags=["文档管理"])


# 静态文件目录（使用绝对路径，避免工作目录依赖）
_STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", include_in_schema=False)
async def homepage() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get(APIPath.HEALTH, response_model=HealthResponse, summary="健康检查")
async def health_check() -> HealthResponse:
    try:
        index = get_index()
        count = get_index_count(index)
    except AgentError as exc:
        logger.error("健康检查失败：%s", exc)
        raise HTTPException(status_code=500, detail=exc.message)
    except Exception as exc:
        logger.error("健康检查失败：%s", exc)
        raise HTTPException(status_code=500, detail=f"健康检查失败：{exc!s}")

    _current_model = (
        settings.XFYUN_MODEL_NAME
        if settings.PROVIDER == ModelProvider.XFYUN
        else settings.ZHIPU_MODEL_NAME
        if settings.PROVIDER == ModelProvider.ZHIPU
        else settings.NIM_MODEL_NAME
    )

    return HealthResponse(
        status=ServiceStatus.OK, vectorstore_count=count, model_name=_current_model
    )


@app.get(f"{APIPath.API_V1}/config/apikey", summary="查询 API Key 状态")
async def get_apikey_status() -> dict[str, bool | str]:
    provider = settings.PROVIDER
    if provider == ModelProvider.XFYUN:
        key = settings.XFYUN_API_KEY
        model = settings.XFYUN_MODEL_NAME
    elif provider == ModelProvider.ZHIPU:
        key = settings.ZHIPU_API_KEY
        model = settings.ZHIPU_MODEL_NAME
    else:
        key = settings.NVIDIA_API_KEY
        model = settings.NIM_MODEL_NAME

    if not key:
        return {
            "configured": False,
            "provider": provider,
            "hint": "",
            "model_name": model,
        }

    # API Key 脱敏
    if len(key) <= 4:
        masked = "****"
    elif len(key) <= 8:
        masked = key[:2] + "***"
    else:
        masked = key[:4] + "***" + key[-4:]

    return {
        "configured": True,
        "provider": provider,
        "hint": masked,
        "model_name": model,
    }


@app.post(f"{APIPath.API_V1}/config/apikey", summary="保存 API Key 和模型名称")
async def set_apikey(body: dict[str, str]) -> dict[str, str | bool]:
    api_key = body.get("api_key", "").strip()
    model_name = body.get("model_name", "").strip()
    provider = body.get("provider", ModelProvider.NIM).strip().lower()

    if not api_key:
        raise HTTPException(status_code=400, detail="API Key 不能为空")
    if provider not in (ModelProvider.NIM, ModelProvider.XFYUN, ModelProvider.ZHIPU):
        raise HTTPException(status_code=400, detail="不支持的 Provider 类型")

    try:
        update_api_key(api_key, model_name, provider)
        return {"message": "配置已保存", "configured": True}
    except AgentError as exc:
        raise HTTPException(status_code=500, detail=exc.message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"保存配置失败：{exc!s}")


app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
