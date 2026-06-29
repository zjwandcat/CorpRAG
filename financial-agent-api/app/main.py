import asyncio
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
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
from app.api.routes_hitl import router as hitl_router
from app.api.routes_mlops import router as mlops_router
from app.api.routes_review import router as review_router
from app.core.config import settings
from app.core.dependencies import (
    _init_llm_cache,
    get_blueprint_client,
    get_vectorstore,
    update_api_key,
)
from app.core.enums import APIPath, ModelProvider, ServiceStatus
from app.core.limiter import limiter
from app.core.logging_config import get_logger, set_request_id
from app.exceptions import AgentError
from app.models.schemas import HealthResponse
from app.observability import PROMETHEUS_AVAILABLE
from app.observability.middleware import CorrelationIdMiddleware, MetricsMiddleware
from app.security import register_api_key

logger = get_logger(__name__)

# audit_logger 由 app.core.logging_config.setup_logging() 统一配置
audit_logger = logging.getLogger("audit")

__all__ = ["app"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    logger.info("=" * 60)
    logger.info("  企业内部办公知识库智能体 API 已启动")
    logger.info("  Provider：%s", settings.PROVIDER)
    _current_model = (
        settings.XFYUN_MODEL_NAME
        if settings.PROVIDER == ModelProvider.XFYUN
        else settings.ZHIPU_MODEL_NAME
        if settings.PROVIDER == ModelProvider.ZHIPU
        else settings.NIM_MODEL_NAME
    )
    logger.info("  模型：%s", _current_model)
    logger.info("  NIM 地址：%s", settings.NIM_BASE_URL)
    _embed_model = settings.NIM_EMBEDDING_MODEL
    logger.info("  Embedding 模型：%s", _embed_model)
    logger.info("  向量库目录：%s", settings.CHROMA_DB_DIR)
    logger.info("  知识库目录：%s", settings.KNOWLEDGE_DIR)
    logger.info("  LLM 缓存类型：%s", settings.LLM_CACHE_TYPE)
    _init_llm_cache()
    logger.info("  API 文档：http://localhost:8001/docs")
    logger.info("  前端界面：http://localhost:8001/")
    logger.info("=" * 60)

    # 注册测试 API Key（仅开发环境）
    if settings.TEST_API_KEY:
        register_api_key(settings.TEST_API_KEY, "admin", "test-key")
        logger.info("  测试 API Key 已注册（仅开发环境）")

    yield


app = FastAPI(
    title="企业内部办公知识库智能体 API",
    description=(
        "## 企业内部办公知识库智能体 API\n\n"
        "基于 LangGraph 状态机 + SSE 流式推送构建的企业内部办公知识库智能问答系统。\n\n"
        "### 快速开始\n\n"
        "| 接口 | 方法 | 路径 | 说明 |\n"
        "|------|------|------|------|\n"
        "| 流式对话 | POST | `/api/v1/chat/stream` | SSE 流式对话（实时状态推送） |\n"
        "| 同步对话 | POST | `/api/v1/chat` | 与 AI 助手对话 |\n"
        "| 上传文档 | POST | `/api/v1/docs/upload` | 上传 PDF/TXT 研报 |\n"
        "| 文档数量 | GET | `/api/v1/docs/count` | 查询向量库文档数 |\n"
        "| 清空文档 | DELETE | `/api/v1/docs/clear` | 清空向量库 |\n"
        "| 清除会话 | DELETE | `/api/v1/session/{id}` | 清除对话历史 |\n"
        "| 健康检查 | GET | `/health` | 服务状态探活 |\n\n"
        "### 技术栈\n\n"
        "- **Web 框架**：FastAPI（自动生成 OpenAPI 文档）\n"
        "- **LLM**：ChatNVIDIA / 讯飞星辰 / 智谱AI\n"
        "- **向量数据库**：ChromaDB\n"
        "- **Embedding**：NVIDIA nv-embedqa-e5-v5（云端 API）\n"
        "- **Agent 框架**：LangGraph 状态机 + SSE 流式推送\n\n"
        "---\n\n"
        "> 访问 [首页](/) 查看完整使用指引"
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Prometheus 指标暴露（优雅降级）
if PROMETHEUS_AVAILABLE:
    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        instrumentator = Instrumentator()
        instrumentator.instrument(app)
        instrumentator.expose(app, endpoint="/metrics", include_in_schema=False)
        logger.info("Prometheus 指标端点已启用：/metrics")
    except ImportError:
        logger.warning("prometheus_fastapi_instrumentator 未安装，跳过 Prometheus 指标暴露")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 可观测性中间件
app.add_middleware(MetricsMiddleware)
app.add_middleware(CorrelationIdMiddleware)


@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> Response:
    # 为每个请求生成唯一追踪 ID，便于跨模块日志关联
    request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:16])
    set_request_id(request_id)

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
            "request_id": request_id,
        }
    )

    # 将 request_id 写入响应头，方便前端/调用方追踪
    response.headers["X-Request-ID"] = request_id

    return response


app.add_middleware(SlowAPIMiddleware)

app.include_router(chat_router, prefix=APIPath.API_V1, tags=["对话"])
app.include_router(docs_router, prefix=APIPath.API_V1, tags=["文档管理"])
app.include_router(hitl_router, prefix=APIPath.API_V1, tags=["HITL审批"])
app.include_router(review_router, prefix=APIPath.API_V1, tags=["代码审查"])
app.include_router(mlops_router, prefix=f"{APIPath.API_V1}/mlops", tags=["MLOps"])


@app.get("/", include_in_schema=False)
async def homepage() -> FileResponse:
    return FileResponse("app/static/index.html")


@app.get(APIPath.HEALTH, response_model=HealthResponse, summary="健康检查")
async def health_check() -> HealthResponse:
    try:
        vectorstore = await asyncio.to_thread(get_vectorstore)
        count = vectorstore._collection.count()
    except AgentError as exc:
        logger.error("健康检查失败：%s", exc)
        raise HTTPException(status_code=500, detail=exc.message) from exc
    except Exception as exc:
        logger.error("健康检查失败：%s", exc)
        raise HTTPException(status_code=500, detail=f"健康检查失败：{exc!s}") from exc

    # 获取当前模型名称
    _current_model = (
        settings.XFYUN_MODEL_NAME
        if settings.PROVIDER == ModelProvider.XFYUN
        else settings.ZHIPU_MODEL_NAME
        if settings.PROVIDER == ModelProvider.ZHIPU
        else settings.NIM_MODEL_NAME
    )

    # 检查 Blueprint 可用性
    blueprint_available = False
    try:
        blueprint_client = get_blueprint_client()
        if blueprint_client.is_configured():
            blueprint_available = await blueprint_client.health_check()
    except Exception:
        pass

    return HealthResponse(
        status=ServiceStatus.OK,
        vectorstore_count=count,
        model_name=_current_model,
        blueprint_available=blueprint_available,
        acceleration_mode="cloud_api",
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

    # API Key 脱敏（三段式）
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


@app.post(f"{APIPath.API_V1}/config/test", summary="连通测试：验证 API Key 是否可用")
async def test_apikey(body: dict[str, str]) -> dict[str, Any]:
    api_key = body.get("api_key", "").strip()
    model_name = body.get("model_name", "").strip()
    provider = body.get("provider", "nim").strip().lower()

    if not api_key:
        raise HTTPException(status_code=400, detail="API Key 不能为空")
    if provider not in ("nim", "xfyun", "zhipu"):
        raise HTTPException(status_code=400, detail="不支持的 Provider 类型")

    import time as _time
    from langchain_core.messages import HumanMessage

    try:
        if provider == ModelProvider.ZHIPU:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=model_name or ModelName.ZHIPU_DEFAULT,
                base_url=settings.ZHIPU_BASE_URL,
                api_key=api_key,
                temperature=0,
                max_tokens=16,
                request_timeout=15,
                cache=False,
            )
        elif provider == ModelProvider.XFYUN:
            from app.rag.xfyun_hmac_llm import XfyunHmacChatModel

            llm = XfyunHmacChatModel(
                model=model_name or ModelName.XFYUN_DEFAULT,
                api_key=api_key,
                base_url=settings.XFYUN_BASE_URL,
                temperature=0,
                max_tokens=16,
                request_timeout=30,
            )
        else:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA

            llm = ChatNVIDIA(
                model=model_name or ModelName.DEEPSEEK_V4,
                base_url=settings.NIM_BASE_URL,
                api_key=api_key,
                temperature=0,
                max_tokens=16,
                model_kwargs={"request_timeout": 30},
            )

        start = _time.perf_counter()
        response = await llm.ainvoke([HumanMessage(content="hi")])
        latency_ms = round((_time.perf_counter() - start) * 1000)

        content = ""
        if hasattr(response, "content") and response.content:
            content = str(response.content)[:64]

        return {
            "success": True,
            "latency_ms": latency_ms,
            "model": model_name or (ModelName.ZHIPU_DEFAULT if provider == ModelProvider.ZHIPU else ModelName.XFYUN_DEFAULT if provider == ModelProvider.XFYUN else ModelName.DEEPSEEK_V4),
            "response_preview": content,
        }
    except Exception as exc:
        logger.warning("连通测试失败（provider=%s）：%s", provider, exc)
        error_msg = str(exc)
        if "429" in error_msg or "rate" in error_msg.lower() or "速率限制" in error_msg:
            return {
                "success": True,
                "latency_ms": 0,
                "model": model_name or (ModelName.ZHIPU_DEFAULT if provider == ModelProvider.ZHIPU else ModelName.XFYUN_DEFAULT if provider == ModelProvider.XFYUN else ModelName.DEEPSEEK_V4),
                "response_preview": "Key 有效，但当前触发速率限制，请稍后使用",
            }
        if "401" in error_msg or "Unauthorized" in error_msg or "authentication" in error_msg.lower():
            detail = "API Key 无效或已过期"
        elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            detail = "连接超时，请检查网络"
        else:
            detail = f"连接失败：{error_msg[:120]}"
        return {
            "success": False,
            "latency_ms": 0,
            "model": model_name or (ModelName.ZHIPU_DEFAULT if provider == ModelProvider.ZHIPU else ModelName.XFYUN_DEFAULT if provider == ModelProvider.XFYUN else ModelName.DEEPSEEK_V4),
            "error": detail,
        }


@app.post(f"{APIPath.API_V1}/config/apikey", summary="保存 API Key 和模型名称")
async def set_apikey(body: dict[str, str]) -> dict[str, str | bool]:
    api_key = body.get("api_key", "").strip()
    model_name = body.get("model_name", "").strip()
    provider = body.get("provider", "nim").strip().lower()

    if not api_key:
        raise HTTPException(status_code=400, detail="API Key 不能为空")
    if provider not in ("nim", "xfyun", "zhipu"):
        raise HTTPException(status_code=400, detail="不支持的 Provider 类型")

    try:
        update_api_key(api_key, model_name, provider)
        return {"message": "配置已保存", "configured": True}
    except AgentError as exc:
        raise HTTPException(status_code=500, detail=exc.message) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"保存配置失败：{exc!s}") from exc


app.mount("/static", StaticFiles(directory="app/static"), name="static")
