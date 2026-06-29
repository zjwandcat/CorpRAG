"""应用配置模块

集中管理应用程序的所有配置项，包括模型供应商配置、NVIDIA NIM 配置、
向量数据库路径、RAG 参数、限流策略、LLM 参数、LangGraph 配置、
Reranker 配置、Blueprint 配置、多Agent协作配置、MCP 配置和审查配置。

配置优先级：环境变量 > nim_config.txt 文件 > 默认值。
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from app.core.enums import ModelName, ModelProvider

_NIM_CONFIG_FILE: Final[Path] = Path(__file__).resolve().parent.parent.parent / "nim_config.txt"


def _load_config_from_file() -> dict[str, str]:
    config: dict[str, str] = {}
    if not _NIM_CONFIG_FILE.exists():
        return config

    content = _NIM_CONFIG_FILE.read_text(encoding="utf-8").strip()
    if not content:
        return config

    if "=" in content:
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            key, _, value = line.partition("=")
            config[key.strip()] = value.strip()
    else:
        config["api_key"] = content

    return config


def save_config_to_file(
    provider: str = "nim",
    api_key: str = "",
    model_name: str = "deepseek-ai/deepseek-v4-flash",
    xfyun_api_key: str = "",
    xfyun_model_name: str = "xopqwen36v35b",
    zhipu_api_key: str = "",
    zhipu_model_name: str = "glm-4.7-flash",
) -> None:
    """保存配置到文件，采用"读取-修改-写入"模式，按固定顺序写入所有字段。"""
    existing = _load_config_from_file()

    # 更新传入的字段
    existing["provider"] = provider.strip()
    if provider.strip() == ModelProvider.NIM:
        existing["api_key"] = api_key.strip()
        existing["model_name"] = model_name.strip()
    elif provider.strip() == ModelProvider.XFYUN:
        existing["xfyun_api_key"] = xfyun_api_key.strip()
        existing["xfyun_model_name"] = xfyun_model_name.strip()
    elif provider.strip() == ModelProvider.ZHIPU:
        existing["zhipu_api_key"] = zhipu_api_key.strip()
        existing["zhipu_model_name"] = zhipu_model_name.strip()

    # 确保所有字段存在
    if "xfyun_base_url" not in existing:
        existing["xfyun_base_url"] = "https://maas-api.cn-huabei-1.xf-yun.com/v2"
    if "zhipu_base_url" not in existing:
        existing["zhipu_base_url"] = "https://open.bigmodel.cn/api/paas/v4"

    # 按固定顺序写入
    xfyun_url = existing.get("xfyun_base_url", "https://maas-api.cn-huabei-1.xf-yun.com/v2")
    zhipu_url = existing.get("zhipu_base_url", "https://open.bigmodel.cn/api/paas/v4")
    lines = [
        f"provider={existing.get('provider', 'nim')}",
        f"api_key={existing.get('api_key', '')}",
        f"model_name={existing.get('model_name', 'deepseek-ai/deepseek-v4-flash')}",
        f"xfyun_api_key={existing.get('xfyun_api_key', '')}",
        f"xfyun_model_name={existing.get('xfyun_model_name', '4.0Ultra')}",
        f"xfyun_base_url={xfyun_url}",
        f"zhipu_api_key={existing.get('zhipu_api_key', '')}",
        f"zhipu_model_name={existing.get('zhipu_model_name', 'glm-4.7-flash')}",
        f"zhipu_base_url={zhipu_url}",
    ]
    content = "\n".join(lines)
    _NIM_CONFIG_FILE.write_text(content, encoding="utf-8")


_config_from_file: dict[str, str] = _load_config_from_file()


@dataclass(slots=True)
class Settings:
    # ---- Provider 配置 ----
    PROVIDER: str = os.getenv("PROVIDER", "") or _config_from_file.get(
        "provider", ModelProvider.NIM
    )
    XFYUN_API_KEY: str = os.getenv("XFYUN_API_KEY", "") or _config_from_file.get(
        "xfyun_api_key", ""
    )
    XFYUN_BASE_URL: str = os.getenv("XFYUN_BASE_URL", "") or _config_from_file.get(
        "xfyun_base_url", "https://maas-api.cn-huabei-1.xf-yun.com/v2"
    )
    XFYUN_MODEL_NAME: str = os.getenv("XFYUN_MODEL_NAME", "") or _config_from_file.get(
        "xfyun_model_name", ModelName.XFYUN_DEFAULT
    )
    ZHIPU_API_KEY: str = os.getenv("ZHIPU_API_KEY", "") or _config_from_file.get(
        "zhipu_api_key", ""
    )
    ZHIPU_BASE_URL: str = os.getenv("ZHIPU_BASE_URL", "") or _config_from_file.get(
        "zhipu_base_url", "https://open.bigmodel.cn/api/paas/v4"
    )
    ZHIPU_MODEL_NAME: str = os.getenv("ZHIPU_MODEL_NAME", "") or _config_from_file.get(
        "zhipu_model_name", ModelName.ZHIPU_DEFAULT
    )

    # ---- NVIDIA NIM 配置 ----
    NIM_BASE_URL: str = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    NIM_MODEL_NAME: str = os.getenv("NIM_MODEL_NAME", "") or _config_from_file.get(
        "model_name", ModelName.DEEPSEEK_V4
    )
    NIM_EMBEDDING_MODEL: str = os.getenv("NIM_EMBEDDING_MODEL", ModelName.NVIDIA_EMBED)
    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "") or _config_from_file.get("api_key", "")

    CHROMA_DB_DIR: Path = Path(os.getenv("CHROMA_DB_DIR", "./chroma_db"))
    KNOWLEDGE_DIR: Path = Path(os.getenv("KNOWLEDGE_DIR", "./data/reports"))

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    TOP_K: int = int(os.getenv("TOP_K", "3"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "1.3"))

    RATE_LIMIT_RPM: int = int(os.getenv("RATE_LIMIT_RPM", "29"))
    RETRY_MAX_ATTEMPTS: int = int(os.getenv("RETRY_MAX_ATTEMPTS", "5"))

    LLM_CACHE_TYPE: str = os.getenv("LLM_CACHE_TYPE", "memory")

    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "1"))
    LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", "0.95"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "16384"))
    LLM_REQUEST_TIMEOUT: int = int(os.getenv("LLM_REQUEST_TIMEOUT", "300"))

    MAX_TOOL_ROUNDS: int = int(os.getenv("MAX_TOOL_ROUNDS", "5"))

    SESSION_MAX_MESSAGES: int = int(os.getenv("SESSION_MAX_MESSAGES", "100"))

    # ---- LangGraph 配置 ----
    LANGGRAPH_MAX_ITERATIONS: int = int(os.getenv("LANGGRAPH_MAX_ITERATIONS", "5"))
    SSE_EVENT_QUEUE_TIMEOUT: int = int(os.getenv("SSE_EVENT_QUEUE_TIMEOUT", "60"))
    SSE_MAX_CONCURRENT_PER_THREAD: int = int(os.getenv("SSE_MAX_CONCURRENT_PER_THREAD", "1"))

    # ---- Reranker 配置 ----
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "nvidia/llama-nemotron-rerank-1b-v2")
    RERANKER_TIMEOUT: int = int(os.getenv("RERANKER_TIMEOUT", "10"))

    # ---- 智谱 Reranker 配置 ----
    ZHIPU_RERANKER_MODEL: str = os.getenv("ZHIPU_RERANKER_MODEL", "rerank")
    ZHIPU_RERANKER_TIMEOUT: int = int(os.getenv("ZHIPU_RERANKER_TIMEOUT", "10"))

    # ---- Blueprint 配置 ----
    BLUEPRINT_API_URL: str = os.getenv("BLUEPRINT_API_URL", "")
    BLUEPRINT_API_KEY: str = os.getenv("BLUEPRINT_API_KEY", "")
    BLUEPRINT_LLM_MODELNAME: str = os.getenv("BLUEPRINT_LLM_MODELNAME", "")
    BLUEPRINT_EMBEDDINGS_MODELNAME: str = os.getenv("BLUEPRINT_EMBEDDINGS_MODELNAME", "")
    BLUEPRINT_EMBEDDINGS_DIMENSIONS: int = int(os.getenv("BLUEPRINT_EMBEDDINGS_DIMENSIONS", "1024"))
    BLUEPRINT_TIMEOUT: int = int(os.getenv("BLUEPRINT_TIMEOUT", "30"))

    # ---- 多Agent协作配置 ----
    MULTI_AGENT_ENABLED: bool = os.getenv("MULTI_AGENT_ENABLED", "false").lower() in (
        "true",
        "1",
        "yes",
    )

    # ---- MCP 配置 ----
    MCP_ENABLED: bool = os.getenv("MCP_ENABLED", "false").lower() in ("true", "1", "yes")
    MCP_GITHUB_ENABLED: bool = os.getenv("MCP_GITHUB_ENABLED", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    MCP_FILESYSTEM_ENABLED: bool = os.getenv("MCP_FILESYSTEM_ENABLED", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    MCP_DATABASE_ENABLED: bool = os.getenv("MCP_DATABASE_ENABLED", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    MCP_WEBSEARCH_ENABLED: bool = os.getenv("MCP_WEBSEARCH_ENABLED", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    MCP_TOOL_CALL_TIMEOUT: int = int(os.getenv("MCP_TOOL_CALL_TIMEOUT", "30"))

    # ---- 审查配置 ----
    WORKER_TIMEOUT_SECONDS: int = int(os.getenv("WORKER_TIMEOUT_SECONDS", "60"))
    MAX_CONCURRENT_REVIEWS: int = int(os.getenv("MAX_CONCURRENT_REVIEWS", "10"))
    REVIEW_CONFIG_FILE: Path = Path(os.getenv("REVIEW_CONFIG_FILE", "./review_config.json"))

    # ---- 安全与合规配置 ----
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_PII_GUARD: bool = os.getenv("ENABLE_PII_GUARD", "true").lower() in ("true", "1", "yes")
    ENABLE_PROMPT_GUARD: bool = os.getenv("ENABLE_PROMPT_GUARD", "true").lower() in (
        "true",
        "1",
        "yes",
    )
    ENABLE_RATE_LIMIT: bool = os.getenv("ENABLE_RATE_LIMIT", "true").lower() in ("true", "1", "yes")
    # 测试用 API Key（仅开发环境使用，生产环境通过 K8s Secret 注入）
    TEST_API_KEY: str = os.getenv("TEST_API_KEY", "")

    # ---- KG 知识图谱配置 ----
    KG_ENABLED: bool = os.getenv("KG_ENABLED", "false").lower() in ("true", "1", "yes")
    KG_MAX_TRIPLETS_PER_DOC: int = int(os.getenv("KG_MAX_TRIPLETS_PER_DOC", "50"))
    KG_SEARCH_MAX_DEPTH: int = int(os.getenv("KG_SEARCH_MAX_DEPTH", "2"))
    KG_STORAGE_PATH: str = os.getenv("KG_STORAGE_PATH", "./kg_store")

    # ---- HITL Human-in-the-Loop 配置 ----
    HITL_ENABLED: bool = os.getenv("HITL_ENABLED", "false").lower() in ("true", "1", "yes")
    HITL_HIGH_RISK_TOOLS: str = os.getenv("HITL_HIGH_RISK_TOOLS", "send_email_notification")
    HITL_APPROVAL_TIMEOUT_SECONDS: int = int(os.getenv("HITL_APPROVAL_TIMEOUT_SECONDS", "300"))

    # ---- Guardrails 死循环防护配置 ----
    GUARDRAILS_ENABLED: bool = os.getenv("GUARDRAILS_ENABLED", "true").lower() in ("true", "1", "yes")
    GUARDRAILS_MAX_TOOL_REPETITION: int = int(os.getenv("GUARDRAILS_MAX_TOOL_REPETITION", "3"))
    GUARDRAILS_REPETITION_WINDOW: int = int(os.getenv("GUARDRAILS_REPETITION_WINDOW", "5"))

    # ---- MLflow 实验追踪配置 ----
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_ENABLED: bool = os.getenv("MLFLOW_ENABLED", "false").lower() in ("true", "1", "yes")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "financial-agent-rag")
    MLFLOW_REQUEST_TIMEOUT: int = int(os.getenv("MLFLOW_REQUEST_TIMEOUT", "5"))

    # ---- 漂移检测配置 ----
    DRIFT_ENABLED: bool = os.getenv("DRIFT_ENABLED", "true").lower() in ("true", "1", "yes")
    DRIFT_DETECTION_METHOD: str = os.getenv("DRIFT_DETECTION_METHOD", "ks_test")
    DRIFT_THRESHOLD: float = float(os.getenv("DRIFT_THRESHOLD", "0.05"))
    DRIFT_REFERENCE_DATASET_SIZE: int = int(os.getenv("DRIFT_REFERENCE_DATASET_SIZE", "100"))
    DRIFT_REFERENCE_EMBEDDINGS_PATH: str = os.getenv(
        "DRIFT_REFERENCE_EMBEDDINGS_PATH", "./drift_reference_embeddings.npy"
    )

    # ---- A/B 测试配置 ----
    AB_TESTING_ENABLED: bool = os.getenv("AB_TESTING_ENABLED", "false").lower() in ("true", "1", "yes")
    AB_BUCKET_A_RATIO: float = float(os.getenv("AB_BUCKET_A_RATIO", "0.5"))
    AB_BUCKET_A_STRATEGY: str = os.getenv("AB_BUCKET_A_STRATEGY", "self_hosted_rag")
    AB_BUCKET_B_STRATEGY: str = os.getenv("AB_BUCKET_B_STRATEGY", "blueprint_rag")

    # ---- 意图预测配置 ----
    INTENT_PREDICTION_ENABLED: bool = os.getenv("INTENT_PREDICTION_ENABLED", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    INTENT_MODEL_PATH: str = os.getenv("INTENT_MODEL_PATH", "./intent_model.pkl")

    # ---- 推荐配置 ----
    RECOMMENDATION_ENABLED: bool = os.getenv("RECOMMENDATION_ENABLED", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    RECOMMENDATION_TOP_K: int = int(os.getenv("RECOMMENDATION_TOP_K", "3"))


settings = Settings()

__all__ = ["Settings", "save_config_to_file", "settings"]
