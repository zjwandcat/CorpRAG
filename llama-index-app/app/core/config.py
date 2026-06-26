import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from app.core.enums import ModelName, ModelProvider

# 指向共享的 nim_config.txt
_NIM_CONFIG_FILE: Final[Path] = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "financial-agent-api"
    / "nim_config.txt"
)


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
    provider: str = ModelProvider.NIM,
    api_key: str = "",
    model_name: str = ModelName.DEEPSEEK_V4,
    xfyun_api_key: str = "",
    xfyun_model_name: str = ModelName.XFYUN_DEFAULT,
    zhipu_api_key: str = "",
    zhipu_model_name: str = ModelName.ZHIPU_DEFAULT,
) -> None:
    """将多 Provider 配置写入共享配置文件
    （读取-修改-写入模式，确保不丢失其他 Provider 配置）。"""
    # 读取已有配置
    existing = _load_config_from_file()

    # 更新传入的字段
    existing["provider"] = provider.strip()
    if provider.strip() == ModelProvider.XFYUN:
        existing["xfyun_api_key"] = xfyun_api_key.strip()
        existing["xfyun_model_name"] = xfyun_model_name.strip()
    elif provider.strip() == ModelProvider.ZHIPU:
        existing["zhipu_api_key"] = zhipu_api_key.strip()
        existing["zhipu_model_name"] = zhipu_model_name.strip()
    else:
        existing["api_key"] = api_key.strip()
        existing["model_name"] = model_name.strip()

    # 确保 xfyun_base_url 字段存在
    if "xfyun_base_url" not in existing:
        existing["xfyun_base_url"] = "https://maas-api.cn-huabei-1.xf-yun.com/v2"
    # 确保 zhipu_base_url 字段存在
    if "zhipu_base_url" not in existing:
        existing["zhipu_base_url"] = "https://open.bigmodel.cn/api/paas/v4"

    # 按固定顺序写入
    lines = [
        f"provider={existing.get('provider', ModelProvider.NIM)}",
        f"api_key={existing.get('api_key', '')}",
        f"model_name={existing.get('model_name', ModelName.DEEPSEEK_V4)}",
        f"xfyun_api_key={existing.get('xfyun_api_key', '')}",
        f"xfyun_model_name={existing.get('xfyun_model_name', ModelName.XFYUN_DEFAULT)}",
        f"xfyun_base_url={existing.get('xfyun_base_url', 'https://maas-api.cn-huabei-1.xf-yun.com/v2')}",
        f"zhipu_api_key={existing.get('zhipu_api_key', '')}",
        f"zhipu_model_name={existing.get('zhipu_model_name', ModelName.ZHIPU_DEFAULT)}",
        f"zhipu_base_url={existing.get('zhipu_base_url', 'https://open.bigmodel.cn/api/paas/v4')}",
    ]
    content = "\n".join(lines)
    _NIM_CONFIG_FILE.write_text(content, encoding="utf-8")


_config_from_file: dict[str, str] = _load_config_from_file()


@dataclass(slots=True)
class Settings:
    NIM_BASE_URL: str = os.getenv("NIM_BASE_URL", "") or _config_from_file.get(
        "NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"
    )
    NIM_MODEL_NAME: str = os.getenv("NIM_MODEL_NAME", "") or _config_from_file.get(
        "model_name", ModelName.DEEPSEEK_V4
    )
    NIM_EMBEDDING_MODEL: str = os.getenv(
        "NIM_EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5"
    )
    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "") or _config_from_file.get(
        "api_key", ""
    )

    # ---- 讯飞星辰配置 ----
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

    # ---- 智谱AI配置 ----
    ZHIPU_API_KEY: str = os.getenv("ZHIPU_API_KEY", "") or _config_from_file.get(
        "zhipu_api_key", ""
    )
    ZHIPU_BASE_URL: str = os.getenv("ZHIPU_BASE_URL", "") or _config_from_file.get(
        "zhipu_base_url", "https://open.bigmodel.cn/api/paas/v4"
    )
    ZHIPU_MODEL_NAME: str = os.getenv("ZHIPU_MODEL_NAME", "") or _config_from_file.get(
        "zhipu_model_name", ModelName.ZHIPU_DEFAULT
    )

    # LlamaIndex 专用向量库目录
    CHROMA_DB_DIR: Path = Path(os.getenv("CHROMA_DB_DIR", "./chroma_db_li"))
    # 指向共享的知识库目录
    _SHARED_DATA_ROOT = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "financial-agent-api"
        / "data"
    )
    KNOWLEDGE_DIR: Path = Path(
        os.getenv(
            "KNOWLEDGE_DIR",
            str(_SHARED_DATA_ROOT / "knowledge_base"),
        )
    )
    # CSV 数据目录
    CSV_DIR: Path = Path(os.getenv("CSV_DIR", str(_SHARED_DATA_ROOT / "csv")))
    # 研报目录
    REPORTS_DIR: Path = Path(
        os.getenv("REPORTS_DIR", str(_SHARED_DATA_ROOT / "reports"))
    )

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

    MAX_TOOL_ROUNDS: int = int(os.getenv("MAX_TOOL_ROUNDS", "10"))

    SESSION_MAX_MESSAGES: int = int(os.getenv("SESSION_MAX_MESSAGES", "100"))


settings = Settings()

__all__ = ["Settings", "save_config_to_file", "settings"]
