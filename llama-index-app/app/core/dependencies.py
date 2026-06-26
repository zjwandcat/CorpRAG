import logging
import threading
import time


from llama_index.core import VectorStoreIndex
from llama_index.core.agent import AgentWorkflow
from llama_index.core.llms import LLM

from app.core.config import save_config_to_file, settings
from app.core.enums import ModelName, ModelProvider
from app.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class RateLimiter:
    """请求频率与并发控制器（LlamaIndex 版本，与 LangChain 侧保持一致）"""

    __slots__ = (
        "_lock",
        "_semaphore",
        "_timestamps",
        "max_rpm",
        "max_concurrent",
        "provider",
    )

    _PROVIDER_DEFAULTS: dict[str, dict[str, int]] = {
        "zhipu": {"max_rpm": 30, "max_concurrent": 3},
        "xfyun": {"max_rpm": 20, "max_concurrent": 2},
        "nim": {"max_rpm": 15, "max_concurrent": 5},
    }

    def __init__(
        self,
        max_rpm: int = 15,
        max_concurrent: int = 5,
        provider: str = "nim",
    ) -> None:
        self.max_rpm = max_rpm
        self.max_concurrent = max_concurrent
        self.provider = provider
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(max_concurrent)
        self._timestamps: list[float] = []

    @classmethod
    def from_provider(
        cls,
        provider: str,
        max_rpm_override: int | None = None,
    ) -> "RateLimiter":
        defaults = cls._PROVIDER_DEFAULTS.get(provider, cls._PROVIDER_DEFAULTS["nim"])
        rpm = max_rpm_override or defaults["max_rpm"]
        concurrent = defaults["max_concurrent"]
        return cls(max_rpm=rpm, max_concurrent=concurrent, provider=provider)

    def acquire(self) -> None:
        self._semaphore.acquire()

    def release(self) -> None:
        self._semaphore.release()

    def wait(self) -> None:
        wait_sec = 0.0
        with self._lock:
            now = time.monotonic()
            cutoff = now - 60.0
            self._timestamps = [t for t in self._timestamps if t > cutoff]
            if len(self._timestamps) >= self.max_rpm:
                oldest = self._timestamps[0]
                wait_sec = oldest + 60.0 - now + 0.1
            self._timestamps.append(max(now, now + wait_sec))
        if wait_sec > 0:
            logger.debug(
                "限速等待 %.1f 秒（%s RPM=%d）",
                wait_sec,
                self.provider,
                self.max_rpm,
            )
            time.sleep(wait_sec)


_rate_limiter_instance: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    global _rate_limiter_instance
    if _rate_limiter_instance is None:
        _rate_limiter_instance = RateLimiter.from_provider(
            provider=settings.PROVIDER,
            max_rpm_override=settings.RATE_LIMIT_RPM,
        )
        logger.info(
            "限速器初始化，Provider=%s, RPM=%d, 并发=%d",
            settings.PROVIDER,
            _rate_limiter_instance.max_rpm,
            _rate_limiter_instance.max_concurrent,
        )
    return _rate_limiter_instance


__all__ = [
    "get_agent",
    "get_index",
    "get_llm",
    "get_rate_limiter",
    "reset_singletons",
    "update_api_key",
]

_llm_instance: LLM | None = None
_index_instance: VectorStoreIndex | None = None
_agent_instance: AgentWorkflow | None = None


def _create_llm() -> LLM:
    if settings.PROVIDER == ModelProvider.XFYUN:
        from llama_index.llms.openai_like import OpenAILike

        return OpenAILike(
            model=settings.XFYUN_MODEL_NAME,
            api_key=settings.XFYUN_API_KEY,
            api_base=settings.XFYUN_BASE_URL,
            is_chat_model=True,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            timeout=settings.LLM_REQUEST_TIMEOUT,
        )
    if settings.PROVIDER == ModelProvider.ZHIPU:
        from llama_index.llms.openai_like import OpenAILike

        return OpenAILike(
            model=settings.ZHIPU_MODEL_NAME,
            api_key=settings.ZHIPU_API_KEY,
            api_base=settings.ZHIPU_BASE_URL,
            is_chat_model=True,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            timeout=settings.LLM_REQUEST_TIMEOUT,
        )
    # NVIDIA NIM：延迟导入，避免 zhipu/xfyun 下触发 API Key 警告
    from llama_index.llms.nvidia import NVIDIA

    return NVIDIA(
        model=settings.NIM_MODEL_NAME,
        api_key=settings.NVIDIA_API_KEY,
        base_url=settings.NIM_BASE_URL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        is_chat_model=True,
        is_function_calling_model=True,
        max_retries=8,  # 429 重试次数（默认3次太少）
        retry_delay=2.0,  # 重试间隔秒数（避免频繁触发限流）
        timeout=settings.LLM_REQUEST_TIMEOUT,
    )


def get_llm() -> LLM:
    global _llm_instance
    if _llm_instance is None:
        provider_label = (
            "讯飞星辰"
            if settings.PROVIDER == ModelProvider.XFYUN
            else "智谱AI"
            if settings.PROVIDER == ModelProvider.ZHIPU
            else "NVIDIA NIM"
        )
        logger.info("初始化 LlamaIndex LLM 单例（Provider：%s）", provider_label)
        _llm_instance = _create_llm()
        model_name = (
            settings.XFYUN_MODEL_NAME
            if settings.PROVIDER == ModelProvider.XFYUN
            else settings.ZHIPU_MODEL_NAME
            if settings.PROVIDER == ModelProvider.ZHIPU
            else settings.NIM_MODEL_NAME
        )
        logger.info("LLM 初始化完成，模型：%s", model_name)
    return _llm_instance


def get_index() -> VectorStoreIndex:
    global _index_instance
    if _index_instance is None:
        logger.info("初始化 VectorStoreIndex 单例")
        from app.rag.index_store import build_or_load_index

        _index_instance = build_or_load_index()
        logger.info("VectorStoreIndex 初始化完成，目录：%s", settings.CHROMA_DB_DIR)
    return _index_instance


def get_agent() -> AgentWorkflow:
    global _agent_instance
    if _agent_instance is None:
        logger.info("初始化 AgentWorkflow 单例")
        from app.agent.workflow import create_agent

        index = get_index()
        llm = get_llm()
        _agent_instance = create_agent(index=index, llm=llm)
        logger.info("AgentWorkflow 初始化完成")
    return _agent_instance


def reset_singletons() -> None:
    global _llm_instance, _index_instance, _agent_instance
    _llm_instance = None
    _index_instance = None
    _agent_instance = None
    logger.info("所有 LlamaIndex 单例已重置")


def update_api_key(
    api_key: str,
    model_name: str = ModelName.DEEPSEEK_V4,
    provider: str = ModelProvider.NIM,
) -> None:
    api_key = api_key.strip()
    if not api_key:
        raise ConfigurationError("API Key 不能为空")

    provider = provider.strip().lower()
    if provider not in (ModelProvider.NIM, ModelProvider.XFYUN, ModelProvider.ZHIPU):
        raise ConfigurationError(f"不支持的 Provider 类型：{provider}")

    model_name = model_name.strip()

    if provider == ModelProvider.XFYUN:
        settings.XFYUN_API_KEY = api_key
        settings.XFYUN_MODEL_NAME = model_name or ModelName.XFYUN_DEFAULT
        settings.PROVIDER = ModelProvider.XFYUN
        save_config_to_file(
            provider=ModelProvider.XFYUN,
            xfyun_api_key=api_key,
            xfyun_model_name=model_name or ModelName.XFYUN_DEFAULT,
        )
    elif provider == ModelProvider.ZHIPU:
        settings.ZHIPU_API_KEY = api_key
        settings.ZHIPU_MODEL_NAME = model_name or ModelName.ZHIPU_DEFAULT
        settings.PROVIDER = ModelProvider.ZHIPU
        save_config_to_file(
            provider=ModelProvider.ZHIPU,
            zhipu_api_key=api_key,
            zhipu_model_name=model_name or ModelName.ZHIPU_DEFAULT,
        )
    else:
        settings.NVIDIA_API_KEY = api_key
        settings.NIM_MODEL_NAME = model_name or ModelName.DEEPSEEK_V4
        settings.PROVIDER = ModelProvider.NIM
        save_config_to_file(
            provider=ModelProvider.NIM,
            api_key=api_key,
            model_name=model_name or ModelName.DEEPSEEK_V4,
        )

    reset_singletons()
    logger.info(
        "Provider 配置已更新（Provider：%s，模型：%s），LlamaIndex 单例已重置",
        provider,
        model_name,
    )
