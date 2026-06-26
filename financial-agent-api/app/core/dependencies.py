import logging
from typing import Any

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from app.agent.chain import AgentChain
from app.agent.tools import (
    get_employee_info,
    make_generate_flowchart_code_tool,
    make_generate_html_prototype_tool,
    make_generate_prd_document_tool,
    make_search_internal_documents_tool,
    make_search_web_tool,
    send_email_notification,
)
from app.core.config import save_config_to_file, settings
from app.core.enums import ModelName, ModelProvider
from app.exceptions import ConfigurationError
from app.rag.vectorstore import build_or_load_vectorstore

logger = logging.getLogger(__name__)

__all__ = [
    "get_agent_chain",
    "get_agent_graph",
    "get_blueprint_client",
    "get_compiled_graph",
    "get_embeddings",
    "get_engine_router",
    "get_llm_with_tools",
    "get_memory_saver",
    "get_reranker",
    "get_tools_by_name",
    "get_vectorstore",
    "reset_singletons",
    "update_api_key",
]

_embeddings_instance: Embeddings | None = None
_vectorstore_instance: Chroma | None = None
_llm_with_tools_instance: Any = None
_agent_chain_instance: AgentChain | None = None
_tools_by_name_instance: dict[str, BaseTool] | None = None
_reranker_instance: Any = None
_blueprint_client_instance: Any = None
_engine_router_instance: Any = None
_memory_saver_instance: Any = None
_compiled_graph_instance: Any = None
_agent_graph_instance: Any = None


def _create_raw_llm() -> BaseChatModel:
    if settings.PROVIDER == ModelProvider.XFYUN:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.XFYUN_MODEL_NAME,
            base_url=settings.XFYUN_BASE_URL,
            api_key=settings.XFYUN_API_KEY,
            temperature=settings.LLM_TEMPERATURE,
            top_p=settings.LLM_TOP_P,
            max_tokens=settings.LLM_MAX_TOKENS,
            request_timeout=settings.LLM_REQUEST_TIMEOUT,
        )
    if settings.PROVIDER == ModelProvider.ZHIPU:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.ZHIPU_MODEL_NAME,
            base_url=settings.ZHIPU_BASE_URL,
            api_key=settings.ZHIPU_API_KEY,
            temperature=settings.LLM_TEMPERATURE,
            top_p=settings.LLM_TOP_P,
            max_tokens=settings.LLM_MAX_TOKENS,
            request_timeout=settings.LLM_REQUEST_TIMEOUT,
        )
    return ChatNVIDIA(
        model=settings.NIM_MODEL_NAME,
        base_url=settings.NIM_BASE_URL,
        api_key=settings.NVIDIA_API_KEY,
        temperature=settings.LLM_TEMPERATURE,
        top_p=settings.LLM_TOP_P,
        max_tokens=settings.LLM_MAX_TOKENS,
        model_kwargs={"request_timeout": settings.LLM_REQUEST_TIMEOUT},
    )


def _build_tools(vectorstore: Chroma, raw_llm: BaseChatModel) -> list[BaseTool]:
    reranker = get_reranker()  # 注入 Reranker
    return [
        make_search_internal_documents_tool(vectorstore, reranker),
        get_employee_info,
        make_search_web_tool(),
        send_email_notification,
        make_generate_prd_document_tool(raw_llm),
        make_generate_flowchart_code_tool(raw_llm),
        make_generate_html_prototype_tool(raw_llm),
    ]


def _build_tools_by_name(tools: list[BaseTool]) -> dict[str, BaseTool]:
    return {t.name: t for t in tools}


def get_embeddings() -> Embeddings:
    global _embeddings_instance

    if _embeddings_instance is None:
        logger.info("初始化 Embeddings 单例")
        if settings.PROVIDER in (ModelProvider.XFYUN, ModelProvider.ZHIPU):
            logger.info(
                "使用本地 HuggingFace Embedding（降级模式），模型：%s", ModelName.HF_EMBED_ZH
            )
            import os

            # 国内环境使用 HuggingFace 镜像站，避免连接超时
            if not os.getenv("HF_ENDPOINT"):
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            from langchain_huggingface import HuggingFaceEmbeddings

            _embeddings_instance = HuggingFaceEmbeddings(
                model_name=ModelName.HF_EMBED_ZH,
            )
        else:
            from app.rag.vectorstore import create_embeddings

            _embeddings_instance = create_embeddings()
            logger.info("Embeddings 初始化完成，模型：%s", settings.NIM_EMBEDDING_MODEL)

    return _embeddings_instance


def get_vectorstore() -> Chroma:
    global _vectorstore_instance

    if _vectorstore_instance is None:
        logger.info("初始化 VectorStore 单例")
        embeddings = get_embeddings()
        _vectorstore_instance = build_or_load_vectorstore(embeddings)
        logger.info("VectorStore 初始化完成，目录：%s", settings.CHROMA_DB_DIR)

    return _vectorstore_instance


def get_tools_by_name() -> dict[str, BaseTool]:
    global _tools_by_name_instance

    if _tools_by_name_instance is None:
        vectorstore = get_vectorstore()
        raw_llm = _create_raw_llm()
        tools = _build_tools(vectorstore, raw_llm)
        _tools_by_name_instance = _build_tools_by_name(tools)
        logger.info("工具映射表已创建：%s", list(_tools_by_name_instance.keys()))

    return _tools_by_name_instance


def get_llm_with_tools() -> Any:
    global _llm_with_tools_instance

    if _llm_with_tools_instance is None:
        logger.info("初始化 LLM 实例")
        raw_llm = _create_raw_llm()
        _model_label = (
            settings.XFYUN_MODEL_NAME
            if settings.PROVIDER == ModelProvider.XFYUN
            else settings.ZHIPU_MODEL_NAME
            if settings.PROVIDER == ModelProvider.ZHIPU
            else settings.NIM_MODEL_NAME
        )
        logger.info("LLM 初始化完成，模型：%s", _model_label)

        vectorstore = get_vectorstore()
        tools = _build_tools(vectorstore, raw_llm)
        _llm_with_tools_instance = raw_llm.bind_tools(tools)
        logger.info("已绑定 %d 个工具到 LLM", len(tools))

    return _llm_with_tools_instance


def get_agent_chain() -> AgentChain:
    global _agent_chain_instance

    if _agent_chain_instance is None:
        logger.info("初始化 AgentChain 单例")
        llm_with_tools = get_llm_with_tools()
        tools_by_name = get_tools_by_name()
        _agent_chain_instance = AgentChain(
            llm_with_tools=llm_with_tools, tools_by_name=tools_by_name
        )
        logger.info("AgentChain 初始化完成")

    return _agent_chain_instance


def get_memory_saver() -> Any:
    """获取 MemorySaver 单例"""
    global _memory_saver_instance

    if _memory_saver_instance is None:
        from langgraph.checkpoint.memory import MemorySaver

        _memory_saver_instance = MemorySaver()
        logger.info("MemorySaver 初始化完成")

    return _memory_saver_instance


def get_compiled_graph() -> Any:
    """获取编译后的 LangGraph 状态图单例"""
    global _compiled_graph_instance

    if _compiled_graph_instance is None:
        from app.agent.graph import build_agent_graph

        llm_with_tools = get_llm_with_tools()
        tools_by_name = get_tools_by_name()
        vectorstore = get_vectorstore()
        reranker = get_reranker()
        memory_saver = get_memory_saver()
        _compiled_graph_instance = build_agent_graph(
            llm_with_tools=llm_with_tools,
            tools_by_name=tools_by_name,
            vectorstore=vectorstore,
            reranker=reranker,
            memory_saver=memory_saver,
        )
        logger.info("LangGraph 状态图编译完成")

    return _compiled_graph_instance


def get_agent_graph() -> Any:
    """获取 AgentGraph 执行器单例（替代 get_agent_chain）"""
    global _agent_graph_instance

    if _agent_graph_instance is None:
        from app.agent.graph import AgentGraph

        compiled_graph = get_compiled_graph()
        memory_saver = get_memory_saver()
        _agent_graph_instance = AgentGraph(compiled_graph, memory_saver)
        logger.info("AgentGraph 初始化完成")

    return _agent_graph_instance


def get_reranker() -> Any:
    """获取 Reranker 单例"""
    global _reranker_instance

    if _reranker_instance is None:
        from app.rag.reranker import Reranker

        if settings.PROVIDER in (ModelProvider.XFYUN, ModelProvider.ZHIPU):
            # zhipu/xfyun 下使用智谱 Reranker（如果可用）或降级
            # 完全不导入 NVIDIARerank，避免 API Key 警告
            if settings.PROVIDER == ModelProvider.ZHIPU and settings.ZHIPU_API_KEY:
                try:
                    _reranker_instance = Reranker.create_for_provider(
                        provider="zhipu",
                        api_key=settings.ZHIPU_API_KEY,
                        base_url=settings.ZHIPU_BASE_URL,
                    )
                    logger.info("Reranker 初始化完成（智谱AI）")
                except Exception as exc:
                    logger.warning("智谱 Reranker 初始化失败，使用降级模式：%s", exc)
                    _reranker_instance = Reranker.create_noop()
            else:
                logger.warning(
                    "当前 Provider（%s）下 Reranker 不可用（降级模式）", settings.PROVIDER
                )
                _reranker_instance = Reranker.create_noop()
        else:
            _reranker_instance = Reranker(
                model=settings.RERANKER_MODEL,
                api_key=settings.NVIDIA_API_KEY,
                base_url=settings.NIM_BASE_URL,
                timeout=settings.RERANKER_TIMEOUT,
            )
            logger.info("Reranker 初始化完成，模型：%s", settings.RERANKER_MODEL)

    return _reranker_instance


def get_blueprint_client() -> Any:
    """获取 BlueprintClient 单例"""
    global _blueprint_client_instance

    if _blueprint_client_instance is None:
        from app.rag.nvidia_blueprint_client import BlueprintClient

        _blueprint_client_instance = BlueprintClient(settings)
        logger.info(
            "BlueprintClient 初始化完成，API URL：%s",
            settings.BLUEPRINT_API_URL or "(未配置)",
        )

    return _blueprint_client_instance


def get_engine_router() -> Any:
    """获取 EngineRouter 单例"""
    global _engine_router_instance

    if _engine_router_instance is None:
        from app.rag.engine_router import EngineRouter

        agent_graph = get_agent_graph()
        blueprint_client = get_blueprint_client()
        _engine_router_instance = EngineRouter(agent_graph, blueprint_client)
        logger.info("EngineRouter 初始化完成")

    return _engine_router_instance


def reset_singletons() -> None:
    global _embeddings_instance, _vectorstore_instance
    global _llm_with_tools_instance, _agent_chain_instance, _tools_by_name_instance
    global _reranker_instance, _blueprint_client_instance, _engine_router_instance
    global _memory_saver_instance, _compiled_graph_instance, _agent_graph_instance

    _embeddings_instance = None
    _vectorstore_instance = None
    _llm_with_tools_instance = None
    _agent_chain_instance = None
    _tools_by_name_instance = None
    _reranker_instance = None
    _blueprint_client_instance = None
    _engine_router_instance = None
    _memory_saver_instance = None
    _compiled_graph_instance = None
    _agent_graph_instance = None

    logger.info("所有单例已重置")


def update_api_key(
    api_key: str,
    model_name: str = "deepseek-ai/deepseek-v4-flash",
    provider: str = "nim",
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
    logger.info("Provider 配置已更新（Provider：%s，模型：%s），单例已重置", provider, model_name)
