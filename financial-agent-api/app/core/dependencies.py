import logging
import threading
from typing import Any, TYPE_CHECKING

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.globals import set_llm_cache
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from app.agent.chain import AgentChain
from app.agent.tools import (
    get_employee_info,
    make_generate_flowchart_code_tool,
    make_generate_html_prototype_tool,
    make_generate_prd_document_tool,
    make_predict_user_intent_tool,
    make_recommend_similar_documents_tool,
    make_search_internal_documents_tool,
    make_search_web_tool,
    send_email_notification,
)
from app.core.config import save_config_to_file, settings
from app.core.enums import ModelName, ModelProvider
from app.exceptions import ConfigurationError
from app.rag.vectorstore import build_or_load_vectorstore

if TYPE_CHECKING:
    from app.agent.review.supervisor import SupervisorAgent
    from app.mcp.client import MCPClient
    from app.mcp.registry import MCPRegistry
    from app.mlops.ab_testing import ABTestRouter
    from app.mlops.drift_detector import QueryDriftDetector
    from app.mlops.evaluator import RAGEvaluator
    from app.mlops.tracking import LLMExperimentTracker
    from app.review.features import FeatureFlags
    from app.review.settings import ReviewSettings

logger = logging.getLogger(__name__)

__all__ = [
    "get_ab_router",
    "get_agent_chain",
    "get_agent_graph",
    "get_blueprint_client",
    "get_compiled_graph",
    "get_drift_detector",
    "get_embeddings",
    "get_engine_router",
    "get_evaluator",
    "get_feature_flags",
    "get_guardrail_detector",
    "get_hitl_manager",
    "get_kg_manager",
    "get_llm_with_tools",
    "get_mcp_client",
    "get_mcp_registry",
    "get_memory_saver",
    "get_reranker",
    "get_review_settings",
    "get_supervisor",
    "get_tools_by_name",
    "get_tracker",
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

# 审查相关单例
_review_settings_instance: "ReviewSettings | None" = None
_feature_flags_instance: "FeatureFlags | None" = None
_mcp_registry_instance: "MCPRegistry | None" = None
_mcp_client_instance: "MCPClient | None" = None
_supervisor_instance: "SupervisorAgent | None" = None

# v5.1 新增单例
_kg_manager_instance: Any = None
_hitl_manager_instance: Any = None
_guardrail_detector_instance: Any = None

# MLOps 单例
_tracker_instance: "LLMExperimentTracker | None" = None
_drift_detector_instance: "QueryDriftDetector | None" = None
_evaluator_instance: "RAGEvaluator | None" = None
_ab_router_instance: "ABTestRouter | None" = None

# 单例初始化线程安全锁
_singleton_lock = threading.RLock()


def _init_llm_cache() -> None:
    """根据 settings.LLM_CACHE_TYPE 初始化语义缓存并注册到 LangChain 全局缓存。

    支持两种缓存后端：
    - "memory": InMemoryCache（默认，进程内缓存）
    - "sqlite": SQLiteCache（持久化到 .langchain_cache.db 文件）

    必须在 LLM 实例创建之前调用，以确保 cache=True 生效。
    """
    cache_type = settings.LLM_CACHE_TYPE.strip().lower()

    if cache_type == "sqlite":
        from langchain_community.cache import SQLiteCache

        cache_path = ".langchain_cache.db"
        cache = SQLiteCache(database_path=cache_path)
        set_llm_cache(cache)
        logger.info("LLM 语义缓存已初始化（SQLite），路径：%s", cache_path)
    else:
        from langchain_core.caches import InMemoryCache

        cache = InMemoryCache()
        set_llm_cache(cache)
        logger.info("LLM 语义缓存已初始化（InMemory）")


def _create_raw_llm() -> BaseChatModel:
    if settings.PROVIDER == ModelProvider.XFYUN:
        from app.rag.xfyun_hmac_llm import XfyunHmacChatModel

        return XfyunHmacChatModel(
            model=settings.XFYUN_MODEL_NAME,
            api_key=settings.XFYUN_API_KEY,
            base_url=settings.XFYUN_BASE_URL,
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
            cache=True,
        )
    return ChatNVIDIA(
        model=settings.NIM_MODEL_NAME,
        base_url=settings.NIM_BASE_URL,
        api_key=settings.NVIDIA_API_KEY,
        temperature=settings.LLM_TEMPERATURE,
        top_p=settings.LLM_TOP_P,
        max_tokens=settings.LLM_MAX_TOKENS,
        model_kwargs={"request_timeout": settings.LLM_REQUEST_TIMEOUT},
        cache=True,
    )


def _build_tools(vectorstore: Chroma, raw_llm: BaseChatModel) -> list[BaseTool]:
    reranker = get_reranker()  # 注入 Reranker
    tools: list[BaseTool] = [
        make_search_internal_documents_tool(vectorstore, reranker),
        get_employee_info,
        make_search_web_tool(),
        send_email_notification,
        make_generate_prd_document_tool(raw_llm),
        make_generate_flowchart_code_tool(raw_llm),
        make_generate_html_prototype_tool(raw_llm),
    ]

    # v5.1 新增：知识图谱检索工具（当 KG_ENABLED=True 时注册）
    if settings.KG_ENABLED:
        kg_manager = get_kg_manager()
        if kg_manager is not None:
            from app.agent.tools import make_search_knowledge_graph_tool

            tools.append(make_search_knowledge_graph_tool(kg_manager))
            logger.info("已注册 search_knowledge_graph 工具")

    # v5.0 新增：相似文档推荐工具（当 RECOMMENDATION_ENABLED=True 时注册）
    if settings.RECOMMENDATION_ENABLED:
        embeddings = get_embeddings()
        tools.append(make_recommend_similar_documents_tool(vectorstore, embeddings))
        logger.info("已注册 recommend_similar_documents 工具")

    # v5.0 新增：用户意图预测工具（当 INTENT_PREDICTION_ENABLED=True 时注册）
    if settings.INTENT_PREDICTION_ENABLED:
        tools.append(make_predict_user_intent_tool())
        logger.info("已注册 predict_user_intent 工具")

    return tools


def _build_tools_by_name(tools: list[BaseTool]) -> dict[str, BaseTool]:
    return {t.name: t for t in tools}


def get_embeddings() -> Embeddings:
    global _embeddings_instance

    if _embeddings_instance is None:
        with _singleton_lock:
            if _embeddings_instance is None:
                logger.info("初始化 Embeddings 单例")
                from app.rag.vectorstore import create_embeddings

                _embeddings_instance = create_embeddings()
                logger.info("Embeddings 初始化完成，模型：%s", settings.NIM_EMBEDDING_MODEL)

    return _embeddings_instance


def get_vectorstore() -> Chroma:
    global _vectorstore_instance

    if _vectorstore_instance is None:
        with _singleton_lock:
            if _vectorstore_instance is None:
                logger.info("初始化 VectorStore 单例")
                embeddings = get_embeddings()
                _vectorstore_instance = build_or_load_vectorstore(embeddings)
                logger.info("VectorStore 初始化完成，目录：%s", settings.CHROMA_DB_DIR)

    return _vectorstore_instance


def get_tools_by_name() -> dict[str, BaseTool]:
    global _tools_by_name_instance

    if _tools_by_name_instance is None:
        with _singleton_lock:
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
        with _singleton_lock:
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
        with _singleton_lock:
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
        with _singleton_lock:
            if _memory_saver_instance is None:
                from langgraph.checkpoint.memory import MemorySaver

                _memory_saver_instance = MemorySaver()
                logger.info("MemorySaver 初始化完成")

    return _memory_saver_instance


def get_compiled_graph() -> Any:
    """获取编译后的 LangGraph 状态图单例"""
    global _compiled_graph_instance

    if _compiled_graph_instance is None:
        with _singleton_lock:
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
        with _singleton_lock:
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
        with _singleton_lock:
            if _reranker_instance is None:
                from app.rag.reranker import Reranker

                if settings.PROVIDER in (ModelProvider.XFYUN, ModelProvider.ZHIPU):
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
        with _singleton_lock:
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
        with _singleton_lock:
            if _engine_router_instance is None:
                from app.rag.engine_router import EngineRouter

                agent_graph = get_agent_graph()
                blueprint_client = get_blueprint_client()
                _engine_router_instance = EngineRouter(agent_graph, blueprint_client)
                logger.info("EngineRouter 初始化完成")

    return _engine_router_instance


# =============================================================================
# 审查相关依赖注入
# =============================================================================


def get_review_settings() -> "ReviewSettings":
    """获取审查配置单例

    从 Settings（环境变量级别）初始化默认值，
    支持从 JSON 文件加载持久化配置。

    Returns:
        ReviewSettings 实例
    """
    global _review_settings_instance

    if _review_settings_instance is None:
        with _singleton_lock:
            if _review_settings_instance is None:
                from app.review.settings import ReviewSettings

                _review_settings_instance = ReviewSettings.from_json()
                logger.info("ReviewSettings 初始化完成")

    return _review_settings_instance


def get_feature_flags() -> "FeatureFlags":
    """获取功能开关管理器单例

    基于 ReviewSettings 创建，提供动态功能开关查询和更新能力。

    Returns:
        FeatureFlags 实例
    """
    global _feature_flags_instance

    if _feature_flags_instance is None:
        with _singleton_lock:
            if _feature_flags_instance is None:
                from app.review.features import FeatureFlags

                review_settings = get_review_settings()
                _feature_flags_instance = FeatureFlags(review_settings)
                logger.info("FeatureFlags 初始化完成")

    return _feature_flags_instance


def get_mcp_registry() -> "MCPRegistry":
    """获取 MCP 工具注册表单例

    维护工具名到 Server/Adapter 的映射，
    支持工具注册、发现和调用路由。

    Returns:
        MCPRegistry 实例
    """
    global _mcp_registry_instance

    if _mcp_registry_instance is None:
        with _singleton_lock:
            if _mcp_registry_instance is None:
                from app.mcp.registry import MCPRegistry

                _mcp_registry_instance = MCPRegistry()
                logger.info("MCPRegistry 初始化完成")

    return _mcp_registry_instance


def get_mcp_client() -> "MCPClient":
    """获取 MCP 客户端单例

    管理多 MCP Server 连接，提供工具发现、调用和健康检查能力。

    Returns:
        MCPClient 实例
    """
    global _mcp_client_instance

    if _mcp_client_instance is None:
        with _singleton_lock:
            if _mcp_client_instance is None:
                from app.mcp.client import MCPClient

                registry = get_mcp_registry()
                features = get_feature_flags()
                _mcp_client_instance = MCPClient(
                    registry=registry,
                    features=features,
                    tool_call_timeout=settings.MCP_TOOL_CALL_TIMEOUT,
                )
                logger.info("MCPClient 初始化完成")

    return _mcp_client_instance


def get_supervisor() -> "SupervisorAgent":
    """获取 Supervisor Agent 单例

    负责任务分发、Worker Agent 调度、结果汇总和最终输出。

    Returns:
        SupervisorAgent 实例
    """
    global _supervisor_instance

    if _supervisor_instance is None:
        with _singleton_lock:
            if _supervisor_instance is None:
                from app.agent.review.supervisor import SupervisorAgent

                llm = _create_raw_llm()
                review_settings = get_review_settings()
                _supervisor_instance = SupervisorAgent(
                    llm=llm,
                    worker_timeout_seconds=review_settings.worker_timeout_seconds,
                )
                logger.info("SupervisorAgent 初始化完成")

    return _supervisor_instance


# =============================================================================
# v5.1 新增依赖注入：KG / HITL / Guardrails
# =============================================================================


def get_kg_manager() -> Any:
    """获取 KnowledgeGraphManager 单例（当 KG_ENABLED=True 时创建）

    知识图谱管理器负责实体关系三元组的提取、检索和持久化。
    当 KG_ENABLED=False 时返回 None，不中断核心链路。

    Returns:
        KnowledgeGraphManager 实例或 None
    """
    global _kg_manager_instance

    if not settings.KG_ENABLED:
        return None

    if _kg_manager_instance is None:
        with _singleton_lock:
            if _kg_manager_instance is None:
                from app.rag.knowledge_graph import KnowledgeGraphManager

                _kg_manager_instance = KnowledgeGraphManager(
                    storage_path=settings.KG_STORAGE_PATH,
                    max_triplets_per_doc=settings.KG_MAX_TRIPLETS_PER_DOC,
                    search_max_depth=settings.KG_SEARCH_MAX_DEPTH,
                )
                logger.info("KnowledgeGraphManager 初始化完成")

    return _kg_manager_instance


def get_hitl_manager() -> Any:
    """获取 HITLManager 单例（当 HITL_ENABLED=True 时创建）

    Human-in-the-Loop 审批管理器，为高风险工具调用提供人工审批机制。
    当 HITL_ENABLED=False 时返回 None，不中断核心链路。

    Returns:
        HITLManager 实例或 None
    """
    global _hitl_manager_instance

    if not settings.HITL_ENABLED:
        return None

    if _hitl_manager_instance is None:
        with _singleton_lock:
            if _hitl_manager_instance is None:
                from app.agent.hitl_manager import HITLManager

                high_risk_tools = [
                    t.strip()
                    for t in settings.HITL_HIGH_RISK_TOOLS.split(",")
                    if t.strip()
                ]
                _hitl_manager_instance = HITLManager(
                    high_risk_tools=high_risk_tools,
                    approval_timeout=settings.HITL_APPROVAL_TIMEOUT_SECONDS,
                )
                logger.info("HITLManager 初始化完成")

    return _hitl_manager_instance


def get_guardrail_detector() -> Any:
    """获取 ToolRepetitionDetector 单例（当 GUARDRAILS_ENABLED=True 时创建）

    工具重复调用检测器，使用滑动窗口检测同一工具的连续重复调用，
    超过阈值时返回 block 动作。
    当 GUARDRAILS_ENABLED=False 时返回 None，不中断核心链路。

    Returns:
        ToolRepetitionDetector 实例或 None
    """
    global _guardrail_detector_instance

    if not settings.GUARDRAILS_ENABLED:
        return None

    if _guardrail_detector_instance is None:
        with _singleton_lock:
            if _guardrail_detector_instance is None:
                from app.security.guardrails import ToolRepetitionDetector

                _guardrail_detector_instance = ToolRepetitionDetector(
                    max_repetition=settings.GUARDRAILS_MAX_TOOL_REPETITION,
                    window_size=settings.GUARDRAILS_REPETITION_WINDOW,
                )
                logger.info("ToolRepetitionDetector 初始化完成")

    return _guardrail_detector_instance


# =============================================================================
# MLOps 依赖注入：Tracker / DriftDetector / Evaluator / ABTestRouter
# =============================================================================


def get_tracker() -> "LLMExperimentTracker":
    """获取 LLMExperimentTracker 单例（当 MLFLOW_ENABLED=True 时创建）

    LLM 实验追踪器，封装 MLflow 自动记录 RAG 链路参数与指标。
    当 MLFLOW_ENABLED=False 时返回禁用状态的 Tracker（所有操作为 no-op）。

    Returns:
        LLMExperimentTracker 实例
    """
    global _tracker_instance

    if _tracker_instance is None:
        with _singleton_lock:
            if _tracker_instance is None:
                from app.mlops.tracking import LLMExperimentTracker

                _tracker_instance = LLMExperimentTracker(
                    tracking_uri=settings.MLFLOW_TRACKING_URI,
                    experiment_name=settings.MLFLOW_EXPERIMENT_NAME,
                    enabled=settings.MLFLOW_ENABLED,
                    request_timeout=settings.MLFLOW_REQUEST_TIMEOUT,
                )
                logger.info(
                    "LLMExperimentTracker 初始化完成 | enabled=%s | uri=%s",
                    settings.MLFLOW_ENABLED,
                    settings.MLFLOW_TRACKING_URI,
                )

    return _tracker_instance


def get_drift_detector() -> "QueryDriftDetector":
    """获取 QueryDriftDetector 单例（当 DRIFT_ENABLED=True 时创建）

    查询漂移检测器，检测用户查询分布相对于参考数据集的偏移。
    当 DRIFT_ENABLED=False 时返回禁用状态的 Detector。

    Returns:
        QueryDriftDetector 实例
    """
    global _drift_detector_instance

    if _drift_detector_instance is None:
        with _singleton_lock:
            if _drift_detector_instance is None:
                from app.core.enums import DriftDetectionMethod
                from app.mlops.drift_detector import QueryDriftDetector

                # 解析检测方法
                method_str = settings.DRIFT_DETECTION_METHOD.lower()
                detection_method = (
                    DriftDetectionMethod.KS_TEST
                    if method_str == "ks_test"
                    else DriftDetectionMethod.MMD
                )

                _drift_detector_instance = QueryDriftDetector(
                    detection_method=detection_method,
                    drift_threshold=settings.DRIFT_THRESHOLD,
                    enabled=settings.DRIFT_ENABLED,
                    reference_dataset_size=settings.DRIFT_REFERENCE_DATASET_SIZE,
                )

                # 加载参考数据集
                if settings.DRIFT_ENABLED:
                    _drift_detector_instance.load_reference_dataset(
                        settings.DRIFT_REFERENCE_EMBEDDINGS_PATH
                    )

                logger.info(
                    "QueryDriftDetector 初始化完成 | enabled=%s | method=%s | threshold=%.4f",
                    settings.DRIFT_ENABLED,
                    detection_method,
                    settings.DRIFT_THRESHOLD,
                )

    return _drift_detector_instance


def get_evaluator() -> "RAGEvaluator":
    """获取 RAGEvaluator 单例

    RAG 评估器，使用 LLM-as-Judge 评估 RAG 输出质量。
    需要 LLM 实例作为 LLM-as-Judge。

    Returns:
        RAGEvaluator 实例
    """
    global _evaluator_instance

    if _evaluator_instance is None:
        with _singleton_lock:
            if _evaluator_instance is None:
                from app.mlops.evaluator import RAGEvaluator

                # 获取 LLM 实例作为 LLM-as-Judge
                llm_judge = _create_raw_llm()

                # 获取 Tracker 实例（用于记录评估结果到 MLflow）
                tracker = get_tracker()

                _evaluator_instance = RAGEvaluator(
                    tracker=tracker,
                    llm_judge=llm_judge,
                )
                logger.info("RAGEvaluator 初始化完成")

    return _evaluator_instance


def get_ab_router() -> "ABTestRouter":
    """获取 ABTestRouter 单例（当 AB_TESTING_ENABLED=True 时创建）

    A/B 测试路由器，基于 session_id 哈希分桶实现流量分发。
    当 AB_TESTING_ENABLED=False 时返回禁用状态的 Router。

    Returns:
        ABTestRouter 实例
    """
    global _ab_router_instance

    if _ab_router_instance is None:
        with _singleton_lock:
            if _ab_router_instance is None:
                from app.core.enums import RAGStrategy
                from app.mlops.ab_testing import ABTestRouter

                # 解析策略字符串
                bucket_a_strategy = RAGStrategy(settings.AB_BUCKET_A_STRATEGY)
                bucket_b_strategy = RAGStrategy(settings.AB_BUCKET_B_STRATEGY)

                _ab_router_instance = ABTestRouter(
                    bucket_a_ratio=settings.AB_BUCKET_A_RATIO,
                    bucket_a_strategy=bucket_a_strategy,
                    bucket_b_strategy=bucket_b_strategy,
                    enabled=settings.AB_TESTING_ENABLED,
                )
                logger.info(
                    "ABTestRouter 初始化完成 | enabled=%s | bucket_a_ratio=%.2f | "
                    "bucket_a_strategy=%s | bucket_b_strategy=%s",
                    settings.AB_TESTING_ENABLED,
                    settings.AB_BUCKET_A_RATIO,
                    bucket_a_strategy,
                    bucket_b_strategy,
                )

    return _ab_router_instance


def reset_singletons() -> None:
    with _singleton_lock:
        global _embeddings_instance, _vectorstore_instance
        global _llm_with_tools_instance, _agent_chain_instance, _tools_by_name_instance
        global _reranker_instance, _blueprint_client_instance, _engine_router_instance
        global _memory_saver_instance, _compiled_graph_instance, _agent_graph_instance
        global _review_settings_instance, _feature_flags_instance
        global _mcp_client_instance, _mcp_registry_instance, _supervisor_instance
        global _kg_manager_instance, _hitl_manager_instance, _guardrail_detector_instance
        global _tracker_instance, _drift_detector_instance, _evaluator_instance, _ab_router_instance

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
        _review_settings_instance = None
        _feature_flags_instance = None
        _mcp_client_instance = None
        _mcp_registry_instance = None
        _supervisor_instance = None
        _kg_manager_instance = None
        _hitl_manager_instance = None
        _guardrail_detector_instance = None
        _tracker_instance = None
        _drift_detector_instance = None
        _evaluator_instance = None
        _ab_router_instance = None

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
