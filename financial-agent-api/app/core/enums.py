"""枚举定义模块

定义应用程序中使用的所有枚举类型，包括模型供应商、模型名称、
工具名称、文档格式、审查类型、严重程度、MCP Server 名称等。
所有枚举均继承自 StrEnum，支持字符串比较和序列化。
"""

from enum import StrEnum


class ModelProvider(StrEnum):
    """模型供应商枚举"""

    NIM = "nim"
    XFYUN = "xfyun"
    ZHIPU = "zhipu"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"


class ModelName(StrEnum):
    """模型名称枚举"""

    NVIDIA_EMBED = "nvidia/nv-embedqa-e5-v5"
    DEEPSEEK_V4 = "deepseek-ai/deepseek-v4-flash"
    OPENAI_EMBED_SMALL = "text-embedding-3-small"
    XFYUN_DEFAULT = "xopqwen36v35b"
    ZHIPU_DEFAULT = "glm-4.7-flash"
    ZHIPU_RERANK = "rerank"
    HF_EMBED_ZH = "BAAI/bge-small-zh-v1.5"


class ToolName(StrEnum):
    """工具名称枚举"""

    SEARCH_INTERNAL_DOCS = "search_internal_documents"
    GET_EMPLOYEE_INFO = "get_employee_info"
    SEARCH_WEB = "search_web"
    SEND_EMAIL = "send_email_notification"
    GENERATE_PRD_DOCUMENT = "generate_prd_document"
    GENERATE_FLOWCHART_CODE = "generate_flowchart_code"
    GENERATE_HTML_PROTOTYPE = "generate_html_prototype"
    RECOMMEND_SIMILAR = "recommend_similar_documents"
    PREDICT_INTENT = "predict_user_intent"
    SEARCH_KNOWLEDGE_GRAPH = "search_knowledge_graph"


class DocumentFormat(StrEnum):
    """文档格式枚举"""

    PDF = ".pdf"
    TXT = ".txt"
    DOCX = ".docx"
    MD = ".md"


class Department(StrEnum):
    """部门枚举"""

    GENERAL = "通用"
    HR = "人事部"
    RD = "研发中心"
    FINANCE = "财务部"


class ServiceStatus(StrEnum):
    """服务状态枚举"""

    OK = "ok"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


class APIPath(StrEnum):
    """API 路径枚举"""

    API_V1 = "/api/v1"
    HEALTH = "/health"


class UserCommand(StrEnum):
    """用户命令枚举"""

    EXIT = "exit"
    QUIT = "quit"
    EXIT_CN = "退出"


class ErrorCode(StrEnum):
    """错误码枚举"""

    RATE_LIMIT = "429"


class RAGEngine(StrEnum):
    """RAG 引擎类型枚举"""

    BUILTIN = "builtin"  # 方案 A：自建 RAG + Reranker
    BLUEPRINT = "blueprint"  # 方案 B：NVIDIA RAG Blueprint


class ResultType(StrEnum):
    """结果类型枚举"""

    TEXT = "text"
    MARKDOWN = "markdown"
    MERMAID = "mermaid"
    HTML = "html"
    SEARCH_RESULTS = "search_results"


class SSEEventType(StrEnum):
    """SSE 事件类型枚举"""

    STREAM_START = "stream_start"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"
    RETRIEVAL_RESULT = "retrieval_result"
    RETRIEVAL_EMPTY = "retrieval_empty"
    AGENT_RETRY = "agent_retry"
    AGENT_FORCE_END = "agent_force_end"
    MEMORY_LOAD_FAILED = "memory_load_failed"
    STREAM_END = "stream_end"
    STREAM_ERROR = "stream_error"
    HITL_APPROVAL_REQUIRED = "hitl_approval_required"
    HITL_APPROVED = "hitl_approved"
    HITL_REJECTED = "hitl_rejected"
    GUARDRAIL_LOOP_DETECTED = "guardrail_loop_detected"


class ReviewType(StrEnum):
    """审查类型枚举"""

    FULL = "full"
    SECURITY = "security"
    ARCHITECTURE = "architecture"
    PERFORMANCE = "performance"
    STYLE = "style"


class ReviewStatus(StrEnum):
    """审查状态枚举"""

    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class Severity(StrEnum):
    """问题严重程度枚举"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class MCPServerName(StrEnum):
    """MCP Server 名称枚举"""

    GITHUB = "github"
    FILESYSTEM = "filesystem"
    DATABASE = "database"
    WEBSEARCH = "websearch"


class ReviewEventType(StrEnum):
    """审查 SSE 事件类型枚举"""

    REVIEW_START = "review_start"
    WORKER_START = "worker_start"
    WORKER_RESULT = "worker_result"
    WORKER_TIMEOUT = "worker_timeout"
    WORKER_ERROR = "worker_error"
    SUMMARY_START = "summary_start"
    REVIEW_END = "review_end"


class HITLStatus(StrEnum):
    """HITL 审批状态枚举"""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class GuardrailAction(StrEnum):
    """护栏动作枚举"""

    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"


class DriftDetectionMethod(StrEnum):
    """漂移检测方法枚举"""

    KS_TEST = "ks_test"
    MMD = "mmd"


class DriftStatus(StrEnum):
    """漂移检测状态枚举"""

    NORMAL = "normal"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class ABBucket(StrEnum):
    """A/B 测试分桶枚举"""

    BUCKET_A = "bucket_a"
    BUCKET_B = "bucket_b"


class RAGStrategy(StrEnum):
    """RAG 策略枚举"""

    SELF_HOSTED_RAG = "self_hosted_rag"
    BLUEPRINT_RAG = "blueprint_rag"


class IntentLabel(StrEnum):
    """意图标签枚举"""

    REIMBURSEMENT = "报销咨询"
    LEAVE_PROCESS = "请假流程"
    IT_SUPPORT = "IT支持"
    HR_MANAGEMENT = "人事管理"
    FINANCIAL_MANAGEMENT = "财务管理"
    OTHER = "其他"


class MLOpsEventType(StrEnum):
    """MLOps 事件类型枚举"""

    DRIFT_ALERT = "drift_alert"
    EVAL_COMPLETE = "eval_complete"
    AB_CONFIG_UPDATED = "ab_config_updated"


__all__ = [
    "ABBucket",
    "APIPath",
    "Department",
    "DocumentFormat",
    "DriftDetectionMethod",
    "DriftStatus",
    "ErrorCode",
    "GuardrailAction",
    "HITLStatus",
    "IntentLabel",
    "MCPServerName",
    "MLOpsEventType",
    "ModelName",
    "ModelProvider",
    "RAGEngine",
    "RAGStrategy",
    "ResultType",
    "ReviewEventType",
    "ReviewStatus",
    "ReviewType",
    "SSEEventType",
    "ServiceStatus",
    "Severity",
    "ToolName",
    "UserCommand",
]
