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


__all__ = [
    "APIPath",
    "Department",
    "DocumentFormat",
    "ErrorCode",
    "ModelName",
    "ModelProvider",
    "RAGEngine",
    "ResultType",
    "SSEEventType",
    "ServiceStatus",
    "ToolName",
    "UserCommand",
]
