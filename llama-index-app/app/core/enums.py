from enum import StrEnum


class ModelProvider(StrEnum):
    NIM = "nim"
    XFYUN = "xfyun"
    ZHIPU = "zhipu"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"


class ModelName(StrEnum):
    NVIDIA_EMBED = "nvidia/nv-embedqa-e5-v5"
    DEEPSEEK_V4 = "deepseek-ai/deepseek-v4-flash"
    OPENAI_EMBED_SMALL = "text-embedding-3-small"
    XFYUN_DEFAULT = "xopqwen36v35b"
    ZHIPU_DEFAULT = "glm-4.7-flash"
    HF_EMBED_ZH = "BAAI/bge-small-zh-v1.5"


class ToolName(StrEnum):
    SEARCH_INTERNAL_DOCS = "search_internal_documents"
    GET_EMPLOYEE_INFO = "get_employee_info"
    SEARCH_WEB = "search_web"
    SEND_EMAIL = "send_email_notification"
    GENERATE_PRD_DOCUMENT = "generate_prd_document"
    GENERATE_FLOWCHART_CODE = "generate_flowchart_code"
    GENERATE_HTML_PROTOTYPE = "generate_html_prototype"
    SEARCH_CSV_DATA = "search_csv_data"


class DocumentFormat(StrEnum):
    PDF = ".pdf"
    TXT = ".txt"
    DOCX = ".docx"
    MD = ".md"
    CSV = ".csv"


class Department(StrEnum):
    GENERAL = "通用"
    HR = "人事部"
    RD = "研发中心"
    FINANCE = "财务部"


class ServiceStatus(StrEnum):
    OK = "ok"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


class APIPath(StrEnum):
    API_V1 = "/api/v1"
    HEALTH = "/health"


class UserCommand(StrEnum):
    EXIT = "exit"
    QUIT = "quit"
    EXIT_CN = "退出"


class ErrorCode(StrEnum):
    RATE_LIMIT = "429"


class ResultType(StrEnum):
    TEXT = "text"
    MARKDOWN = "markdown"
    MERMAID = "mermaid"
    HTML = "html"
    SEARCH_RESULTS = "search_results"


__all__ = [
    "APIPath",
    "Department",
    "DocumentFormat",
    "ErrorCode",
    "ModelName",
    "ModelProvider",
    "ResultType",
    "ServiceStatus",
    "ToolName",
    "UserCommand",
]
