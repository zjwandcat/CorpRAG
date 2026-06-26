from typing import Annotated, Literal

from pydantic import BaseModel, Field

from app.core.enums import SSEEventType

__all__ = [
    # === 原有模型 ===
    "ChatRequest",
    "ChatResponse",
    "HealthResponse",
    "PRDExportRequest",
    "SourceReference",
    "ToolCallStep",
    "UploadRequest",
    "UploadResponse",
    # === SSE 事件模型 ===
    "SSEEventData",
    "SSEEvent",
    "StreamStartData",
    "AgentStartData",
    "AgentEndData",
    "ToolCallData",
    "ToolResultData",
    "ToolErrorData",
    "RetrievalResultData",
    "RetrievalEmptyData",
    "AgentRetryData",
    "AgentForceEndData",
    "MemoryLoadFailedData",
    "StreamEndData",
    "StreamErrorData",
]


class SourceReference(BaseModel):
    source: Annotated[str, Field(description="来源文件名", examples=["报销流程规范.pdf"])]
    department: Annotated[
        str, Field(default="通用", description="文档所属部门", examples=["财务部"])
    ]
    score: Annotated[
        float,
        Field(
            default=0.0,
            ge=0.0,
            le=1.0,
            description="相似度分数（0-1）",
            examples=[0.87],
        ),
    ]
    snippet: Annotated[
        str,
        Field(
            default="",
            description="命中的文本片段摘要",
            examples=["员工报销需在费用发生后30天内提交..."],
        ),
    ]
    rerank_score: Annotated[
        float | None,
        Field(
            default=None,
            description="Reranker 精排分数（仅方案 A 有值），None 表示未经过精排",
        ),
    ]


class ToolCallStep(BaseModel):
    tool_name: Annotated[
        str,
        Field(
            description="工具名称",
            examples=["search_internal_documents"],
        ),
    ]
    tool_args: Annotated[
        dict[str, object],
        Field(
            default_factory=dict,
            description="工具调用参数",
            examples=[{"query": "报销流程", "department": "财务部"}],
        ),
    ]
    tool_result: Annotated[
        str,
        Field(
            default="",
            description="工具执行结果（文本或代码）",
        ),
    ]
    tool_result_type: Annotated[
        Literal["text", "markdown", "mermaid", "html", "search_results"],
        Field(
            default="text",
            description="工具结果的内容类型，用于前端分发渲染",
            examples=["search_results"],
        ),
    ]
    sources: Annotated[
        list[SourceReference],
        Field(
            default_factory=list,
            description="检索来源溯源列表（仅检索类工具有值）",
        ),
    ]
    duration_ms: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            description="工具执行耗时（毫秒）",
            examples=[320],
        ),
    ]
    status: Annotated[
        Literal["success", "error"],
        Field(
            default="success",
            description="工具执行状态",
            examples=["success"],
        ),
    ]


class ChatRequest(BaseModel):
    query: Annotated[
        str,
        Field(
            min_length=1,
            max_length=2000,
            description="用户输入的问题",
            examples=["查一下报销流程"],
        ),
    ]
    session_id: Annotated[
        str | None,
        Field(
            default=None,
            description="会话 ID，用于多轮对话记忆。为空时自动生成新会话",
            examples=["550e8400-e29b-41d4-a716-446655440000"],
        ),
    ] = None
    rag_engine: Annotated[
        Literal["builtin", "blueprint"],
        Field(
            default="builtin",
            description="RAG 引擎选择：builtin=自建 RAG，blueprint=NVIDIA Blueprint",
        ),
    ] = "builtin"
    stream: Annotated[
        bool,
        Field(
            default=False,
            description="是否使用 SSE 流式输出。true 时使用 /chat/stream，false 时使用 /chat",
        ),
    ] = False


class ChatResponse(BaseModel):
    answer: Annotated[str, Field(description="AI 生成的回答内容")]
    answer_format: Annotated[
        Literal["text", "markdown"],
        Field(
            default="markdown",
            description="回答格式标记，前端据此选择渲染方式",
            examples=["markdown"],
        ),
    ]
    tools_used: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="本次对话中使用的工具名称列表（向后兼容）",
            examples=[["search_internal_documents", "get_employee_info"]],
        ),
    ]
    intermediate_steps: Annotated[
        list[ToolCallStep],
        Field(
            default_factory=list,
            description="Function Calling 执行步骤详情列表",
        ),
    ]
    total_duration_ms: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            description="整个对话处理总耗时（毫秒）",
            examples=[1500],
        ),
    ]
    session_id: Annotated[str, Field(description="会话 ID，可用于下次请求继续对话")]
    rag_engine: Annotated[
        Literal["builtin", "blueprint"],
        Field(
            default="builtin",
            description="实际使用的 RAG 引擎",
        ),
    ] = "builtin"
    is_fallback: Annotated[
        bool,
        Field(
            default=False,
            description="是否发生降级（Blueprint 不可用时降级到自建 RAG）",
        ),
    ] = False
    fallback_message: Annotated[
        str,
        Field(
            default="",
            description="降级提示信息，非空时前端展示降级提示",
        ),
    ] = ""


class UploadRequest(BaseModel):
    department: Annotated[
        str,
        Field(
            default="通用",
            description="文档所属部门，用于权限过滤",
            examples=["研发中心", "人事部", "财务部", "通用"],
        ),
    ]


class UploadResponse(BaseModel):
    filename: Annotated[str, Field(description="上传的文件名")]
    chunks_added: Annotated[int, Field(ge=0, description="添加到向量库的文本块数量")]
    message: Annotated[str, Field(description="操作结果消息")]
    department: Annotated[str, Field(default="通用", description="文档所属部门")]


class HealthResponse(BaseModel):
    status: Annotated[str, Field(description="服务状态")]
    vectorstore_count: Annotated[int, Field(ge=0, description="向量库中的文档数量")]
    model_name: Annotated[str, Field(description="当前使用的模型名称")]
    blueprint_available: Annotated[
        bool,
        Field(
            default=False,
            description="Blueprint 是否已配置且可用",
        ),
    ] = False


class PRDExportRequest(BaseModel):
    feature_name: Annotated[str, Field(description="功能名称")]
    content: Annotated[str, Field(description="PRD 文档内容（Markdown 格式）")]


# =============================================================================
# SSE 事件数据模型
# =============================================================================


class SSEEventData(BaseModel):
    """SSE 事件数据基类"""

    thread_id: Annotated[str, Field(description="会话线程 ID")]
    timestamp: Annotated[str, Field(description="ISO 8601 时间戳")]


class StreamStartData(SSEEventData):
    """stream_start 事件数据"""

    pass


class AgentStartData(SSEEventData):
    """agent_start 事件数据"""

    iteration: Annotated[int, Field(description="当前推理轮次", ge=1)]


class AgentEndData(SSEEventData):
    """agent_end 事件数据"""

    tool_calls: Annotated[
        list[dict] | None,
        Field(default=None, description="LLM 请求的工具调用列表"),
    ] = None
    has_final_answer: Annotated[bool, Field(description="是否已生成最终回答")]


class ToolCallData(SSEEventData):
    """tool_call 事件数据"""

    tool_name: Annotated[str, Field(description="工具名称")]
    tool_args: Annotated[dict, Field(description="工具调用参数")]


class ToolResultData(SSEEventData):
    """tool_result 事件数据"""

    tool_name: Annotated[str, Field(description="工具名称")]
    result_summary: Annotated[str, Field(description="结果摘要（脱敏）")]
    duration_ms: Annotated[int, Field(description="执行耗时毫秒", ge=0)]
    status: Annotated[Literal["success", "error"], Field(description="执行状态")]


class ToolErrorData(SSEEventData):
    """tool_error 事件数据"""

    tool_name: Annotated[str, Field(description="工具名称")]
    error_message: Annotated[str, Field(description="用户友好的错误提示")]


class RetrievalResultData(SSEEventData):
    """retrieval_result 事件数据"""

    sources: Annotated[list[SourceReference], Field(description="检索来源列表")]
    total_count: Annotated[int, Field(description="检索结果总数", ge=0)]


class RetrievalEmptyData(SSEEventData):
    """retrieval_empty 事件数据"""

    query: Annotated[str, Field(description="原始检索查询")]


class AgentRetryData(SSEEventData):
    """agent_retry 事件数据"""

    attempt: Annotated[int, Field(description="当前重试次数", ge=1)]
    max_attempts: Annotated[int, Field(description="最大重试次数")]


class AgentForceEndData(SSEEventData):
    """agent_force_end 事件数据"""

    iteration: Annotated[int, Field(description="当前循环次数")]
    max_iterations: Annotated[int, Field(description="最大循环次数")]


class MemoryLoadFailedData(SSEEventData):
    """memory_load_failed 事件数据"""

    pass


class StreamEndData(SSEEventData):
    """stream_end 事件数据"""

    chat_response: Annotated[ChatResponse, Field(description="完整的对话响应")]


class StreamErrorData(SSEEventData):
    """stream_error 事件数据"""

    error_type: Annotated[str, Field(description="错误类型标识")]
    error_message: Annotated[str, Field(description="用户友好的错误提示")]


class SSEEvent(BaseModel):
    """SSE 事件完整结构"""

    event: SSEEventType
    data: SSEEventData
    id: Annotated[str, Field(description="事件唯一标识，格式: {thread_id}-{序号}")]
