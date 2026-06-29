from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.core.enums import (
    ReviewStatus,
    ReviewType,
    Severity,
    SSEEventType,
)

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
    "HITLApprovalRequiredData",
    # === 审查相关模型 ===
    "MCPCallRequest",
    "MCPCallResponse",
    "MCPToolInfo",
    "ReviewConfigResponse",
    "ReviewConfigUpdateRequest",
    "ReviewFinding",
    "ReviewRequest",
    "ReviewResponse",
    "ReviewResultItem",
    "WorkerResult",
    # === HITL 审批模型 ===
    "HITLApprovalRequest",
    "HITLApprovalResponse",
    "HITLApprovalAction",
    "HITLApprovalResult",
    # === Guardrails 模型 ===
    "GuardrailLoopDetectedData",
    "GuardrailViolation",
    # === Knowledge Graph 模型 ===
    "KnowledgeGraphResult",
    "ToolCallRecord",
    "HitlPendingApproval",
    # === MLOps 模型 ===
    "EvalDatasetItem",
    "EvalRequest",
    "EvalResponse",
    "ABConfigRequest",
    "ABConfigResponse",
    "ABMetricsResponse",
    "DriftStatusResponse",
    "RecommendationItem",
    "IntentPredictionResult",
    "MLOpsHealthResponse",
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
    rerank_duration_ms: Annotated[
        float,
        Field(
            default=0.0,
            ge=0.0,
            description="Reranker 精排耗时（毫秒），0 表示未执行精排",
        ),
    ] = 0.0
    rerank_device: Annotated[
        str,
        Field(
            default="",
            description="Reranker 运行设备，空字符串表示未执行精排",
        ),
    ] = ""


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
    processing_time_ms: Annotated[
        float,
        Field(
            default=0.0,
            ge=0.0,
            description="文档处理耗时（毫秒），包含解析+向量化",
        ),
    ] = 0.0
    acceleration_mode: Annotated[
        str,
        Field(
            default="cloud_api",
            description="加速模式：cloud_api=纯云端 API",
        ),
    ] = "cloud_api"


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
    acceleration_mode: Annotated[
        str,
        Field(
            default="cloud_api",
            description="加速模式：cloud_api=纯云端 API",
        ),
    ] = "cloud_api"


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


class HITLApprovalRequiredData(SSEEventData):
    """hitl_approval_required 事件数据"""

    approval_id: Annotated[str, Field(description="审批唯一标识")]
    tool_name: Annotated[str, Field(description="待审批的工具名称")]
    tool_args: Annotated[dict[str, Any], Field(default_factory=dict, description="工具调用参数")]


class SSEEvent(BaseModel):
    """SSE 事件完整结构"""

    event: SSEEventType
    data: SSEEventData
    id: Annotated[str, Field(description="事件唯一标识，格式: {thread_id}-{序号}")]


# =============================================================================
# 审查相关数据模型
# =============================================================================


class ReviewFinding(BaseModel):
    """审查问题模型"""

    severity: Annotated[Severity, Field(description="严重程度")]
    description: Annotated[str, Field(description="问题描述")]
    location: Annotated[str, Field(default="", description="问题位置")]
    suggestion: Annotated[str | None, Field(default=None, description="修复建议")]


class ReviewResultItem(BaseModel):
    """审查结果项模型"""

    dimension: Annotated[ReviewType, Field(description="审查维度")]
    status: Annotated[ReviewStatus, Field(description="审查状态")]
    findings: Annotated[
        list[ReviewFinding], Field(default_factory=list, description="发现的问题列表")
    ]
    duration_ms: Annotated[int, Field(ge=0, description="该维度审查耗时（毫秒）")]


class ReviewRequest(BaseModel):
    """审查请求模型"""

    code_content: Annotated[
        str | None,
        Field(default=None, max_length=100000, description="待审查的代码内容"),
    ]
    code_url: Annotated[
        str | None,
        Field(default=None, description="代码仓库 PR 链接"),
    ]
    review_type: Annotated[
        ReviewType,
        Field(default=ReviewType.FULL, description="审查类型"),
    ]
    session_id: Annotated[
        str | None,
        Field(default=None, description="会话 ID"),
    ]
    stream: Annotated[
        bool,
        Field(default=False, description="是否使用 SSE 流式输出"),
    ]

    @model_validator(mode="after")
    def validate_code_source(self) -> "ReviewRequest":
        """校验 code_content 和 code_url 二选一"""
        if not self.code_content and not self.code_url:
            raise ValueError("必须提供 code_content 或 code_url 之一")
        return self


class ReviewResponse(BaseModel):
    """审查响应模型"""

    session_id: Annotated[str, Field(description="审查会话 ID")]
    review_type: Annotated[ReviewType, Field(description="实际执行的审查类型")]
    results: Annotated[list[ReviewResultItem], Field(description="各维度审查结果列表")]
    summary: Annotated[str, Field(description="最终汇总报告（Markdown 格式）")]
    total_duration_ms: Annotated[int, Field(ge=0, description="审查总耗时（毫秒）")]
    is_fallback: Annotated[bool, Field(default=False, description="是否发生降级")]
    fallback_message: Annotated[str, Field(default="", description="降级提示信息")]


class MCPToolInfo(BaseModel):
    """MCP 工具信息模型"""

    name: Annotated[str, Field(description="工具名称")]
    description: Annotated[str, Field(description="工具描述")]
    parameters: Annotated[dict, Field(description="工具参数定义（JSON Schema）")]
    server_name: Annotated[str, Field(description="所属 MCP Server 名称")]


class MCPCallRequest(BaseModel):
    """MCP 工具调用请求模型"""

    tool_name: Annotated[str, Field(description="工具名称")]
    arguments: Annotated[dict[str, object], Field(description="工具调用参数")]
    server_name: Annotated[str | None, Field(default=None, description="目标 MCP Server 名称")]


class MCPCallResponse(BaseModel):
    """MCP 工具调用响应模型"""

    tool_name: Annotated[str, Field(description="工具名称")]
    result: Annotated[object, Field(description="工具执行结果")]
    server_name: Annotated[str, Field(description="执行该工具的 Server 名称")]
    duration_ms: Annotated[int, Field(ge=0, description="执行耗时（毫秒）")]


class ReviewConfigResponse(BaseModel):
    """配置查询响应模型"""

    multi_agent_enabled: Annotated[bool, Field(description="多Agent协作开关")]
    mcp_enabled: Annotated[bool, Field(description="MCP 总开关")]
    mcp_servers: Annotated[dict[str, bool], Field(description="各 MCP Server 启用状态")]
    review_types: Annotated[list[ReviewType], Field(description="启用的审查类型")]
    worker_timeout_seconds: Annotated[int, Field(ge=10, le=300, description="Worker 超时时间")]
    max_concurrent_reviews: Annotated[int, Field(ge=1, le=100, description="最大并发审查数")]
    warning: Annotated[str | None, Field(default=None, description="持久化警告")]


class ReviewConfigUpdateRequest(BaseModel):
    """配置更新请求模型，所有字段可选"""

    multi_agent_enabled: Annotated[bool | None, Field(default=None, description="多Agent协作开关")]
    mcp_enabled: Annotated[bool | None, Field(default=None, description="MCP 总开关")]
    mcp_servers: Annotated[
        dict[str, bool] | None, Field(default=None, description="各 MCP Server 启用状态")
    ]
    review_types: Annotated[
        list[ReviewType] | None, Field(default=None, description="启用的审查类型")
    ]
    worker_timeout_seconds: Annotated[
        int | None, Field(default=None, ge=10, le=300, description="Worker 超时时间")
    ]
    max_concurrent_reviews: Annotated[
        int | None, Field(default=None, ge=1, le=100, description="最大并发审查数")
    ]


class WorkerResult(BaseModel):
    """内部 Worker 结果模型"""

    dimension: Annotated[ReviewType, Field(description="审查维度")]
    status: Annotated[ReviewStatus, Field(description="审查状态")]
    findings: Annotated[
        list[ReviewFinding], Field(default_factory=list, description="发现的问题列表")
    ]
    duration_ms: Annotated[int, Field(ge=0, description="审查耗时（毫秒）")]
    error_message: Annotated[str | None, Field(default=None, description="错误信息")]


# =============================================================================
# HITL 审批数据模型
# =============================================================================


class HITLApprovalRequest(BaseModel):
    """HITL 审批请求模型"""

    tool_name: str = Field(..., description="待审批的工具名称")
    tool_args: dict[str, Any] = Field(default_factory=dict, description="工具调用参数")
    session_id: str = Field(..., description="会话 ID")
    thread_id: str = Field(..., description="LangGraph 线程 ID")


class HITLApprovalResponse(BaseModel):
    """HITL 审批响应模型"""

    approval_id: str = Field(..., description="审批唯一标识")
    status: str = Field(default="pending", description="审批状态")
    tool_name: str = Field(..., description="待审批的工具名称")
    tool_args: dict[str, Any] = Field(default_factory=dict, description="工具调用参数")
    created_at: str = Field(..., description="创建时间 ISO 8601")


class HITLApprovalAction(BaseModel):
    """HITL 审批操作模型"""

    approval_id: str = Field(..., description="审批唯一标识")
    action: Literal["approve", "reject"] = Field(..., description="审批动作")
    reason: str | None = Field(default=None, description="审批理由")


class HITLApprovalResult(BaseModel):
    """HITL 审批结果模型"""

    approval_id: str = Field(..., description="审批唯一标识")
    status: str = Field(..., description="审批最终状态")
    resolved_at: str = Field(..., description="审批时间 ISO 8601")
    reason: str | None = Field(default=None, description="审批理由")


# =============================================================================
# Guardrails 数据模型
# =============================================================================


class GuardrailLoopDetectedData(SSEEventData):
    """Guardrails 死循环检测 SSE 事件数据"""

    tool_name: str = Field(..., description="触发检测的工具名称")
    repetition_count: int = Field(..., description="重复调用次数")
    action: str = Field(default="block", description="护栏动作")


class GuardrailViolation(BaseModel):
    """护栏违规记录"""

    violation_type: Literal["infinite_loop", "recursion_limit"] = Field(
        ..., description="违规类型"
    )
    tool_name: str = Field(..., description="工具名称")
    details: str = Field(default="", description="违规详情")
    timestamp: str = Field(..., description="违规时间 ISO 8601")


# =============================================================================
# Knowledge Graph 数据模型
# =============================================================================


class KnowledgeGraphResult(BaseModel):
    """知识图谱检索结果"""

    entity: str = Field(..., description="实体名称")
    relation: str = Field(..., description="关系类型")
    target_entity: str = Field(..., description="目标实体")
    confidence: float = Field(default=1.0, description="置信度")
    source_document: str = Field(default="", description="来源文档")


class ToolCallRecord(BaseModel):
    """工具调用记录（用于死循环检测）"""

    tool_name: str = Field(..., description="工具名称")
    tool_args: dict[str, Any] = Field(default_factory=dict, description="工具参数")
    timestamp: str = Field(..., description="调用时间 ISO 8601")
    iteration: int = Field(default=0, description="迭代轮次")


class HitlPendingApproval(BaseModel):
    """HITL 审批挂起信息"""

    task_id: str = Field(..., description="审批任务唯一标识")
    tool_name: str = Field(..., description="待审批的工具名称")
    tool_args: dict[str, Any] = Field(default_factory=dict, description="工具调用参数")
    tool_call_id: str = Field(default="", description="LangGraph ToolMessage ID")
    requested_at: str = Field(..., description="请求时间 ISO 8601")
    status: Literal["pending", "approved", "rejected"] = Field(
        default="pending", description="审批状态"
    )


# =============================================================================
# MLOps 数据模型
# =============================================================================


class EvalDatasetItem(BaseModel):
    """评估数据集条目"""

    query: str = Field(..., description="用户查询")
    ground_truth: str = Field(..., description="标准答案")
    expected_contexts: list[str] = Field(default_factory=list, description="期望的上下文片段")


class EvalRequest(BaseModel):
    """评估请求模型"""

    dataset_path: str = Field(default="tests/eval/eval_dataset.json", description="评估数据集路径")


class EvalResponse(BaseModel):
    """评估响应模型"""

    faithfulness_score: float = Field(..., ge=0.0, le=1.0, description="忠实度得分")
    answer_relevancy_score: float = Field(..., ge=0.0, le=1.0, description="答案相关性得分")
    context_precision_score: float = Field(..., ge=0.0, le=1.0, description="上下文精确率")
    context_recall_score: float = Field(..., ge=0.0, le=1.0, description="上下文召回率")
    eval_timestamp: str = Field(..., description="评估时间戳")
    dataset_version: str = Field(default="v1.0", description="数据集版本")


class ABConfigRequest(BaseModel):
    """A/B 测试配置请求"""

    bucket_a_ratio: float = Field(default=0.5, ge=0.0, le=1.0, description="Bucket A 流量比例")
    bucket_a_strategy: str = Field(default="self_hosted_rag", description="Bucket A 策略")
    bucket_b_strategy: str = Field(default="blueprint_rag", description="Bucket B 策略")
    enabled: bool = Field(default=True, description="是否启用")


class ABConfigResponse(BaseModel):
    """A/B 测试配置响应"""

    bucket_a_ratio: float
    bucket_a_strategy: str
    bucket_b_strategy: str
    enabled: bool


class ABMetricsResponse(BaseModel):
    """A/B 测试指标响应"""

    bucket_a_avg_latency_ms: float = Field(default=0.0, description="Bucket A 平均延迟")
    bucket_b_avg_latency_ms: float = Field(default=0.0, description="Bucket B 平均延迟")
    bucket_a_total_requests: int = Field(default=0, description="Bucket A 总请求数")
    bucket_b_total_requests: int = Field(default=0, description="Bucket B 总请求数")


class DriftStatusResponse(BaseModel):
    """漂移检测状态响应"""

    status: str = Field(default="normal", description="检测器状态")
    last_check_timestamp: str | None = Field(default=None, description="最近检测时间")
    last_drift_score: float | None = Field(default=None, description="最近漂移分数")
    alert_count: int = Field(default=0, description="告警次数")


class RecommendationItem(BaseModel):
    """推荐条目"""

    document_id: str = Field(..., description="文档 ID")
    content: str = Field(..., description="文档内容")
    source: str = Field(default="", description="来源")
    similarity_score: float = Field(default=0.0, ge=0.0, le=1.0, description="相似度得分")


class IntentPredictionResult(BaseModel):
    """意图预测结果"""

    intent_label: str = Field(..., description="意图标签")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")


class MLOpsHealthResponse(BaseModel):
    """MLOps 健康检查响应"""

    mlflow_connected: bool = Field(default=False, description="MLflow 连接状态")
    drift_detector_status: str = Field(default="unavailable", description="漂移检测器状态")
    ab_testing_enabled: bool = Field(default=False, description="A/B 测试是否启用")
