from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "DocumentProcessingPerformance",
    "HardwareInfo",
    "HardwareStatusResponse",
    "HealthResponse",
    "OptimizedUploadResponse",
    "PRDExportRequest",
    "RetrievalPerformance",
    "RetrievalTestResponse",
    "SourceReference",
    "ToolCallStep",
    "UploadRequest",
    "UploadResponse",
]


class SourceReference(BaseModel):
    source: Annotated[
        str,
        Field(description="来源文件名", examples=["报销流程规范.pdf"]),
    ]
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
        Literal["cuda", "openvino_gpu", "cpu", ""],
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
        Literal["pure_cloud", "hybrid_acceleration"],
        Field(
            default="pure_cloud",
            description="加速模式",
        ),
    ] = "pure_cloud"


class HealthResponse(BaseModel):
    status: Annotated[str, Field(description="服务状态")]
    vectorstore_count: Annotated[int, Field(ge=0, description="向量库中的文档数量")]
    model_name: Annotated[str, Field(description="当前使用的模型名称")]
    acceleration_mode: Annotated[
        Literal["pure_cloud", "hybrid_acceleration"],
        Field(
            default="pure_cloud",
            description="加速模式：pure_cloud=纯云端，hybrid_acceleration=混合加速",
        ),
    ] = "pure_cloud"
    local_models_loaded: Annotated[
        bool,
        Field(
            default=False,
            description="本地模型（Embedding）是否已成功加载",
        ),
    ] = False


class PRDExportRequest(BaseModel):
    feature_name: Annotated[str, Field(description="功能名称")]
    content: Annotated[str, Field(description="PRD 文档内容（Markdown 格式）")]


class DocumentProcessingPerformance(BaseModel):
    """文档处理性能指标"""
    ocr_time_ms: Annotated[float, Field(default=0.0, ge=0.0, description="OCR 识别耗时（毫秒）")]
    cleaning_time_ms: Annotated[float, Field(default=0.0, ge=0.0, description="文本清洗耗时（毫秒）")]
    slicing_time_ms: Annotated[float, Field(default=0.0, ge=0.0, description="文本切片耗时（毫秒）")]
    total_time_ms: Annotated[float, Field(default=0.0, ge=0.0, description="总处理耗时（毫秒）")]
    device: Annotated[str, Field(default="cpu", description="处理设备：cuda/openvino_gpu/cpu")]


class RetrievalPerformance(BaseModel):
    """检索性能指标"""
    bm25_time_ms: Annotated[float, Field(default=0.0, ge=0.0, description="BM25 检索耗时（毫秒）")]
    vector_time_ms: Annotated[float, Field(default=0.0, ge=0.0, description="向量检索耗时（毫秒）")]
    rrf_time_ms: Annotated[float, Field(default=0.0, ge=0.0, description="RRF 融合耗时（毫秒）")]
    reranker_time_ms: Annotated[float, Field(default=0.0, ge=0.0, description="Reranker 精排耗时（毫秒）")]
    total_time_ms: Annotated[float, Field(default=0.0, ge=0.0, description="总检索耗时（毫秒）")]
    device: Annotated[str, Field(default="cpu", description="检索设备：cuda/openvino_gpu/cpu")]
    use_reranker: Annotated[bool, Field(default=False, description="是否使用了 Reranker")]


class HardwareInfo(BaseModel):
    """硬件信息"""
    device: Annotated[Literal["cuda", "openvino_gpu", "cpu"], Field(description="计算设备类型")]
    mode: Annotated[Literal["pure_cloud", "hybrid_acceleration"], Field(description="加速模式")]
    gpu_name: Annotated[str, Field(default="", description="GPU 名称")]
    gpu_memory_gb: Annotated[float, Field(default=0.0, ge=0.0, description="GPU 显存大小（GB）")]
    available_optimizations: Annotated[list[str], Field(default_factory=list, description="可用优化功能列表")]


class HardwareStatusResponse(BaseModel):
    """硬件状态接口响应"""
    device: Annotated[Literal["cuda", "openvino_gpu", "cpu"], Field(description="计算设备类型")]
    mode: Annotated[Literal["pure_cloud", "hybrid_acceleration"], Field(description="加速模式")]
    gpu_name: Annotated[str, Field(default="", description="GPU 名称")]
    gpu_memory_gb: Annotated[float, Field(default=0.0, ge=0.0, description="GPU 显存大小（GB）")]
    available_optimizations: Annotated[list[str], Field(default_factory=list, description="可用优化功能列表")]


class OptimizedUploadResponse(BaseModel):
    """优化文档上传接口响应"""
    filename: Annotated[str, Field(description="上传的文件名")]
    chunks_added: Annotated[int, Field(ge=0, description="添加到向量库的文本块数量")]
    message: Annotated[str, Field(description="操作结果消息")]
    department: Annotated[str, Field(default="通用", description="文档所属部门")]
    processing_performance: Annotated[DocumentProcessingPerformance, Field(description="文档处理性能指标")]
    hardware_info: Annotated[HardwareInfo, Field(description="硬件信息")]


class RetrievalTestResponse(BaseModel):
    """检索性能测试接口响应"""
    query: Annotated[str, Field(description="查询文本")]
    results: Annotated[list[dict[str, Any]], Field(default_factory=list, description="检索结果")]
    performance: Annotated[RetrievalPerformance, Field(description="检索性能指标")]
    hardware_info: Annotated[HardwareInfo, Field(description="硬件信息")]
