"""LlamaIndex Agent 工作流模块

使用 LlamaIndex 0.14.x 的 AgentWorkflow 替代 LangChain 的 AgentChain，
支持多轮工具调用和会话记忆管理。

关键组件（LlamaIndex 0.14.x 新 API）：
- AgentWorkflow：LlamaIndex 0.14.x 的 Agent 工作流框架
- ChatMemoryBuffer：会话记忆管理
- 系统提示词与 LangChain 版本保持一致
"""

import logging


from llama_index.core import VectorStoreIndex
from llama_index.core.agent import AgentWorkflow
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.nvidia import NVIDIA

from app.agent.tools import (
    get_employee_info_tool,
    make_generate_flowchart_code_tool,
    make_generate_html_prototype_tool,
    make_generate_prd_document_tool,
    make_search_csv_data_tool,
    make_search_internal_documents_tool,
    make_search_web_tool,
    send_email_notification_tool,
)
from app.core.config import settings

logger = logging.getLogger(__name__)

__all__ = ["clear_session", "create_agent", "get_or_create_memory"]

_SYSTEM_PROMPT = (
    "你是一个企业内部办公效能助手，不仅支持检索内部文档和通讯录，还具备产品经理效能辅助能力。\n\n"
    "工具选择决策规则（按优先级排序）：\n"
    "1. 查询员工联系方式、职位、部门 → 使用 get_employee_info 工具\n"
    "2. 查询公司内部文档、规章制度、流程规范 → 使用 search_internal_documents 工具\n"
    "3. 查询互联网上的最新信息、行业动态、外部知识 → 使用 search_web 工具\n"
    "4. 需要通知或提醒其他员工 → 使用 send_email_notification 工具\n"
    "5. 写需求文档、PRD 或功能说明 → 必须调用 generate_prd_document 工具\n"
    "6. 画流程图、业务流转图 → 必须调用 generate_flowchart_code 工具\n"
    "7. 画原型、设计页面布局 → 必须调用 generate_html_prototype 工具\n"
    "8. 查询数据统计、表格数据筛选、汇总计算 → 使用 search_csv_data 工具\n\n"
    "重要：联网搜索 vs 内部文档的判断标准：\n"
    "- 如果用户问题明确涉及公司内部制度、流程、规范 → search_internal_documents\n"
    "- 如果用户问题涉及外部信息"
    "（如行业趋势、市场数据、新闻资讯、技术博客、公开知识）"
    "→ search_web\n"
    "- 如果 search_internal_documents 返回的结果与用户问题不相关"
    " → 应该再尝试 search_web\n"
    "- 如果不确定信息来源 → 先尝试 search_internal_documents，"
    "结果不相关时再用 search_web\n\n"
    "重要原则：\n"
    "- 必须先调用对应工具获取信息，再基于工具返回结果给出自然语言回答\n"
    "- 不要在没有调用工具的情况下编造数据\n"
    "- 回答时注明信息来源（文档名或搜索结果）\n"
    "- 对于产品经理效能相关的请求，必须使用对应的生成工具，不要自行编写文档内容\n"
    "- 如果检索结果与用户问题不相关，"
    "不要反复尝试不同的关键词搜索，应直接告知用户未找到相关信息并给出建议"
)

# 会话存储：session_id -> ChatMemoryBuffer
_sessions: dict[str, ChatMemoryBuffer] = {}


def create_agent(index: VectorStoreIndex, llm: NVIDIA) -> AgentWorkflow:
    """创建 LlamaIndex AgentWorkflow。

    LlamaIndex 0.14.x 使用 AgentWorkflow 替代旧版 ReActAgent.from_tools()。

    Args:
        index: VectorStoreIndex 实例
        llm: NVIDIA LLM 实例

    Returns:
        AgentWorkflow 实例
    """
    logger.info("创建 AgentWorkflow...")

    tools = [
        make_search_internal_documents_tool(index),
        get_employee_info_tool,
        make_search_web_tool(),
        send_email_notification_tool,
        make_generate_prd_document_tool(llm),
        make_generate_flowchart_code_tool(llm),
        make_generate_html_prototype_tool(llm),
        make_search_csv_data_tool(settings.CSV_DIR, llm),
    ]

    logger.info("已注册 %d 个工具", len(tools))

    agent = AgentWorkflow.from_tools_or_functions(
        tools_or_functions=tools,
        llm=llm,
        system_prompt=_SYSTEM_PROMPT,
        verbose=True,
    )

    logger.info("AgentWorkflow 创建完成")
    return agent


def get_or_create_memory(session_id: str) -> ChatMemoryBuffer:
    """获取或创建会话记忆。

    Args:
        session_id: 会话 ID

    Returns:
        ChatMemoryBuffer 实例
    """
    if session_id not in _sessions:
        _sessions[session_id] = ChatMemoryBuffer.from_defaults(
            token_limit=settings.SESSION_MAX_MESSAGES * 500,
        )
    return _sessions[session_id]


def clear_session(session_id: str) -> None:
    """清除会话记忆。

    Args:
        session_id: 会话 ID
    """
    if session_id in _sessions:
        del _sessions[session_id]
        logger.info("已清除会话：%s", session_id)
    else:
        logger.warning("会话 %s 不存在", session_id)
