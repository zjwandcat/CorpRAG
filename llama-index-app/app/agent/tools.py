"""LlamaIndex Agent 工具模块

将 LangChain 的 7 个 @tool 转换为 LlamaIndex 的 FunctionTool，
并新增 CSV 统计查找工具，集成 NVIDIA Blueprints 的文档生成优化模板。

关键适配：
- FunctionTool.from_defaults(fn=func) 替代 @tool 装饰器
- LLM 调用使用 llm.chat(messages) 替代 llm.invoke(messages)
- ChatMessage 来自 llama_index.core.llms
- PandasQueryEngine 用于 CSV 数据的智能统计查询
"""

import logging
from pathlib import Path

import pandas as pd
from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.query_engine import PandasQueryEngine
from llama_index.core.tools import FunctionTool
from llama_index.llms.nvidia import NVIDIA


from app.rag.index_store import hybrid_retrieve

logger = logging.getLogger(__name__)

# 演示用硬编码员工数据，未接入真实 OA 系统
_DEMO_EMPLOYEES: dict[str, dict[str, str]] = {
    "张三": {
        "部门": "研发中心",
        "职位": "高级工程师",
        "邮箱": "zhangsan@company.com",
        "电话": "13800001111",
    },
    "李四": {
        "部门": "人事部",
        "职位": "HR经理",
        "邮箱": "lisi@company.com",
        "电话": "13800002222",
    },
    "王五": {
        "部门": "财务部",
        "职位": "财务主管",
        "邮箱": "wangwu@company.com",
        "电话": "13800003333",
    },
    "赵六": {
        "部门": "研发中心",
        "职位": "产品经理",
        "邮箱": "zhaoliu@company.com",
        "电话": "13800004444",
    },
}

__all__ = [
    "get_employee_info_tool",
    "make_generate_flowchart_code_tool",
    "make_generate_html_prototype_tool",
    "make_generate_prd_document_tool",
    "make_search_csv_data_tool",
    "make_search_internal_documents_tool",
    "make_search_web_tool",
    "send_email_notification_tool",
]


def make_search_internal_documents_tool(index: VectorStoreIndex) -> FunctionTool:
    """创建检索内部文档工具（LlamaIndex 版本）。"""

    def search_internal_documents(query: str, department: str = "通用") -> str:
        """检索公司内部办公文档，仅用于查询公司内部的规章制度、流程规范、技术手册等内部资料。不支持查询外部互联网信息。

        Args:
            query: 检索关键词，例如 "报销流程"、"请假制度"
            department: 部门名称，用于过滤检索范围，默认为"通用"表示不限部门

        Returns:
            相关文档的文本内容，如果未找到则返回提示信息
        """
        logger.info(
            "执行工具 search_internal_documents，query=%s, department=%s",
            query,
            department,
        )

        results = hybrid_retrieve(
            query=query,
            index=index,
            department=department,
            top_k=3,
        )

        if not results:
            result = f"未找到关于「{query}」的内部文档（部门：{department}）。"
            logger.info("工具返回结果：%s", result)
            return result

        formatted_parts: list[str] = []
        for r in results:
            source = r.node.metadata.get("source", "未知来源")
            content = r.node.get_content()
            formatted_parts.append(f"【来源：{source}】\n{content}")

        result = "\n\n".join(formatted_parts)
        logger.info("检索到 %d 个相关文本块（混合检索）", len(results))
        return result

    return FunctionTool.from_defaults(fn=search_internal_documents)


def get_employee_info(employee_name: str) -> str:
    """⚠️ 演示数据：使用硬编码员工信息，未接入真实 OA 系统。

    当用户询问某位员工的联系方式、职位、部门等信息时调用此工具。

    Args:
        employee_name: 员工姓名，例如 "张三"

    Returns:
        员工信息字符串（包含演示数据标识），如果未找到则返回"查无此人"
    """
    logger.info("执行工具 get_employee_info，employee_name=%s", employee_name)
    info = _DEMO_EMPLOYEES.get(employee_name)
    if info is None:
        result = f"查无此人：{employee_name}"
    else:
        details = "、".join([f"{k}：{v}" for k, v in info.items()])
        result = f"{employee_name} — {details}（演示数据）"
    logger.info("工具返回结果：%s", result)
    return result


get_employee_info_tool = FunctionTool.from_defaults(fn=get_employee_info)


def make_search_web_tool() -> FunctionTool:
    """创建联网搜索工具（LlamaIndex 版本）。"""

    def search_web(query: str) -> str:
        """联网搜索外部信息。当用户问题涉及以下场景时必须使用此工具：
        - 互联网上的最新资讯、新闻、行业动态
        - 外部公开知识（如技术博客、百科知识、市场数据）
        - 内部文档检索未找到相关信息时，作为补充搜索
        - 任何非公司内部的信息查询

        注意：不要用此工具搜索公司内部文档，内部文档请使用 search_internal_documents。

        Args:
            query: 搜索关键词，例如 "2024年AI行业趋势"、"比亚迪最新财报"

        Returns:
            搜索结果文本
        """
        logger.info("执行工具 search_web，query=%s", query)
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
            if not results:
                result = f"未找到关于「{query}」的搜索结果。"
            else:
                formatted: list[str] = []
                for r in results:
                    title = r.get("title", "无标题")
                    body = r.get("body", "无摘要")
                    href = r.get("href", "")
                    formatted.append(f"【{title}】\n{body}\n来源：{href}")
                result = "\n\n".join(formatted)
        except Exception as exc:
            result = (
                f"联网搜索暂时不可用（{type(exc).__name__}），"
                "建议先尝试查询内部知识库文档。"
            )
        logger.info("工具返回结果：%s...", result[:100])
        return result

    return FunctionTool.from_defaults(fn=search_web)


def send_email_notification(to_employee: str, subject: str, body: str) -> str:
    """⚠️ 模拟功能：不会真实发送邮件，仅返回模拟成功消息。

    当用户需要通知、提醒或发送信息给其他员工时调用此工具。
    实际项目中可集成 SMTP 服务实现真实邮件发送。

    Args:
        to_employee: 收件人姓名，例如 "张三"
        subject: 邮件主题
        body: 邮件正文

    Returns:
        模拟发送结果消息（包含 [模拟] 前缀标识）
    """
    logger.info(
        "执行工具 send_email_notification，to_employee=%s, subject=%s",
        to_employee,
        subject,
    )
    logger.warning(
        "模拟发送邮件：未接入 SMTP 服务，收件人=%s，主题=%s",
        to_employee,
        subject,
    )
    result = f"[模拟] 邮件已成功发送给 {to_employee}，主题：{subject}"
    logger.info("工具返回结果：%s", result)
    return result


send_email_notification_tool = FunctionTool.from_defaults(fn=send_email_notification)


def _invoke_llm(llm: NVIDIA, system_prompt: str, user_content: str) -> str:
    """调用 LlamaIndex NVIDIA LLM 的辅助函数。"""
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
        ChatMessage(role=MessageRole.USER, content=user_content),
    ]
    response = llm.chat(messages)
    return str(response.content) if response.content else ""


def make_generate_prd_document_tool(llm: NVIDIA) -> FunctionTool:
    """创建 PRD 文档生成工具（LlamaIndex 版本，集成 NVIDIA Blueprints 优化模板）。"""

    def generate_prd_document(feature_name: str, brief_description: str) -> str:
        """根据功能名称和简述，生成结构化的标准 PRD 文档。

        当产品经理需要快速撰写产品需求文档时调用此工具，
        输出包含需求背景、目标用户、核心功能点、
        业务流程、异常处理、验收标准的 Markdown 文档。

        集成 NVIDIA Blueprints 的文档生成优化模板，提升 PRD 的结构化程度和完整性。

        Args:
            feature_name: 功能名称，例如 "智能报销审批"
            brief_description: 功能简要描述，
                例如 "通过 AI 自动审核报销单据，减少人工审批工作量"

        Returns:
            格式化的 Markdown PRD 文档字符串
        """
        logger.info("执行工具 generate_prd_document，feature_name=%s", feature_name)
        try:
            # NVIDIA Blueprints 优化的 PRD 生成模板
            system_prompt = (
                "你是一位资深产品经理，请根据以下信息撰写一份结构化的产品需求文档（PRD）。\n\n"
                "## NVIDIA Blueprints PRD 生成模板\n\n"
                "请按以下结构输出 Markdown 格式的 PRD：\n"
                "1. 需求背景 — 描述业务痛点和需求产生的背景\n"
                "2. 目标用户 — 明确产品的核心用户群体和使用场景\n"
                "3. 核心功能点 — 列出功能的关键特性（使用表格形式）\n"
                "4. 业务流程 — 描述完整的业务流转路径\n"
                "5. 异常处理 — 列出可能的异常场景及应对策略\n"
                "6. 验收标准 — 定义可量化的验收指标\n"
                "7. 优先级与排期 — 建议的开发优先级和里程碑\n\n"
                "### 输出要求\n"
                "- 使用 Markdown 格式\n"
                "- 核心功能点使用表格展示\n"
                "- 业务流程使用编号列表\n"
                "- 验收标准必须可量化、可测试"
            )
            user_content = f"功能名称：{feature_name}\n简要描述：{brief_description}"
            result = _invoke_llm(llm, system_prompt, user_content)
        except Exception as exc:
            result = f"PRD 文档生成失败：{exc}"
        logger.info("工具返回结果：%s...", result[:100])
        return result

    return FunctionTool.from_defaults(fn=generate_prd_document)


def make_generate_flowchart_code_tool(llm: NVIDIA) -> FunctionTool:
    """创建流程图代码生成工具（LlamaIndex 版本）。"""

    def generate_flowchart_code(process_description: str) -> str:
        """将业务流程描述转化为 Mermaid.js flowchart 语法代码。

        当产品经理需要将文字描述的流程转化为可视化流程图时调用此工具，
        输出合法的 Mermaid 代码，可直接在 Draw.io 等工具中渲染。

        Args:
            process_description: 业务流程的文字描述，
                例如 "用户提交报销单 -> 主管审批 -> 财务复核 -> 打款"

        Returns:
            包含 Mermaid flowchart 代码的字符串
        """
        logger.info(
            "执行工具 generate_flowchart_code，process_description=%s",
            process_description,
        )
        try:
            system_prompt = (
                "你是一位流程设计专家，"
                "请将以下业务流程描述转化为 Mermaid.js flowchart 语法代码。\n"
                "仅输出合法的 Mermaid 代码，不要输出任何其他解释文字。\n"
                "请使用 graph TD 或 graph LR 语法，确保节点命名清晰、流程完整。"
            )
            result = _invoke_llm(
                llm, system_prompt, f"业务流程描述：{process_description}"
            )
        except Exception as exc:
            result = f"流程图代码生成失败：{exc}"
        logger.info("工具返回结果：%s...", result[:100])
        return result

    return FunctionTool.from_defaults(fn=generate_flowchart_code)


def make_generate_html_prototype_tool(llm: NVIDIA) -> FunctionTool:
    """创建 HTML 原型生成工具（LlamaIndex 版本）。"""

    def generate_html_prototype(page_description: str) -> str:
        """根据页面描述生成单文件 HTML 低保真原型。

        当产品经理需要快速生成前端低保真原型时调用此工具，
        输出使用 Tailwind CSS 的完整单文件 HTML 代码，包含按钮、表单和布局。

        Args:
            page_description: 页面描述，
                例如 "报销提交页面，包含金额输入框、事由文本框、提交按钮和取消按钮"

        Returns:
            完整的 HTML 代码字符串
        """
        logger.info(
            "执行工具 generate_html_prototype，page_description=%s",
            page_description,
        )
        try:
            system_prompt = (
                "你是一位前端原型设计专家，"
                "请根据以下页面描述生成一个完整的单文件 HTML 低保真原型。\n"
                "要求：\n"
                "1. 使用 Tailwind CSS（通过 CDN 引入：https://cdn.tailwindcss.com）\n"
                "2. 包含简单的按钮、表单和布局\n"
                "3. 代码完整可运行，不需要额外依赖\n"
                "4. 仅输出 HTML 代码，不要输出任何解释文字"
            )
            result = _invoke_llm(llm, system_prompt, f"页面描述：{page_description}")
        except Exception as exc:
            result = f"HTML 原型生成失败：{exc}"
        logger.info("工具返回结果：%s...", result[:100])
        return result

    return FunctionTool.from_defaults(fn=generate_html_prototype)


def make_search_csv_data_tool(csv_dir: Path, llm: NVIDIA) -> FunctionTool:
    """创建 CSV 数据统计查找工具（LlamaIndex 版本，新增功能）。

    使用 PandasQueryEngine 将自然语言查询转化为 pandas 操作，
    实现 CSV 数据的智能统计和查找。
    """

    def search_csv_data(query: str, file_name: str | None = None) -> str:
        """在 CSV 数据文件中执行统计查找。

        当用户需要查询数据统计、表格数据筛选、汇总计算时调用此工具。

        Args:
            query: 查询描述，例如 "2024年销售额最高的部门"
            file_name: 可选的 CSV 文件名

        Returns:
            统计结果文本
        """
        logger.info(
            "执行工具 search_csv_data，query=%s, file_name=%s",
            query,
            file_name,
        )

        csv_path = Path(csv_dir)
        if not csv_path.exists():
            return f"CSV 数据目录不存在：{csv_dir}"

        csv_files = list(csv_path.glob("*.csv"))
        if file_name:
            csv_files = [f for f in csv_files if f.name == file_name]

        if not csv_files:
            return f"未找到匹配的 CSV 数据文件（目录：{csv_dir}）"

        results: list[str] = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                query_engine = PandasQueryEngine(df=df, llm=llm)
                response = query_engine.query(query)
                results.append(f"【数据源：{csv_file.name}】\n{response}")
            except Exception as exc:
                results.append(f"【数据源：{csv_file.name}】\n查询失败：{exc}")

        result = "\n\n".join(results) if results else "未找到匹配的 CSV 数据文件"
        logger.info("工具返回结果：%s...", result[:100])
        return result

    return FunctionTool.from_defaults(fn=search_csv_data)
