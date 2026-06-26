import logging
from typing import Any

from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool

logger = logging.getLogger(__name__)

__all__ = [
    "get_employee_info",
    "make_generate_flowchart_code_tool",
    "make_generate_html_prototype_tool",
    "make_generate_prd_document_tool",
    "make_search_internal_documents_tool",
    "make_search_web_tool",
    "send_email_notification",
]


def make_search_internal_documents_tool(vectorstore: Chroma, reranker: Any = None) -> BaseTool:
    @tool
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

        from app.rag.vectorstore import hybrid_search

        all_docs = vectorstore.get()["documents"]
        all_documents: list[str] = [doc for doc in all_docs if doc is not None] if all_docs else []

        # Step 1: hybrid_search（向量 + BM25 + RRF 融合）
        docs = hybrid_search(
            query=query,
            department=department,
            vectorstore=vectorstore,
            all_documents=all_documents,
            top_k=3,
        )

        if not docs:
            result = f"未找到关于「{query}」的内部文档（部门：{department}）。"
            logger.info("工具返回结果：%s", result)
            return result

        # Step 2: Reranker 精排
        rerank_metadata: list[dict[str, Any]] = []
        if reranker is not None:
            rerank_results = reranker.rerank(query=query, documents=docs, top_n=3)
            if rerank_results:
                docs = [r.document for r in rerank_results]
                # 更新 metadata 中的 rerank_score
                for r in rerank_results:
                    r.document.metadata["rerank_score"] = r.relevance_score
                logger.info("Reranker 精排完成，top %d 结果", len(rerank_results))
            else:
                logger.warning("Reranker 降级，使用 RRF 原始排序")

        formatted_parts: list[str] = []
        for d in docs:
            source = d.metadata.get("source", "未知来源")
            formatted_parts.append(f"【来源：{source}】\n{d.page_content}")

        result = "\n\n".join(formatted_parts)

        # 构建 RAG_METADATA 注释块（供 _extract_sources 解析）
        for d in docs:
            source = d.metadata.get("source", "未知来源")
            rerank_score = d.metadata.get("rerank_score")
            rerank_metadata.append(
                {
                    "source": source,
                    "score": float(rerank_score) if rerank_score is not None else 0.0,
                    "rerank_score": float(rerank_score) if rerank_score is not None else None,
                }
            )

        if rerank_metadata:
            import json

            metadata_json = json.dumps({"sources": rerank_metadata}, ensure_ascii=False)
            result += f"\n\n<!--RAG_METADATA:{metadata_json}-->"

        logger.info("检索到 %d 个相关文本块（混合检索 + Reranker）", len(docs))
        return result

    return search_internal_documents


@tool
def get_employee_info(employee_name: str) -> str:
    """查询员工信息（模拟 OA 系统接口）。

    当用户询问某位员工的联系方式、职位、部门等信息时调用此工具。

    Args:
        employee_name: 员工姓名，例如 "张三"

    Returns:
        员工信息字符串，如果未找到则返回"查无此人"
    """
    logger.info("执行工具 get_employee_info，employee_name=%s", employee_name)
    employees: dict[str, dict[str, str]] = {
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
    info = employees.get(employee_name)
    if info is None:
        result = f"查无此人：{employee_name}"
    else:
        details = "、".join([f"{k}：{v}" for k, v in info.items()])
        result = f"{employee_name} — {details}"
    logger.info("工具返回结果：%s", result)
    return result


def make_search_web_tool() -> BaseTool:
    @tool
    def search_web(query: str) -> str:
        """联网搜索外部信息。当用户问题涉及以下场景时必须使用此工具：
        - 互联网上的最新资讯、新闻、行业动态
        - 外部公开知识（如技术博客、百科知识、市场数据）
        - 内部文档检索未找到相关信息时，作为补充搜索
        - 任何非公司内部的信息查询

        注意：不要用此工具搜索公司内部文档，内部文档请使用 search_internal_documents。

        Args:
            query: 搜索关键词，例如 "2024年AI行业趋势"

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
            result = f"联网搜索暂时不可用（{type(exc).__name__}），建议先尝试查询内部知识库文档。"
        logger.info("工具返回结果：%s...", result[:100])
        return result

    return search_web


@tool
def send_email_notification(to_employee: str, subject: str, body: str) -> str:
    """模拟发送邮件通知给指定员工。

    当用户需要通知、提醒或发送信息给其他员工时调用此工具。
    实际项目中可集成 SMTP 服务实现真实邮件发送。

    Args:
        to_employee: 收件人姓名，例如 "张三"
        subject: 邮件主题
        body: 邮件正文

    Returns:
        发送结果消息
    """
    logger.info(
        "执行工具 send_email_notification，to_employee=%s, subject=%s",
        to_employee,
        subject,
    )
    result = f"邮件已成功发送给 {to_employee}，主题：{subject}"
    logger.info("工具返回结果：%s", result)
    return result


def _invoke_llm(llm: BaseChatModel, system_prompt: str, user_content: str) -> str:
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]
    response = llm.invoke(messages)
    content = response.content
    return str(content) if content else ""


def make_generate_prd_document_tool(llm: BaseChatModel) -> BaseTool:
    @tool
    def generate_prd_document(feature_name: str, brief_description: str) -> str:
        """根据功能名称和简述，生成结构化的标准 PRD 文档。

        当产品经理需要快速撰写产品需求文档时调用此工具，
        输出包含需求背景、目标用户、核心功能点、
        业务流程、异常处理、验收标准的 Markdown 文档。

        Args:
            feature_name: 功能名称，例如 "智能报销审批"
            brief_description: 功能简要描述，
                例如 "通过 AI 自动审核报销单据，减少人工审批工作量"

        Returns:
            格式化的 Markdown PRD 文档字符串
        """
        logger.info("执行工具 generate_prd_document，feature_name=%s", feature_name)
        try:
            system_prompt = (
                "你是一位资深产品经理，请根据以下信息撰写一份结构化的产品需求文档（PRD）。\n"
                "请按以下结构输出 Markdown 格式的 PRD：\n"
                "1. 需求背景\n"
                "2. 目标用户\n"
                "3. 核心功能点\n"
                "4. 业务流程\n"
                "5. 异常处理\n"
                "6. 验收标准"
            )
            user_content = f"功能名称：{feature_name}\n简要描述：{brief_description}"
            result = _invoke_llm(llm, system_prompt, user_content)
        except Exception as exc:
            result = f"PRD 文档生成失败：{exc}"
        logger.info("工具返回结果：%s...", result[:100])
        return result

    return generate_prd_document


def make_generate_flowchart_code_tool(llm: BaseChatModel) -> BaseTool:
    @tool
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
            result = _invoke_llm(llm, system_prompt, f"业务流程描述：{process_description}")
        except Exception as exc:
            result = f"流程图代码生成失败：{exc}"
        logger.info("工具返回结果：%s...", result[:100])
        return result

    return generate_flowchart_code


def make_generate_html_prototype_tool(llm: BaseChatModel) -> BaseTool:
    @tool
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
        logger.info("执行工具 generate_html_prototype，page_description=%s", page_description)
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

    return generate_html_prototype
