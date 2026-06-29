"""LlamaIndex 文档加载模块

使用 LlamaIndex 的文件加载器替代 LangChain 的加载器，
支持 PDF / TXT / DOCX / MD / CSV 五种格式的文档加载。

关键区别：
- LlamaIndex Document 来自 llama_index.core.schema.Document
- 加载器接口为 reader.load_data(file=file_path)，返回 list[Document]
- CSV 使用 PandasCSVReader 保留表格结构
"""

import logging
from pathlib import Path

from llama_index.core.schema import Document
from llama_index.readers.file import (
    DocxReader,
    FlatReader,
    MarkdownReader,
    PandasCSVReader,
    PDFReader,
)


from app.exceptions import DocumentLoadError, UnsupportedFormatError

logger = logging.getLogger(__name__)

# 支持的文件格式
SUPPORTED_FORMATS = {".pdf", ".txt", ".docx", ".md", ".csv"}


def load_single_file(file_path: Path, department: str | None = None) -> list[Document]:
    """加载单个文件，支持 PDF/TXT/DOCX/MD/CSV。

    使用 LlamaIndex 的文件加载器替代 LangChain 的加载器。

    Args:
        file_path: 文件路径
        department: 文档所属部门，为 None 时自动推断

    Returns:
        LlamaIndex Document 列表

    Raises:
        DocumentLoadError: 文档加载失败
        UnsupportedFormatError: 不支持的文件格式
    """
    if not file_path.exists():
        raise DocumentLoadError(
            message=f"文件不存在：{file_path}",
            details=f"路径: {file_path.absolute()}",
        )

    suffix = file_path.suffix.lower()

    match suffix:
        case ".pdf":
            reader = PDFReader()
            documents = reader.load_data(file=file_path)
        case ".txt":
            reader = FlatReader()
            documents = reader.load_data(file=file_path)
        case ".docx":
            reader = DocxReader()
            documents = reader.load_data(file=file_path)
        case ".md":
            reader = MarkdownReader()
            documents = reader.load_data(file=file_path)
        case ".csv":
            # CSV 特殊处理：使用 PandasCSVReader 保留表格结构
            reader = PandasCSVReader()
            documents = reader.load_data(file=file_path)
        case _:
            raise UnsupportedFormatError(
                format=suffix,
                details="仅支持 .pdf、.txt、.docx、.md 和 .csv 格式",
            )

    if department is None:
        department = _infer_department_from_path(file_path)

    # 添加元数据
    for doc in documents:
        doc.metadata["source"] = file_path.name
        doc.metadata["department"] = department
        doc.metadata["file_type"] = suffix

    logger.info(
        "加载文件：%s，部门：%s，文档数：%d",
        file_path.name,
        department,
        len(documents),
    )
    return documents


def _infer_department_from_path(file_path: Path) -> str:
    """从文件路径推断部门。"""
    path_parts = [p.lower() for p in file_path.parts]

    if any("人事部" in p or "hr" in p for p in path_parts):
        return "人事部"
    if any("研发" in p or "rd" in p for p in path_parts):
        return "研发中心"
    if any("财务" in p or "finance" in p for p in path_parts):
        return "财务部"

    return "通用"


def load_directory(directory: Path, department: str | None = None) -> list[Document]:
    """加载目录下所有支持的文件（递归扫描子目录）。

    使用 rglob 递归扫描目录及其子目录中的所有文件，
    确保 data/ 目录下的嵌套结构（如 data/reports/）也能被索引。

    Args:
        directory: 目录路径
        department: 文档所属部门

    Returns:
        LlamaIndex Document 列表
    """
    if not directory.exists():
        logger.warning("目录不存在：%s", directory)
        return []

    documents: list[Document] = []
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_dir():
            continue
        if file_path.suffix.lower() not in SUPPORTED_FORMATS:
            continue
        try:
            docs = load_single_file(file_path, department)
            documents.extend(docs)
        except (DocumentLoadError, UnsupportedFormatError) as exc:
            logger.warning("跳过文件 %s：%s", file_path.name, exc)
        except Exception as exc:
            logger.warning("加载文件 %s 失败：%s", file_path.name, exc)

    logger.info("从目录 %s 加载了 %d 个文档", directory, len(documents))
    return documents


__all__ = ["SUPPORTED_FORMATS", "load_directory", "load_single_file"]
