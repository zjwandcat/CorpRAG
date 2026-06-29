import logging
import os
import time
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.md import partition_md

from app.core.config import settings
from app.core.enums import DocumentFormat
from app.core.logging_config import log_function_call
from app.exceptions import DocumentLoadError, UnsupportedFormatError

logger = logging.getLogger(__name__)


def load_single_file(file_path: Path, department: str | None = None) -> list[Document]:
    if not file_path.exists():
        raise DocumentLoadError(
            message=f"文件不存在：{file_path}", details=f"路径: {file_path.absolute()}"
        )

    load_start = time.monotonic()
    file_size = 0
    try:
        file_size = os.path.getsize(file_path)
    except OSError:
        file_size = -1

    suffix = file_path.suffix.lower()

    match suffix:
        case DocumentFormat.PDF:
            pdf_loader = PyPDFLoader(str(file_path))
            documents = pdf_loader.load()
        case DocumentFormat.TXT:
            txt_loader = TextLoader(str(file_path), encoding="utf-8")
            documents = txt_loader.load()
        case DocumentFormat.DOCX:
            docx_loader = Docx2txtLoader(str(file_path))
            documents = docx_loader.load()
        case DocumentFormat.MD:
            elements = partition_md(filename=str(file_path))
            documents = [
                Document(page_content=str(el), metadata={"source": file_path.name})
                for el in elements
                if str(el).strip()
            ]
        case _:
            raise UnsupportedFormatError(
                format=suffix, details="仅支持 .pdf、.txt、.docx 和 .md 格式"
            )

    if department is None:
        department = _infer_department_from_path(file_path)

    for doc in documents:
        doc.metadata["source"] = file_path.name
        doc.metadata["department"] = department

    duration_ms = (time.monotonic() - load_start) * 1000
    file_size_kb = round(file_size / 1024, 1) if file_size >= 0 else -1
    logger.info(
        "加载文件：%s，部门：%s，文档数：%d，文件大小：%.1fKB，耗时=%.1fms",
        file_path.name,
        department,
        len(documents),
        file_size_kb,
        duration_ms,
    )
    log_function_call(
        func_name="load_single_file",
        duration_ms=duration_ms,
        result_summary=f"文件={file_path.name}，部门={department}，文档数={len(documents)}，大小={file_size_kb}KB",
    )
    return documents


def _infer_department_from_path(file_path: Path) -> str:
    path_parts = [p.lower() for p in file_path.parts]

    if any("人事部" in p or "hr" in p for p in path_parts):
        return "人事部"
    if any("研发" in p or "rd" in p for p in path_parts):
        return "研发中心"
    if any("财务" in p or "finance" in p for p in path_parts):
        return "财务部"

    return "通用"


def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    split_start = time.monotonic()

    if chunk_size is None:
        chunk_size = settings.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = settings.CHUNK_OVERLAP

    # 统计切片前的文档总字符数
    total_chars_before = sum(len(doc.page_content) for doc in documents)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # 统计切片后的文本块总字符数
    total_chars_after = sum(len(chunk.page_content) for chunk in chunks)
    duration_ms = (time.monotonic() - split_start) * 1000

    logger.info(
        "文档切片完成：%d 个文档 -> %d 个文本块，切片前总字符=%d，切片后总字符=%d，耗时=%.1fms",
        len(documents),
        len(chunks),
        total_chars_before,
        total_chars_after,
        duration_ms,
    )
    log_function_call(
        func_name="split_documents",
        duration_ms=duration_ms,
        result_summary=f"{len(documents)}个文档->{len(chunks)}个文本块，字符={total_chars_before}->{total_chars_after}",
    )
    return chunks


__all__ = ["_infer_department_from_path", "load_single_file", "split_documents"]
