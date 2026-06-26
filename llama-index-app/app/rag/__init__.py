"""
RAG 模块包：LlamaIndex 文档加载和索引管理。

提供文档加载（loader）和索引存储/混合检索（index_store）功能。
"""

from app.rag.index_store import (
    add_documents_to_index,
    build_or_load_index,
    clear_index,
    hybrid_retrieve,
    search_in_index,
)
from app.rag.loader import SUPPORTED_FORMATS, load_directory, load_single_file

__all__ = [
    "SUPPORTED_FORMATS",
    "add_documents_to_index",
    "build_or_load_index",
    "clear_index",
    "hybrid_retrieve",
    "load_directory",
    "load_single_file",
    "search_in_index",
]
