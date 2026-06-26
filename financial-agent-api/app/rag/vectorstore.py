import logging
from pathlib import Path
from typing import Any, Final

import jieba
import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.core.enums import DocumentFormat, ModelProvider
from app.exceptions import DocumentLoadError, VectorStoreError
from app.rag.loader import load_single_file, split_documents

logger = logging.getLogger(__name__)

__all__ = [
    "add_document_to_vectorstore",
    "build_or_load_vectorstore",
    "clear_vectorstore",
    "create_embeddings",
    "hybrid_search",
    "search_in_vectorstore",
]

_bm25_cache: dict[str, Any] = {}


def create_embeddings() -> Embeddings:
    if settings.PROVIDER in (ModelProvider.XFYUN, ModelProvider.ZHIPU):
        logger.info("使用本地 HuggingFace Embedding（降级模式），模型：BAAI/bge-small-zh-v1.5")
        import os

        # 国内环境使用 HuggingFace 镜像站，避免连接超时
        if not os.getenv("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

    logger.info("初始化 NVIDIA Embedding 模型：%s", settings.NIM_EMBEDDING_MODEL)
    logger.info("NIM 服务地址：%s", settings.NIM_BASE_URL)
    return NVIDIAEmbeddings(
        model=settings.NIM_EMBEDDING_MODEL,
        api_key=settings.NVIDIA_API_KEY,
        base_url=settings.NIM_BASE_URL,
    )


def _vectorstore_exists(chroma_db_dir: Path) -> bool:
    return chroma_db_dir.exists() and any(chroma_db_dir.iterdir())


def _vectorstore_count(vectorstore: Chroma) -> int:
    try:
        return vectorstore._collection.count()
    except Exception as exc:
        logger.warning("无法获取向量库文档数量：%s", exc)
        return 0


def build_or_load_vectorstore(embeddings: Embeddings | None = None) -> Chroma:
    if embeddings is None:
        embeddings = create_embeddings()

    chroma_db_dir = Path(settings.CHROMA_DB_DIR)
    knowledge_dir = Path(settings.KNOWLEDGE_DIR)

    if _vectorstore_exists(chroma_db_dir):
        logger.info("发现已有向量库：%s", chroma_db_dir)
        vectorstore = Chroma(
            persist_directory=str(chroma_db_dir),
            embedding_function=embeddings,
        )
        count = _vectorstore_count(vectorstore)
        logger.info("向量库中的文档数量：%d", count)
        return vectorstore

    logger.info("未找到已有向量库，开始加载文档并入库...")

    if not knowledge_dir.exists():
        logger.info("创建知识库目录：%s", knowledge_dir)
        knowledge_dir.mkdir(parents=True, exist_ok=True)

    documents: list[Document] = []
    supported_suffixes: Final[set[DocumentFormat]] = {
        DocumentFormat.PDF,
        DocumentFormat.TXT,
    }

    for file_path in sorted(knowledge_dir.iterdir()):
        if file_path.suffix.lower() not in supported_suffixes:
            continue

        try:
            docs = load_single_file(file_path)
            documents.extend(docs)
        except DocumentLoadError:
            raise
        except Exception as exc:
            raise DocumentLoadError(
                f"加载文件 {file_path.name} 失败",
                details=str(exc),
            ) from exc

    if not documents:
        logger.warning("没有加载到任何文档，创建空向量库")
        return Chroma(
            persist_directory=str(chroma_db_dir),
            embedding_function=embeddings,
        )

    chunks = split_documents(documents)

    if not chunks:
        logger.warning("文档切片后为空，创建空向量库")
        return Chroma(
            persist_directory=str(chroma_db_dir),
            embedding_function=embeddings,
        )

    logger.info("正在构建向量库，共 %d 个文本块...", len(chunks))
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(chroma_db_dir),
        )
    except Exception as exc:
        raise VectorStoreError(
            "构建向量库失败",
            details=str(exc),
        ) from exc

    count = _vectorstore_count(vectorstore)
    logger.info("向量库构建完成：%s，共 %d 个文本块", chroma_db_dir, count)
    return vectorstore


def add_document_to_vectorstore(
    vectorstore: Chroma, file_path: Path, department: str = "通用"
) -> int:
    logger.info("正在添加文档到向量库：%s，部门：%s", file_path.name, department)

    try:
        documents = load_single_file(file_path)
    except DocumentLoadError:
        raise
    except Exception as exc:
        raise DocumentLoadError(
            f"加载文档 {file_path.name} 失败",
            details=str(exc),
        ) from exc

    if not documents:
        logger.warning("文档加载后为空")
        return 0

    chunks = split_documents(documents)
    if not chunks:
        logger.warning("文档切片后为空")
        return 0

    for chunk in chunks:
        chunk.metadata["department"] = department

    logger.info("正在添加 %d 个文本块到向量库...", len(chunks))
    try:
        vectorstore.add_documents(chunks)
    except Exception as exc:
        raise VectorStoreError(
            "添加文档到向量库失败",
            details=str(exc),
        ) from exc

    _bm25_cache.clear()
    count = _vectorstore_count(vectorstore)
    logger.info("文档添加完成，当前向量库共有 %d 个文本块", count)
    return len(chunks)


def clear_vectorstore(vectorstore: Chroma) -> None:
    logger.info("正在清空向量库...")
    try:
        vectorstore._collection.delete()
        _bm25_cache.clear()
        logger.info("向量库已清空")
    except Exception as exc:
        raise VectorStoreError(
            "清空向量库失败",
            details=str(exc),
        ) from exc


def search_in_vectorstore(
    vectorstore: Chroma,
    query: str,
    top_k: int | None = None,
) -> list[Document]:
    if top_k is None:
        top_k = settings.TOP_K

    logger.info("正在检索：%s，top_k=%d", query, top_k)

    try:
        docs = vectorstore.similarity_search(query, k=top_k)
    except Exception as exc:
        raise VectorStoreError(
            "向量检索失败",
            details=str(exc),
        ) from exc

    logger.info("检索到 %d 个文本块", len(docs))
    return docs


def _get_or_create_bm25(all_documents: list[str]) -> BM25Okapi:
    cache_key = str(hash(tuple(all_documents)))
    if cache_key in _bm25_cache:
        return _bm25_cache[cache_key]

    tokenized_docs = [list(jieba.cut(doc)) for doc in all_documents]
    bm25 = BM25Okapi(tokenized_docs)
    _bm25_cache.clear()
    _bm25_cache[cache_key] = bm25
    return bm25


def hybrid_search(
    query: str,
    department: str,
    vectorstore: Chroma,
    all_documents: list[str],
    top_k: int = 3,
) -> list[Document]:
    rrf_k: Final[int] = 60

    logger.info("混合检索：query=%s, department=%s, top_k=%d", query, department, top_k)

    vector_results: list[Document] = []
    try:
        search_kwargs: dict[str, Any] = {"k": 5}
        if department != "通用":
            search_kwargs["filter"] = {"department": department}
        vector_results = vectorstore.similarity_search(query, **search_kwargs)
        logger.info("向量检索命中 %d 个文本块", len(vector_results))
    except Exception as exc:
        logger.warning("向量检索失败：%s", exc)

    bm25_results: list[str] = []
    try:
        bm25 = _get_or_create_bm25(all_documents)
        bm25_scores = bm25.get_scores(list(jieba.cut(query)))
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:5]
        bm25_results = [all_documents[i] for i in bm25_top_indices if bm25_scores[i] > 0]
        logger.info("BM25 检索命中 %d 个文本块", len(bm25_results))
    except Exception as exc:
        logger.warning("BM25 检索失败：%s", exc)

    rrf_scores: dict[str, float] = {}
    doc_content_map: dict[str, Document] = {}

    for rank, doc in enumerate(vector_results):
        content_key = doc.page_content[:200]
        rrf_scores[content_key] = rrf_scores.get(content_key, 0.0) + 1.0 / (rrf_k + rank + 1)
        doc_content_map[content_key] = doc

    for rank, content in enumerate(bm25_results):
        content_key = content[:200]
        rrf_scores[content_key] = rrf_scores.get(content_key, 0.0) + 1.0 / (rrf_k + rank + 1)
        if content_key not in doc_content_map:
            doc_content_map[content_key] = Document(
                page_content=content, metadata={"source": "BM25检索", "department": department}
            )

    sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)[:top_k]
    results = [doc_content_map[k] for k in sorted_keys]

    logger.info("RRF 融合后返回 %d 个文本块", len(results))
    return results
