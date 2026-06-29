import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import jieba
import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.core.enums import DocumentFormat, ModelProvider
from app.core.logging_config import log_function_call, log_rag_query
from app.exceptions import ConfigurationError, DocumentLoadError, VectorStoreError
from app.rag.loader import load_single_file, split_documents

if TYPE_CHECKING:
    from app.rag.knowledge_graph import KnowledgeGraphManager

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
    create_start = time.monotonic()

    # Embedding 模型选择策略：
    # 1. 优先使用 NVIDIA Embedding（最稳定，维度 1024）
    # 2. NVIDIA 不可用时，尝试智谱 Embedding
    # 3. 都不可用时，回退到 HuggingFace 本地模型

    # 优先使用 NVIDIAEmbeddings 云端 API
    if settings.NVIDIA_API_KEY:
        try:
            logger.info("初始化 NVIDIA Embedding 模型：%s", settings.NIM_EMBEDDING_MODEL)
            logger.info("NIM 服务地址：%s", settings.NIM_BASE_URL)
            nvidia_embeddings = NVIDIAEmbeddings(
                model=settings.NIM_EMBEDDING_MODEL,
                api_key=settings.NVIDIA_API_KEY,
                base_url=settings.NIM_BASE_URL,
            )
            logger.info("NVIDIA Embedding 模型初始化成功")
            duration_ms = (time.monotonic() - create_start) * 1000
            log_function_call(
                func_name="create_embeddings",
                duration_ms=duration_ms,
                result_summary=f"云端API模式，模型={settings.NIM_EMBEDDING_MODEL}",
            )
            return nvidia_embeddings
        except Exception as exc:
            duration_ms = (time.monotonic() - create_start) * 1000
            logger.warning(
                "NVIDIA Embedding 初始化失败（%s），耗时=%.1fms，尝试其他后端",
                exc,
                duration_ms,
            )

    # 尝试智谱 Embedding
    if settings.ZHIPU_API_KEY:
        try:
            from langchain_openai import OpenAIEmbeddings

            logger.info("初始化智谱 Embedding 模型：embedding-3")
            logger.info("智谱服务地址：%s", settings.ZHIPU_BASE_URL)
            zhipu_embeddings = OpenAIEmbeddings(
                model="embedding-3",
                api_key=settings.ZHIPU_API_KEY,
                base_url=settings.ZHIPU_BASE_URL,
            )
            logger.info("智谱 Embedding 模型初始化成功")
            duration_ms = (time.monotonic() - create_start) * 1000
            log_function_call(
                func_name="create_embeddings",
                duration_ms=duration_ms,
                result_summary="智谱API模式，模型=embedding-3",
            )
            return zhipu_embeddings
        except Exception as exc:
            duration_ms = (time.monotonic() - create_start) * 1000
            logger.warning(
                "智谱 Embedding 初始化失败（%s），耗时=%.1fms，回退到本地模型",
                exc,
                duration_ms,
            )

    # 最终回退：HuggingFace 本地模型
    return _create_hf_embeddings(create_start)


def _create_hf_embeddings(start_time: float) -> Embeddings:
    """创建 HuggingFace 本地 Embedding 模型（回退方案）

    使用 BAAI/bge-small-zh-v1.5 中文模型，无需 API Key，离线可用。
    首次使用会自动从 HuggingFace Hub 下载模型（约 100MB）。

    Args:
        start_time: 计时起始时间

    Returns:
        HuggingFace Embeddings 实例
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        hf_model_name = "BAAI/bge-small-zh-v1.5"
        logger.info("初始化 HuggingFace 本地 Embedding 模型：%s", hf_model_name)
        hf_embeddings = HuggingFaceEmbeddings(
            model_name=hf_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        duration_ms = (time.monotonic() - start_time) * 1000
        logger.info("HuggingFace 本地 Embedding 模型初始化成功，耗时=%.1fms", duration_ms)
        log_function_call(
            func_name="create_embeddings",
            duration_ms=duration_ms,
            result_summary=f"本地回退模式，模型={hf_model_name}",
        )
        return hf_embeddings
    except Exception as hf_exc:
        duration_ms = (time.monotonic() - start_time) * 1000
        logger.error(
            "HuggingFace 本地 Embedding 也初始化失败（%s），耗时=%.1fms",
            hf_exc,
            duration_ms,
        )
        raise ConfigurationError(
            "所有 Embedding 后端均初始化失败，请检查网络连接或安装 sentence-transformers",
            details=str(hf_exc),
        ) from hf_exc


def _vectorstore_exists(chroma_db_dir: Path) -> bool:
    return chroma_db_dir.exists() and any(chroma_db_dir.iterdir())


def _vectorstore_count(vectorstore: Chroma) -> int:
    try:
        return vectorstore._collection.count()
    except Exception as exc:
        logger.warning("无法获取向量库文档数量：%s", exc)
        return 0


def build_or_load_vectorstore(embeddings: Embeddings | None = None) -> Chroma:
    build_start = time.monotonic()

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
        duration_ms = (time.monotonic() - build_start) * 1000
        log_function_call(
            func_name="build_or_load_vectorstore",
            duration_ms=duration_ms,
            result_summary=f"加载已有向量库，文档数={count}",
        )
        return vectorstore

    logger.info("未找到已有向量库，开始加载文档并入库...")

    if not knowledge_dir.exists():
        logger.info("创建知识库目录：%s", knowledge_dir)
        knowledge_dir.mkdir(parents=True, exist_ok=True)

    documents: list[Document] = []
    supported_suffixes: Final[set[DocumentFormat]] = set(DocumentFormat)

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
        duration_ms = (time.monotonic() - build_start) * 1000
        log_function_call(
            func_name="build_or_load_vectorstore",
            duration_ms=duration_ms,
            result_summary="创建空向量库（无文档）",
        )
        return Chroma(
            persist_directory=str(chroma_db_dir),
            embedding_function=embeddings,
        )

    chunks = split_documents(documents)

    if not chunks:
        logger.warning("文档切片后为空，创建空向量库")
        duration_ms = (time.monotonic() - build_start) * 1000
        log_function_call(
            func_name="build_or_load_vectorstore",
            duration_ms=duration_ms,
            result_summary="创建空向量库（切片为空）",
        )
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
    duration_ms = (time.monotonic() - build_start) * 1000
    logger.info(
        "向量库构建完成：%s，共 %d 个文本块，耗时=%.1fms", chroma_db_dir, count, duration_ms
    )
    log_function_call(
        func_name="build_or_load_vectorstore",
        duration_ms=duration_ms,
        result_summary=f"构建完成，文本块数={count}",
    )
    return vectorstore


def add_document_to_vectorstore(
    vectorstore: Chroma,
    file_path: Path,
    department: str = "通用",
    kg_manager: "KnowledgeGraphManager | None" = None,
) -> int:
    add_start = time.monotonic()
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
        duration_ms = (time.monotonic() - add_start) * 1000
        log_function_call(
            func_name="add_document_to_vectorstore",
            duration_ms=duration_ms,
            result_summary=f"文件={file_path.name}，文档加载后为空",
        )
        return 0

    chunks = split_documents(documents)
    if not chunks:
        logger.warning("文档切片后为空")
        duration_ms = (time.monotonic() - add_start) * 1000
        log_function_call(
            func_name="add_document_to_vectorstore",
            duration_ms=duration_ms,
            result_summary=f"文件={file_path.name}，切片后为空",
        )
        return 0

    for chunk in chunks:
        chunk.metadata["department"] = department

    # 知识图谱三元组提取：对每个 chunk 调用 kg_manager.extract_triplets()
    if kg_manager is not None:
        try:
            from langchain_core.language_models import BaseChatModel

            from app.core.dependencies import _create_raw_llm

            llm: BaseChatModel = _create_raw_llm()
            all_triplets: list[dict[str, str]] = []
            for chunk in chunks:
                chunk_text = chunk.page_content
                if chunk_text and chunk_text.strip():
                    triplets = kg_manager.extract_triplets(text=chunk_text, llm=llm)
                    all_triplets.extend(triplets)

            if all_triplets:
                added = kg_manager.add_triplets(all_triplets, source=file_path.name)
                logger.info(
                    "知识图谱三元组提取完成，文件=%s，提取=%d条，入库=%d条",
                    file_path.name,
                    len(all_triplets),
                    added,
                )
                # 持久化图谱
                kg_manager.persist()
        except Exception as exc:
            logger.warning(
                "知识图谱三元组提取失败，不影响文档入库主流程：%s（异常类型=%s）",
                exc,
                type(exc).__name__,
            )

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
    duration_ms = (time.monotonic() - add_start) * 1000
    logger.info("文档添加完成，当前向量库共有 %d 个文本块，耗时=%.1fms", count, duration_ms)
    log_function_call(
        func_name="add_document_to_vectorstore",
        duration_ms=duration_ms,
        result_summary=f"文件={file_path.name}，新增={len(chunks)}块，总计={count}块",
    )
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

    search_start = time.monotonic()
    try:
        docs = vectorstore.similarity_search(query, k=top_k)
    except Exception as exc:
        raise VectorStoreError(
            "向量检索失败",
            details=str(exc),
        ) from exc

    duration_ms = (time.monotonic() - search_start) * 1000
    logger.info("检索到 %d 个文本块，耗时=%.1fms", len(docs), duration_ms)

    # 提取相似度分数（如果可用）
    scores: list[float] = []
    for doc in docs:
        score = doc.metadata.get("score") or doc.metadata.get("relevance_score")
        if score is not None:
            scores.append(float(score))

    log_rag_query(
        query=query,
        top_k=top_k,
        hit_count=len(docs),
        duration_ms=duration_ms,
        scores=scores if scores else None,
    )

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
    hybrid_start = time.monotonic()

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

    duration_ms = (time.monotonic() - hybrid_start) * 1000
    top_rrf_scores = [round(rrf_scores[k], 4) for k in sorted_keys]

    logger.info(
        "RRF 融合后返回 %d 个文本块，耗时=%.1fms，向量命中=%d，BM25命中=%d",
        len(results),
        duration_ms,
        len(vector_results),
        len(bm25_results),
    )

    log_rag_query(
        query=query,
        top_k=top_k,
        hit_count=len(results),
        duration_ms=duration_ms,
        scores=top_rrf_scores,
    )

    return results
