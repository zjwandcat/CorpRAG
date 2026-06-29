"""LlamaIndex 索引存储与混合检索模块

实现 VectorStoreIndex + ChromaDB + BM25 混合检索，
集成 NVIDIA Blueprints 的 RAG 优化模板（QueryFusionRetriever + reciprocal_rerank），
并支持 Reranker 精排（SentenceTransformerRerank / FlagEmbeddingReranker）。

关键组件：
- ChromaVectorStore：LlamaIndex 对 ChromaDB 的封装
- SentenceSplitter：LlamaIndex 的文档切片器，
  替代 LangChain 的 RecursiveCharacterTextSplitter
- BM25Retriever：关键词检索
- QueryFusionRetriever：RRF 融合检索（NVIDIA Blueprints 优化方案）
- SentenceTransformerRerank / FlagEmbeddingReranker：Reranker 精排
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import chromadb
from llama_index.core import Settings as LlamaSettings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import Document, NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.core.config import settings
from app.exceptions import VectorStoreError
from app.rag.loader import load_directory, load_single_file

if TYPE_CHECKING:
    from llama_index.core.postprocessor.types import BaseNodePostprocessor

logger = logging.getLogger(__name__)

__all__ = [
    "add_documents_to_index",
    "build_or_load_index",
    "clear_index",
    "configure_embed_model",
    "get_index_count",
    "hybrid_retrieve",
    "search_in_index",
]

# 标记是否已配置 Embedding 模型
_embed_model_configured: bool = False

# 缓存 Reranker 实例（None 表示尚未初始化，False 表示不可用）
_reranker_instance: "BaseNodePostprocessor | bool | None" = None


def _create_reranker() -> "BaseNodePostprocessor | None":
    """创建 Reranker 精排器实例。

    优先使用 FlagEmbeddingReranker（BAAI/bge-reranker 系列，中文效果好），
    不可用时回退到 SentenceTransformerRerank，
    均不可用时返回 None（降级到原始排序）。

    Reranker 初始化失败不会影响系统正常启动，
    仅记录警告日志并降级到无精排模式。

    Returns:
        BaseNodePostprocessor 实例，或 None（降级模式）
    """
    global _reranker_instance

    # 已初始化过：返回缓存结果
    if _reranker_instance is not None:
        if _reranker_instance is False:
            return None
        return _reranker_instance

    # 配置关闭 Reranker
    if not settings.RERANKER_ENABLED:
        logger.info("Reranker 已通过配置禁用（RERANKER_ENABLED=false）")
        _reranker_instance = False
        return None

    reranker_model = settings.RERANKER_MODEL
    reranker_top_n = settings.RERANKER_TOP_N

    # 尝试 FlagEmbeddingReranker（推荐，中文 rerank 效果更好）
    try:
        from llama_index.postprocessor.flag_embedding_reranker import (
            FlagEmbeddingReranker,
        )

        reranker = FlagEmbeddingReranker(
            model=reranker_model,
            top_n=reranker_top_n,
            use_fp16=True,
        )
        _reranker_instance = reranker
        logger.info(
            "Reranker 初始化完成（FlagEmbeddingReranker），模型：%s，top_n=%d",
            reranker_model,
            reranker_top_n,
        )
        return reranker
    except ImportError:
        logger.debug("FlagEmbeddingReranker 不可用，尝试 SentenceTransformerRerank")
    except Exception as exc:
        logger.warning(
            "FlagEmbeddingReranker 初始化失败：%s，尝试 SentenceTransformerRerank",
            exc,
        )

    # 回退到 SentenceTransformerRerank
    try:
        from llama_index.postprocessor.sbert import SentenceTransformerRerank

        reranker = SentenceTransformerRerank(
            model=reranker_model,
            top_n=reranker_top_n,
        )
        _reranker_instance = reranker
        logger.info(
            "Reranker 初始化完成（SentenceTransformerRerank），模型：%s，top_n=%d",
            reranker_model,
            reranker_top_n,
        )
        return reranker
    except ImportError:
        logger.warning(
            "SentenceTransformerRerank 也不可用，"
            "请安装 llama-index-postprocessor-flag-embedding-reranker "
            "或 llama-index-postprocessor-sbert"
        )
    except Exception as exc:
        logger.warning("SentenceTransformerRerank 初始化失败：%s", exc)

    # 全部失败：降级模式
    logger.warning("Reranker 不可用，将使用原始排序（降级模式）")
    _reranker_instance = False
    return None


def configure_embed_model() -> None:
    """配置 LlamaIndex 全局 Embedding 模型。

    优先使用 NVIDIA NIM Embedding（云端 API），失败时回退到本地 HuggingFace Embedding（CPU 模式）。
    本地 GPU 仅加速预处理环节（OCR、文本清洗、向量检索），不用于 Embedding 生成。
    """
    global _embed_model_configured
    if _embed_model_configured:
        return

    try:
        from llama_index.embeddings.nvidia import NVIDIAEmbedding

        logger.info("配置 NVIDIA Embedding 模型：%s", settings.NIM_EMBEDDING_MODEL)
        embed_model = NVIDIAEmbedding(
            model=settings.NIM_EMBEDDING_MODEL,
            api_key=settings.NVIDIA_API_KEY,
            base_url=settings.NIM_BASE_URL,
        )
        LlamaSettings.embed_model = embed_model
        _embed_model_configured = True
        logger.info("NVIDIA Embedding 模型配置完成")
    except Exception as exc:
        logger.warning(
            "NVIDIA Embedding 配置失败：%s，尝试 HuggingFace 本地 Embedding（CPU 模式）",
            exc,
        )
        try:
            import os

            # 国内环境使用 HuggingFace 镜像站，避免连接超时
            if not os.getenv("HF_ENDPOINT"):
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            LlamaSettings.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-zh-v1.5",
            )
            _embed_model_configured = True
            logger.info("已配置 HuggingFace 本地 Embedding（BAAI/bge-small-zh-v1.5，CPU 模式）")
        except Exception as exc2:
            logger.error(
                "无法配置任何 Embedding 模型：%s，请安装 llama-index-embeddings-nvidia",
                exc2,
            )


def _create_chroma_store() -> tuple[chromadb.PersistentClient, ChromaVectorStore]:
    """创建 ChromaDB 向量存储。

    使用 LlamaIndex 专用目录 chroma_db_li，与 LangChain 版本的 chroma_db 隔离。

    Returns:
        (chromadb.PersistentClient, ChromaVectorStore) 元组
    """
    chroma_db_dir = Path(settings.CHROMA_DB_DIR)
    chroma_db_dir.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(chroma_db_dir))
    chroma_collection = chroma_client.get_or_create_collection(
        "llamaindex_default",
        metadata={"hnsw:space": "cosine"},
    )
    chroma_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return chroma_client, chroma_store


def _create_splitter() -> SentenceSplitter:
    """创建文档切片器。

    使用 LlamaIndex 的 SentenceSplitter
    替代 LangChain 的 RecursiveCharacterTextSplitter，
    支持中文分隔符优化。

    Returns:
        SentenceSplitter 实例
    """
    return SentenceSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separator="\n",
        paragraph_separator="\n\n",
        secondary_chunking_regex="[^，。！？]+[，。！？]?",
    )


def _get_all_data_dirs() -> list[Path]:
    """获取所有数据目录列表。

    优先加载项目根目录的 data/，同时兼容 financial-agent-api/data/。

    Returns:
        数据目录路径列表
    """
    data_dirs: list[Path] = []

    # 主知识库目录（项目根目录的 data/）
    knowledge_dir = Path(settings.KNOWLEDGE_DIR)
    if knowledge_dir.exists():
        data_dirs.append(knowledge_dir)

    # 研报目录
    reports_dir = Path(settings.REPORTS_DIR)
    if reports_dir.exists() and reports_dir not in data_dirs:
        data_dirs.append(reports_dir)

    # 兼容 financial-agent-api 的 knowledge_base 目录
    shared_kb = Path(settings._SHARED_DATA_ROOT) / "knowledge_base"
    if shared_kb.exists() and shared_kb not in data_dirs:
        data_dirs.append(shared_kb)

    return data_dirs


def _get_indexed_sources(index: VectorStoreIndex) -> set[str]:
    """获取已在索引中的文件名集合。

    Args:
        index: VectorStoreIndex 实例

    Returns:
        已索引的文件名集合
    """
    try:
        _, chroma_store = _create_chroma_store()
        all_docs = chroma_store._collection.get(include=["metadatas"])
        sources: set[str] = set()
        for meta in (all_docs.get("metadatas") or []):
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sources
    except Exception as exc:
        logger.warning("无法获取已索引文件列表：%s", exc)
        return set()


def build_or_load_index() -> VectorStoreIndex:
    """构建或加载 LlamaIndex VectorStoreIndex。

    如果 chroma_db_li 目录已有数据，则加载已有索引，
    并检查是否有新文件需要增量索引；
    否则从所有数据目录加载文档并构建新索引。

    Returns:
        VectorStoreIndex 实例

    Raises:
        VectorStoreError: 向量库操作失败
    """
    logger.info("开始构建/加载 VectorStoreIndex...")

    # 确保 Embedding 模型已配置（必须在构建索引前调用）
    configure_embed_model()

    try:
        _, chroma_store = _create_chroma_store()
        storage_context = StorageContext.from_defaults(vector_store=chroma_store)

        # 检查是否已有数据
        existing_count = chroma_store._collection.count()
        if existing_count > 0:
            logger.info("发现已有向量库数据（%d 条），加载已有索引", existing_count)
            splitter = _create_splitter()
            index = VectorStoreIndex.from_vector_store(
                vector_store=chroma_store,
                storage_context=storage_context,
                transformations=[splitter],
            )

            # 检查是否有新文件需要增量索引
            _check_and_index_new_files(index)

            logger.info("已有索引加载完成")
            return index

        # 从所有数据目录加载文档
        all_documents: list[Document] = []
        data_dirs = _get_all_data_dirs()

        for data_dir in data_dirs:
            logger.info("从目录加载文档：%s", data_dir)
            docs = load_directory(data_dir)
            all_documents.extend(docs)

        if not all_documents:
            logger.warning("没有加载到任何文档，创建空索引")
            splitter = _create_splitter()
            index = VectorStoreIndex.from_vector_store(
                vector_store=chroma_store,
                storage_context=storage_context,
                transformations=[splitter],
            )
            return index

        # 构建索引
        splitter = _create_splitter()
        logger.info("正在构建索引，共 %d 个文档...", len(all_documents))
        index = VectorStoreIndex.from_documents(
            all_documents,
            storage_context=storage_context,
            transformations=[splitter],
            show_progress=True,
        )
        logger.info("索引构建完成，共 %d 个文档", len(all_documents))
        return index

    except Exception as exc:
        raise VectorStoreError(
            "构建/加载索引失败",
            details=str(exc),
        ) from exc


def _check_and_index_new_files(index: VectorStoreIndex) -> None:
    """检查数据目录中是否有新文件，并增量索引。

    对比已索引的文件名和数据目录中的文件名，
    将新增文件增量添加到索引中。

    Args:
        index: VectorStoreIndex 实例
    """
    indexed_sources = _get_indexed_sources(index)
    data_dirs = _get_all_data_dirs()

    new_documents: list[Document] = []
    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        for file_path in sorted(data_dir.rglob("*")):
            if file_path.is_dir():
                continue
            if file_path.suffix.lower() not in {".pdf", ".txt", ".docx", ".md", ".csv"}:
                continue
            if file_path.name not in indexed_sources:
                logger.info("发现新文件：%s，将增量索引", file_path.name)
                try:
                    docs = load_single_file(file_path)
                    new_documents.extend(docs)
                except Exception as exc:
                    logger.warning("加载新文件 %s 失败：%s", file_path.name, exc)

    if new_documents:
        logger.info("发现 %d 个新文档，开始增量索引...", len(new_documents))
        for doc in new_documents:
            try:
                index.insert(doc)
            except Exception as exc:
                logger.warning("增量索引文档失败：%s", exc)
        logger.info("增量索引完成，共添加 %d 个文档", len(new_documents))
    else:
        logger.info("没有发现新文件需要增量索引")


def add_documents_to_index(
    index: VectorStoreIndex,
    file_path: Path,
    department: str = "通用",
) -> int:
    """添加文档到索引。

    Args:
        index: VectorStoreIndex 实例
        file_path: 文件路径
        department: 文档所属部门

    Returns:
        添加的文档数量

    Raises:
        VectorStoreError: 添加文档失败
    """
    from app.rag.loader import load_single_file

    logger.info("正在添加文档到索引：%s，部门：%s", file_path.name, department)

    try:
        documents = load_single_file(file_path, department)
    except Exception as exc:
        raise VectorStoreError(
            f"加载文档 {file_path.name} 失败",
            details=str(exc),
        ) from exc

    if not documents:
        logger.warning("文档加载后为空")
        return 0

    try:
        for doc in documents:
            index.insert(doc)
        logger.info("文档添加完成，共 %d 个文档", len(documents))
        return len(documents)
    except Exception as exc:
        raise VectorStoreError(
            "添加文档到索引失败",
            details=str(exc),
        ) from exc


def clear_index() -> None:
    """清空索引。

    删除 ChromaDB 集合中的所有文档数据。

    Raises:
        VectorStoreError: 清空索引失败
    """
    logger.info("正在清空索引...")
    try:
        _, chroma_store = _create_chroma_store()
        # 获取所有文档ID并删除
        all_ids = chroma_store._collection.get()["ids"]
        if all_ids:
            chroma_store._collection.delete(ids=all_ids)
        logger.info("索引已清空")
    except Exception as exc:
        raise VectorStoreError(
            "清空索引失败",
            details=str(exc),
        ) from exc


def get_index_count(index: VectorStoreIndex) -> int:
    """获取索引中的文档数量。

    Args:
        index: VectorStoreIndex 实例

    Returns:
        索引中的文档数量
    """
    try:
        _, chroma_store = _create_chroma_store()
        return chroma_store._collection.count()
    except Exception as exc:
        logger.warning("无法获取索引文档数量：%s", exc)
        return 0


def hybrid_retrieve(
    query: str,
    index: VectorStoreIndex,
    department: str = "通用",
    top_k: int = 3,
) -> list[NodeWithScore]:
    """LlamaIndex 混合检索：向量 + BM25 + RRF 融合 + Reranker 精排。

    集成 NVIDIA Blueprints 的 RAG 优化模板，使用 QueryFusionRetriever
    的 reciprocal_rerank 模式实现 RRF 融合排序，提升企业级检索性能。
    融合后可选 Reranker 精排，进一步提升检索质量。

    检索流程：
    1. 向量检索器（VectorStoreIndex.as_retriever）获取 top-5 候选
    2. BM25 检索器获取 top-5 候选
    3. QueryFusionRetriever 以 reciprocal_rerank 模式融合两路结果
    4. Reranker 精排（可选，配置控制，不可用时降级到原始排序）
    5. 按部门过滤后返回 top_k 结果

    Args:
        query: 检索查询
        index: VectorStoreIndex 实例
        department: 部门过滤
        top_k: 返回结果数量

    Returns:
        检索结果列表（NodeWithScore）

    Raises:
        VectorStoreError: 检索完全失败
    """
    logger.info("混合检索：query=%s, department=%s, top_k=%d", query, department, top_k)

    try:
        # 向量检索器
        vector_retriever = index.as_retriever(similarity_top_k=5)

        # BM25 检索器
        use_bm25 = False
        try:
            nodes = index.docstore.get_nodes(list(index.docstore.docs.keys()))
            if nodes:
                bm25_retriever = BM25Retriever.from_defaults(
                    nodes=nodes,
                    similarity_top_k=5,
                )
                use_bm25 = True
            else:
                logger.warning("docstore 中无节点，跳过 BM25 检索")
        except Exception as exc:
            logger.warning("BM25 检索器初始化失败，仅使用向量检索：%s", exc)

        if use_bm25:
            # RRF 融合检索器（NVIDIA Blueprints 优化：使用 reciprocal_rerank 模式）
            fusion_retriever = QueryFusionRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                num_queries=1,  # 不改写查询
                similarity_top_k=top_k,
                mode="reciprocal_rerank",  # RRF 模式（Blueprints 优化）
            )
            results = fusion_retriever.retrieve(query)
        else:
            results = vector_retriever.retrieve(query)

        # Reranker 精排（可选步骤，不可用时降级到原始排序）
        reranker = _create_reranker()
        if reranker is not None:
            try:
                results = reranker.postprocess_nodes(
                    nodes=results,
                    query_str=query,
                )
                logger.info("Reranker 精排完成，返回 %d 个结果", len(results))
            except Exception as exc:
                logger.warning("Reranker 精排失败，使用原始排序：%s", exc)

        # 部门过滤（保留目标部门和通用部门的结果）
        if department != "通用":
            results = [
                r
                for r in results
                if r.node.metadata.get("department", "通用") in (department, "通用")
            ]

        logger.info("混合检索返回 %d 个结果", len(results))
        return results[:top_k]

    except Exception as exc:
        logger.warning("混合检索失败，回退到向量检索：%s", exc)
        # 回退到纯向量检索
        try:
            vector_retriever = index.as_retriever(similarity_top_k=top_k)
            results = vector_retriever.retrieve(query)
            logger.info("向量检索返回 %d 个结果", len(results))
            return results
        except Exception as fallback_exc:
            raise VectorStoreError(
                "检索失败",
                details=str(fallback_exc),
            ) from fallback_exc


def search_in_index(
    index: VectorStoreIndex,
    query: str,
    top_k: int | None = None,
) -> list[NodeWithScore]:
    """纯向量检索。

    Args:
        index: VectorStoreIndex 实例
        query: 检索查询
        top_k: 返回结果数量，默认使用 settings.TOP_K

    Returns:
        检索结果列表（NodeWithScore）

    Raises:
        VectorStoreError: 向量检索失败
    """
    if top_k is None:
        top_k = settings.TOP_K

    logger.info("向量检索：%s，top_k=%d", query, top_k)

    try:
        retriever = index.as_retriever(similarity_top_k=top_k)
        results = retriever.retrieve(query)
        logger.info("检索到 %d 个结果", len(results))
        return results
    except Exception as exc:
        raise VectorStoreError(
            "向量检索失败",
            details=str(exc),
        ) from exc
