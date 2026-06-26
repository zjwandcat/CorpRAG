"""LlamaIndex 索引存储与混合检索模块

实现 VectorStoreIndex + ChromaDB + BM25 混合检索，
集成 NVIDIA Blueprints 的 RAG 优化模板（QueryFusionRetriever + reciprocal_rerank）。

关键组件：
- ChromaVectorStore：LlamaIndex 对 ChromaDB 的封装
- SentenceSplitter：LlamaIndex 的文档切片器，
  替代 LangChain 的 RecursiveCharacterTextSplitter
- BM25Retriever：关键词检索
- QueryFusionRetriever：RRF 融合检索（NVIDIA Blueprints 优化方案）
"""

import logging
from pathlib import Path


import chromadb
from llama_index.core import Settings as LlamaSettings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.core.config import settings
from app.exceptions import VectorStoreError
from app.rag.loader import load_directory

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


def configure_embed_model() -> None:
    """配置 LlamaIndex 全局 Embedding 模型为 NVIDIA NIM。

    LlamaIndex 默认使用 OpenAI Embedding，需要显式配置为 NVIDIA Embedding。
    此函数在首次构建索引前调用，确保 embed_model 正确初始化。
    必须在应用启动时调用一次。
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
            "NVIDIA Embedding 配置失败：%s，尝试 HuggingFace 本地 Embedding",
            exc,
        )
        try:
            import os

            # 国内环境使用 HuggingFace 镜像站，避免连接超时
            if not os.getenv("HF_ENDPOINT"):
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            LlamaSettings.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-zh-v1.5"
            )
            _embed_model_configured = True
            logger.info("已配置 HuggingFace 本地 Embedding（BAAI/bge-small-zh-v1.5）")
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


def build_or_load_index() -> VectorStoreIndex:
    """构建或加载 LlamaIndex VectorStoreIndex。

    如果 chroma_db_li 目录已有数据，则加载已有索引；
    否则从知识库目录加载文档并构建新索引。

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
            logger.info("已有索引加载完成")
            return index

        # 加载文档
        knowledge_dir = Path(settings.KNOWLEDGE_DIR)
        if not knowledge_dir.exists():
            logger.warning("知识库目录不存在：%s，创建空索引", knowledge_dir)
            knowledge_dir.mkdir(parents=True, exist_ok=True)

        documents = load_directory(knowledge_dir)

        if not documents:
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
        logger.info("正在构建索引，共 %d 个文档...", len(documents))
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[splitter],
            show_progress=True,
        )
        logger.info("索引构建完成，共 %d 个文档", len(documents))
        return index

    except Exception as exc:
        raise VectorStoreError(
            "构建/加载索引失败",
            details=str(exc),
        ) from exc


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
    """LlamaIndex 混合检索：向量 + BM25 + RRF 融合。

    集成 NVIDIA Blueprints 的 RAG 优化模板，使用 QueryFusionRetriever
    的 reciprocal_rerank 模式实现 RRF 融合排序，提升企业级检索性能。

    检索流程：
    1. 向量检索器（VectorStoreIndex.as_retriever）获取 top-5 候选
    2. BM25 检索器获取 top-5 候选
    3. QueryFusionRetriever 以 reciprocal_rerank 模式融合两路结果
    4. 按部门过滤后返回 top_k 结果

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
