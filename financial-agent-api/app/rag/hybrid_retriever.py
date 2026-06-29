# app/rag/hybrid_retriever.py
"""GPU 加速混合检索器

在混合加速模式下，利用本地 GPU 加速检索环节：
- 向量检索加速（GPU 余弦相似度计算）
- BM25 检索（CPU 任务）
- RRF 融合排序
- 可选：本地 Reranker 精排（4GB 显存可运行 bge-reranker-base）

云端 Embedding API 和 Reranker API 不受影响。
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.core.hardware import hardware_manager
from app.core.logging_config import log_rag_query

logger = logging.getLogger(__name__)

__all__ = ["GPUAcceleratedRetriever", "RetrievalMetrics"]


class RetrievalMetrics:
    """检索性能指标收集器"""

    def __init__(self) -> None:
        self.bm25_time_ms: float = 0.0
        self.vector_time_ms: float = 0.0
        self.rrf_time_ms: float = 0.0
        self.reranker_time_ms: float = 0.0
        self.total_time_ms: float = 0.0
        self.device: str = "cpu"
        self.use_reranker: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "bm25_time_ms": round(self.bm25_time_ms, 1),
            "vector_time_ms": round(self.vector_time_ms, 1),
            "rrf_time_ms": round(self.rrf_time_ms, 1),
            "reranker_time_ms": round(self.reranker_time_ms, 1),
            "total_time_ms": round(self.total_time_ms, 1),
            "device": self.device,
            "use_reranker": self.use_reranker,
        }


class LocalReranker:
    """本地 Reranker（4GB 显存可运行）

    使用 BAAI/bge-reranker-base 模型进行本地精排，
    仅在 CUDA 模式下启用，约占用 280MB 显存。

    注意：这是可选功能，不影响云端 API Reranker。
    """

    def __init__(self, device: str = "cuda") -> None:
        import os

        if not os.getenv("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        self._device = device
        self._model = None
        self._tokenizer = None
        self._available = False

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
            self._model = AutoModelForSequenceClassification.from_pretrained(
                "BAAI/bge-reranker-base"
            )

            if device == "cuda":
                self._model = self._model.to("cuda")
                logger.info("本地 Reranker 加载至 CUDA GPU")
            else:
                logger.info("本地 Reranker 加载至 CPU")

            self._model.eval()
            self._available = True
        except Exception as exc:
            logger.warning("本地 Reranker 加载失败：%s", exc)
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def device(self) -> str:
        return self._device

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 3,
    ) -> tuple[list[Document], float]:
        """执行本地精排

        Args:
            query: 用户查询
            documents: 文档列表
            top_k: 返回数量

        Returns:
            (精排后的文档列表, 耗时毫秒)
        """
        if not self._available or not documents:
            return documents[:top_k], 0.0

        import torch

        start_time = time.monotonic()

        with torch.no_grad():
            inputs = self._tokenizer(
                [query] * len(documents),
                [doc.page_content for doc in documents],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )

            if self._device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            scores = self._model(**inputs).logits.squeeze(-1)
            if scores.dim() == 0:
                scores = scores.unsqueeze(0)

        duration_ms = (time.monotonic() - start_time) * 1000
        sorted_indices = scores.argsort(descending=True)[:top_k]

        reranked: list[Document] = []
        for idx in sorted_indices:
            i = idx.item()
            score = scores[i].item()  # bge-reranker-base 是 CrossEncoder，logits 已是相似度分数，无需 sigmoid
            doc = documents[i]
            doc_copy = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "rerank_score": score},
            )
            reranked.append(doc_copy)

        logger.info(
            "本地 Reranker 精排完成，耗时 %.1fms，设备：%s",
            duration_ms,
            self._device,
        )

        return reranked, duration_ms


class GPUAcceleratedRetriever:
    """GPU 加速的混合检索器

    在混合加速模式下，利用本地 GPU 加速检索环节：
    - 向量检索：GPU 余弦相似度计算（比 CPU 快 5-10 倍）
    - BM25 检索：CPU 任务
    - RRF 融合排序
    - 可选：本地 Reranker 精排

    云端 Embedding API 获取向量后，本地 GPU 加速相似度计算。
    云端 Reranker API 不受影响，本地 Reranker 为可选替代。

    使用方式：
        retriever = GPUAcceleratedRetriever(vectorstore, all_documents)
        results, metrics = retriever.retrieve("报销流程", top_k=3)
    """

    RRF_K: int = 60  # RRF 常数

    def __init__(
        self,
        vectorstore: Chroma,
        all_documents: list[str],
        use_local_reranker: bool = False,
    ) -> None:
        """初始化检索器

        Args:
            vectorstore: ChromaDB 向量库
            all_documents: 所有文档文本列表（用于 BM25）
            use_local_reranker: 是否使用本地 Reranker（默认 False，使用云端 API）
        """
        self._vectorstore = vectorstore
        self._all_documents = all_documents
        self._use_gpu = hardware_manager.is_hybrid_acceleration()
        self._device = hardware_manager.device if self._use_gpu else "cpu"

        # BM25 索引
        self._bm25: BM25Okapi | None = None
        self._build_bm25_index()

        # GPU 向量存储
        self._gpu_embeddings: Any = None
        self._gpu_documents: list[str] = []
        if self._use_gpu:
            self._build_gpu_vector_store()

        # 本地 Reranker（可选）
        self._local_reranker: LocalReranker | None = None
        if use_local_reranker and hardware_manager.supports_local_reranker():
            self._local_reranker = LocalReranker(device=self._device)

    def _build_bm25_index(self) -> None:
        """构建 BM25 索引"""
        import jieba

        tokenized_docs = [list(jieba.cut(doc)) for doc in self._all_documents]
        self._bm25 = BM25Okapi(tokenized_docs)
        logger.info("BM25 索引构建完成：%d 个文档", len(self._all_documents))

    def _build_gpu_vector_store(self) -> None:
        """构建 GPU 向量存储

        从 ChromaDB 获取所有 embeddings，转换为 GPU tensor。
        GPU 余弦相似度计算比 CPU 快 5-10 倍。
        """
        try:
            import torch

            # 从 ChromaDB 获取所有文档和 embeddings
            collection = self._vectorstore._collection
            data = collection.get(include=["embeddings", "documents"])

            if data["embeddings"] is not None and len(data["embeddings"]) > 0:
                self._gpu_embeddings = torch.tensor(
                    data["embeddings"], dtype=torch.float32
                ).to(self._device)
                self._gpu_documents = data["documents"] or []
                logger.info(
                    "GPU 向量存储构建完成：%d 个文档，设备=%s",
                    len(self._gpu_documents),
                    self._device,
                )
            else:
                logger.warning("向量库为空，GPU 向量存储未构建")
        except Exception as exc:
            logger.warning("GPU 向量存储构建失败：%s", exc)
            self._gpu_embeddings = None

    def retrieve(
        self,
        query: str,
        department: str = "通用",
        top_k: int = 3,
        use_reranker: bool = False,
    ) -> tuple[list[Document], RetrievalMetrics]:
        """混合检索：BM25 + 向量检索 + RRF 融合 + 可选 Reranker

        Args:
            query: 查询文本
            department: 部门过滤
            top_k: 返回文档数量
            use_reranker: 是否使用 Reranker 精排

        Returns:
            (检索结果列表, 性能指标)
        """
        metrics = RetrievalMetrics()
        metrics.device = self._device
        metrics.use_reranker = use_reranker

        total_start = time.monotonic()

        # Step 1: BM25 检索
        bm25_start = time.monotonic()
        bm25_results = self._bm25_search(query, top_k * 2)
        metrics.bm25_time_ms = (time.monotonic() - bm25_start) * 1000

        # Step 2: 向量检索（GPU 加速）
        vector_start = time.monotonic()
        vector_results = self._vector_search(query, department, top_k * 2)
        metrics.vector_time_ms = (time.monotonic() - vector_start) * 1000

        # Step 3: RRF 融合
        rrf_start = time.monotonic()
        fused_results = self._rrf_fusion(bm25_results, vector_results, top_k)
        metrics.rrf_time_ms = (time.monotonic() - rrf_start) * 1000

        # Step 4: 可选 Reranker 精排
        if use_reranker and fused_results:
            reranker_start = time.monotonic()
            fused_results = self._rerank(query, fused_results, top_k)
            metrics.reranker_time_ms = (time.monotonic() - reranker_start) * 1000

        metrics.total_time_ms = (time.monotonic() - total_start) * 1000

        # 收集 RRF 融合后的分数用于日志
        rrf_scores_list: list[float] = []
        for doc in fused_results:
            score = doc.metadata.get("rerank_score") or doc.metadata.get("rrf_score")
            if score is not None:
                rrf_scores_list.append(float(score))

        logger.info(
            "混合检索完成：BM25=%.1fms，向量=%.1fms，RRF=%.1fms，Reranker=%.1fms，总计=%.1fms，设备=%s",
            metrics.bm25_time_ms,
            metrics.vector_time_ms,
            metrics.rrf_time_ms,
            metrics.reranker_time_ms,
            metrics.total_time_ms,
            metrics.device,
        )

        log_rag_query(
            query=query,
            top_k=top_k,
            hit_count=len(fused_results),
            duration_ms=metrics.total_time_ms,
            scores=rrf_scores_list if rrf_scores_list else None,
        )

        return fused_results, metrics

    def _bm25_search(
        self, query: str, top_k: int
    ) -> list[tuple[str, float, int]]:
        """BM25 检索

        Returns:
            [(文档内容, BM25分数, 原始索引), ...]
        """
        if self._bm25 is None:
            logger.info("BM25 索引未构建，跳过 BM25 检索")
            return []

        import jieba

        tokenized_query = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [
            (self._all_documents[i], scores[i], i)
            for i in top_indices
            if scores[i] > 0
        ]

        # 记录 BM25 命中详情
        if results:
            top_scores = [round(r[1], 4) for r in results[:5]]
            logger.info(
                "BM25 检索命中 %d 个文档，top分数=%s，查询分词数=%d",
                len(results),
                top_scores,
                len(tokenized_query),
            )
        else:
            logger.info("BM25 检索无命中结果，查询分词数=%d", len(tokenized_query))

        return results

    def _vector_search(
        self, query: str, department: str, top_k: int
    ) -> list[tuple[Document, float, int]]:
        """向量检索（GPU 加速）

        如果 GPU 向量存储可用，使用 GPU 计算余弦相似度；
        否则回退到 ChromaDB 的 similarity_search。

        Returns:
            [(文档, 相似度分数, 排名), ...]
        """
        results: list[tuple[Document, float, int]] = []

        if self._use_gpu and self._gpu_embeddings is not None:
            # GPU 加速向量检索
            try:
                import torch

                # 通过 ChromaDB 的 embedding_function 获取查询向量
                query_embedding = self._vectorstore.embedding_function.embed_query(query)  # 使用公开属性，避免访问私有接口
                query_tensor = torch.tensor(
                    [query_embedding], dtype=torch.float32
                ).to(self._device)

                # GPU 余弦相似度计算
                with torch.no_grad():
                    similarities = torch.cosine_similarity(
                        query_tensor, self._gpu_embeddings, dim=1
                    )
                    top_indices = torch.topk(similarities, min(top_k, len(similarities)))

                # 获取对应文档
                for rank, (idx, score) in enumerate(
                    zip(top_indices.indices.cpu().numpy(), top_indices.values.cpu().numpy())
                ):
                    doc_idx = int(idx)
                    if doc_idx < len(self._gpu_documents):
                        doc = Document(
                            page_content=self._gpu_documents[doc_idx],
                            metadata={"source": "vector_search", "department": department},
                        )
                        results.append((doc, float(score), rank))

                logger.info("GPU 向量检索命中 %d 个文档", len(results))
                if results:
                    top_scores = [round(r[1], 4) for r in results[:5]]
                    logger.info("GPU 向量检索 top 分数=%s，设备=%s", top_scores, self._device)
                return results

            except Exception as exc:
                logger.warning("GPU 向量检索失败，回退到 ChromaDB：%s", exc)

        # 回退到 ChromaDB similarity_search
        try:
            search_kwargs: dict[str, Any] = {"k": top_k}
            if department != "通用":
                search_kwargs["filter"] = {"department": department}
            docs = self._vectorstore.similarity_search(query, **search_kwargs)
            results = [(doc, 0.0, rank) for rank, doc in enumerate(docs)]
            logger.info("ChromaDB 向量检索命中 %d 个文档", len(results))
            if results:
                logger.info("ChromaDB 向量检索完成，部门过滤=%s", department if department != "通用" else "无")
        except Exception as exc:
            logger.warning("ChromaDB 向量检索失败：%s", exc)

        return results

    def _rrf_fusion(
        self,
        bm25_results: list[tuple[str, float, int]],
        vector_results: list[tuple[Document, float, int]],
        top_k: int,
    ) -> list[Document]:
        """RRF 融合排序

        RRF 公式：score = Σ 1/(k + rank)
        """
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        # BM25 分数
        for rank, (content, _score, _idx) in enumerate(bm25_results):
            key = content[:200]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.RRF_K + rank + 1)
            if key not in doc_map:
                doc_map[key] = Document(
                    page_content=content,
                    metadata={"source": "BM25检索"},
                )

        # 向量检索分数
        for rank, (doc, _score, _idx) in enumerate(vector_results):
            key = doc.page_content[:200]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.RRF_K + rank + 1)
            if key not in doc_map:
                doc_map[key] = doc

        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)[
            :top_k
        ]
        return [doc_map[k] for k in sorted_keys]

    def _rerank(
        self, query: str, documents: list[Document], top_k: int
    ) -> list[Document]:
        """Reranker 精排

        优先使用本地 Reranker（如果已初始化），
        否则使用云端 API Reranker（通过依赖注入）。
        """
        if self._local_reranker and self._local_reranker.is_available:
            reranked, _duration_ms = self._local_reranker.rerank(
                query, documents, top_k
            )
            return reranked

        # 回退到云端 API Reranker
        try:
            from app.core.dependencies import get_reranker

            reranker = get_reranker()
            rerank_results = reranker.rerank(query=query, documents=documents, top_n=top_k)
            if rerank_results:
                return [r.document for r in rerank_results]
        except Exception as exc:
            logger.warning("云端 Reranker 失败：%s", exc)

        return documents[:top_k]