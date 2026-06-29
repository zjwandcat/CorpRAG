"""Reranker 精排封装模块

支持多种 Reranker 后端：
- NVIDIARerank (nvidia/llama-nemotron-rerank-1b-v2)
- 智谱AI Reranker（通过 OpenAI 兼容 API 调用）
- NoOp 降级模式（Reranker 不可用时返回原始排序）
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import requests
from langchain_core.documents import Document

from app.core.logging_config import log_function_call

__all__ = ["RerankResult", "Reranker"]

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Reranker 返回的单条结果"""

    index: int  # 原始文档列表中的索引
    relevance_score: float  # 精排相关度分数 (0-1)
    document: Document  # 原始 Document 对象


class Reranker:
    """Reranker 精排封装（多 Provider 支持）"""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        timeout: int,
        provider: str = "nim",
    ) -> None:
        self._model = model
        self._timeout = timeout
        self._provider = provider
        self._api_key = api_key
        self._base_url = base_url
        self._available = False
        self._client = None

        if provider == "zhipu" and api_key:
            # 智谱 Reranker：使用 HTTP API
            self._available = True
            logger.info("Reranker 初始化完成（智谱AI），base_url=%s", base_url)
        elif provider == "nim" and api_key:
            # NVIDIA Reranker：使用 NVIDIARerank SDK
            try:
                from langchain_nvidia_ai_endpoints import NVIDIARerank

                self._client = NVIDIARerank(
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                )
                self._available = True
                logger.info("Reranker 初始化完成，模型：%s", model)
            except Exception as exc:
                self._client = None
                logger.warning("Reranker 初始化失败，将使用降级模式：%s", exc)
        else:
            logger.warning("Reranker 配置不完整，使用降级模式")

    @classmethod
    def create_for_provider(
        cls,
        provider: str,
        api_key: str,
        base_url: str,
        model: str = "",
        timeout: int = 10,
    ) -> Reranker:
        """根据 Provider 创建对应的 Reranker 实例"""
        if provider == "zhipu":
            return cls(
                model=model or "rerank",
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                provider="zhipu",
            )
        return cls(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            provider=provider,
        )

    @classmethod
    def create_noop(cls) -> Reranker:
        """创建降级模式的 Reranker（不调用任何 API）"""
        return cls(model="", api_key="", base_url="", timeout=0, provider="noop")

    def is_available(self) -> bool:
        """检查 Reranker 是否可用"""
        return self._available

    def rerank(self, query: str, documents: list[Document], top_n: int = 3) -> list[RerankResult]:
        """对文档列表进行精排

        Args:
            query: 用户原始查询
            documents: hybrid_search 返回的 Document 列表
            top_n: 返回数量

        Returns:
            精排后的 RerankResult 列表，降级时返回原始文档列表（rerank_score=0.0）
        """
        rerank_start = time.monotonic()

        # 空结果处理：documents 为空时直接返回空列表
        if not documents:
            return []

        # 降级模式：Reranker 不可用时返回原始文档列表
        if not self.is_available():
            logger.warning("Reranker 不可用，使用原始排序（降级模式）")
            duration_ms = (time.monotonic() - rerank_start) * 1000
            log_function_call(
                func_name="rerank",
                duration_ms=duration_ms,
                result_summary=f"降级模式，文档数={len(documents)}，top_n={top_n}",
            )
            return [
                RerankResult(index=i, relevance_score=0.0, document=doc)
                for i, doc in enumerate(documents[:top_n])
            ]

        if self._provider == "zhipu":
            results = self._rerank_zhipu(query, documents, top_n)
        elif self._provider == "nim" and self._client is not None:
            results = self._rerank_nvidia(query, documents, top_n)
        else:
            results = self._rerank_fallback(documents, top_n)

        duration_ms = (time.monotonic() - rerank_start) * 1000
        rerank_scores = [round(r.relevance_score, 4) for r in results]
        logger.info(
            "Reranker 精排完成，provider=%s，返回 %d 个结果，耗时=%.1fms，分数=%s",
            self._provider,
            len(results),
            duration_ms,
            rerank_scores,
        )
        log_function_call(
            func_name="rerank",
            duration_ms=duration_ms,
            result_summary=f"provider={self._provider}，结果数={len(results)}，分数={rerank_scores}",
        )

        return results

    def _rerank_zhipu(
        self, query: str, documents: list[Document], top_n: int
    ) -> list[RerankResult]:
        """使用智谱AI Reranker API 进行精排"""
        try:
            url = f"{self._base_url.rstrip('/')}/rerank"
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            doc_texts = [doc.page_content for doc in documents]
            payload = {
                "model": self._model or "rerank",
                "query": query,
                "documents": doc_texts,
                "top_n": min(top_n, len(doc_texts)),
            }

            response = requests.post(url, json=payload, headers=headers, timeout=self._timeout)

            if response.status_code == 429:
                logger.warning("智谱 Reranker 限流（429），使用原始排序")
                return self._rerank_fallback(documents, top_n)

            if response.status_code != 200:
                logger.warning("智谱 Reranker 返回 %d，使用原始排序", response.status_code)
                return self._rerank_fallback(documents, top_n)

            data = response.json()
            results_data = data.get("results", [])

            if not results_data:
                logger.warning("智谱 Reranker 返回空结果，使用原始排序")
                return self._rerank_fallback(documents, top_n)

            results: list[RerankResult] = []
            for item in results_data:
                idx = item.get("index", 0)
                score = float(item.get("relevance_score", 0.0))
                doc = documents[idx] if idx < len(documents) else documents[0]
                # 将 rerank_score 写入 metadata
                doc_copy = Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "rerank_score": score},
                )
                results.append(RerankResult(index=idx, relevance_score=score, document=doc_copy))

            results.sort(key=lambda r: r.relevance_score, reverse=True)
            results = results[:top_n]

            # 记录每个文档的精排分数
            rerank_scores = [round(r.relevance_score, 4) for r in results]
            logger.info(
                "智谱 Reranker 精排完成，返回 %d 个结果，分数=%s",
                len(results),
                rerank_scores,
            )
            return results

        except requests.Timeout:
            logger.warning("智谱 Reranker 超时，使用原始排序")
            return self._rerank_fallback(documents, top_n)
        except Exception as exc:
            logger.warning("智谱 Reranker 调用失败，使用原始排序：%s", exc)
            return self._rerank_fallback(documents, top_n)

    def _rerank_nvidia(
        self, query: str, documents: list[Document], top_n: int
    ) -> list[RerankResult]:
        """使用 NVIDIARerank 进行精排（原有逻辑）"""
        try:
            compressed = self._client.compress_documents(
                documents=documents,
                query=query,
            )

            if not compressed:
                logger.warning("Reranker 返回空结果，使用原始排序")
                return self._rerank_fallback(documents, top_n)

            results: list[RerankResult] = []
            for doc in compressed:
                score = float(doc.metadata.get("relevance_score", 0.0))
                original_index = self._find_original_index(doc, documents)
                results.append(
                    RerankResult(index=original_index, relevance_score=score, document=doc)
                )

            results.sort(key=lambda r: r.relevance_score, reverse=True)
            results = results[:top_n]

            # 记录每个文档的精排分数
            rerank_scores = [round(r.relevance_score, 4) for r in results]
            logger.info(
                "Reranker 精排完成，返回 %d 个结果（top %d），分数=%s",
                len(results),
                top_n,
                rerank_scores,
            )
            return results

        except Exception as exc:
            logger.warning("Reranker 调用失败，使用原始排序（降级模式）：%s", exc)
            return self._rerank_fallback(documents, top_n)

    def _rerank_fallback(self, documents: list[Document], top_n: int) -> list[RerankResult]:
        """降级模式：返回原始排序"""
        return [
            RerankResult(index=i, relevance_score=0.0, document=doc)
            for i, doc in enumerate(documents[:top_n])
        ]

    @staticmethod
    def _find_original_index(doc: Document, original_docs: list[Document]) -> int:
        """在原始文档列表中查找匹配的索引

        Args:
            doc: 精排后的文档
            original_docs: 原始文档列表

        Returns:
            匹配的索引，未找到时返回 -1
        """
        for i, original in enumerate(original_docs):
            if original.page_content == doc.page_content:
                return i
        return -1
