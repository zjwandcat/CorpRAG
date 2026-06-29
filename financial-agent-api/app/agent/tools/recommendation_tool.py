"""推荐工具和意图预测工具

提供 v5.0 MLOps 升级的两个核心工具：
- recommend_similar_documents: 相似文档推荐工具
- predict_user_intent: 用户意图预测工具
"""

import pickle
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool, tool
from langchain_core.documents import Document

from app.core.config import settings
from app.core.enums import IntentLabel
from app.core.logging_config import get_logger, log_agent_step, log_function_call
from app.models.schemas import RecommendationItem, IntentPredictionResult
from app.observability.metrics import TOOL_CALL_COUNT, TOOL_CALL_LATENCY

logger = get_logger(__name__)

__all__ = [
    "make_recommend_similar_documents_tool",
    "make_predict_user_intent_tool",
]

# 性能约束常量
_RECOMMENDATION_TIMEOUT_MS = 3000  # 3 秒
_INTENT_PREDICTION_TIMEOUT_MS = 500  # 500ms


def make_recommend_similar_documents_tool(
    vectorstore: Chroma,
    embeddings: Embeddings,
) -> BaseTool:
    """创建相似文档推荐工具

    Args:
        vectorstore: ChromaDB 向量存储实例
        embeddings: Embeddings 实例（云端 API）

    Returns:
        LangChain Tool 实例
    """

    @tool
    def recommend_similar_documents(
        query: str,
        exclude_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[RecommendationItem]:
        """推荐与查询相似但未在当前 RAG 上下文中出现的文档。

        用于在用户查询后，推荐相关但尚未检索到的文档，扩展用户视野。

        Args:
            query: 用户查询文本
            exclude_ids: 需要排除的文档 ID 列表（当前 RAG 上下文中的文档）
            top_k: 返回的推荐文档数量，默认使用配置中的 RECOMMENDATION_TOP_K

        Returns:
            推荐文档列表，每个条目包含文档 ID、内容、来源和相似度得分
        """
        step_start = time.monotonic()
        effective_top_k = top_k if top_k is not None else settings.RECOMMENDATION_TOP_K
        exclude_ids = exclude_ids or []

        logger.info(
            "执行工具 recommend_similar_documents，query=%s, exclude_ids=%d, top_k=%d",
            query,
            len(exclude_ids),
            effective_top_k,
        )

        try:
            # Step 1: 提取 query Embedding（使用云端 API）
            query_embedding = embeddings.embed_query(query)

            # Step 2: ChromaDB 相似度检索
            # 为了过滤掉 exclude_ids，我们需要检索更多结果
            fetch_k = effective_top_k + len(exclude_ids) + 10  # 多检索一些以应对过滤
            results = vectorstore.similarity_search_by_vector_with_relevance_scores(
                embedding=query_embedding,
                k=fetch_k,
            )

            # Step 3: 排除当前 RAG 上下文
            filtered_results: list[tuple[Document, float]] = []
            for doc, score in results:
                doc_id = doc.metadata.get("id", "")
                if doc_id not in exclude_ids:
                    filtered_results.append((doc, score))
                    if len(filtered_results) >= effective_top_k:
                        break

            # Step 4: 构建推荐结果
            recommendations: list[RecommendationItem] = []
            for doc, score in filtered_results:
                item = RecommendationItem(
                    document_id=doc.metadata.get("id", ""),
                    content=doc.page_content,
                    source=doc.metadata.get("source", ""),
                    similarity_score=float(score),
                )
                recommendations.append(item)

            duration_ms = (time.monotonic() - step_start) * 1000

            # 性能约束检查
            if duration_ms > _RECOMMENDATION_TIMEOUT_MS:
                logger.warning(
                    "推荐工具执行超时：%.1fms > %dms",
                    duration_ms,
                    _RECOMMENDATION_TIMEOUT_MS,
                )

            logger.info(
                "推荐工具返回 %d 条结果，耗时=%.1fms",
                len(recommendations),
                duration_ms,
            )

            log_agent_step(
                step_name="recommend_similar_documents",
                tool_name="recommend_similar_documents",
                duration_ms=duration_ms,
                status="success",
            )
            log_function_call(
                func_name="recommend_similar_documents",
                kwargs={"query": query, "exclude_ids_count": len(exclude_ids)},
                duration_ms=duration_ms,
                result_summary=f"recommendation_count={len(recommendations)}",
            )

            TOOL_CALL_COUNT.labels(
                tool_name="recommend_similar_documents", status="success"
            ).inc()
            TOOL_CALL_LATENCY.labels(
                tool_name="recommend_similar_documents"
            ).observe(duration_ms / 1000.0)

            return recommendations

        except Exception as exc:
            # 异常降级：返回空结果
            duration_ms = (time.monotonic() - step_start) * 1000
            logger.warning(
                "推荐工具执行失败，降级返回空结果：%s，耗时=%.1fms",
                exc,
                duration_ms,
            )

            log_agent_step(
                step_name="recommend_similar_documents",
                tool_name="recommend_similar_documents",
                duration_ms=duration_ms,
                status="error",
            )
            log_function_call(
                func_name="recommend_similar_documents",
                kwargs={"query": query, "exclude_ids_count": len(exclude_ids)},
                duration_ms=duration_ms,
                result_summary=f"error: {type(exc).__name__}",
            )

            TOOL_CALL_COUNT.labels(
                tool_name="recommend_similar_documents", status="error"
            ).inc()
            TOOL_CALL_LATENCY.labels(
                tool_name="recommend_similar_documents"
            ).observe(duration_ms / 1000.0)

            return []

    return recommend_similar_documents


def make_predict_user_intent_tool(model_path: str | None = None) -> BaseTool:
    """创建用户意图预测工具

    Args:
        model_path: 预训练模型文件路径，默认使用配置中的 INTENT_MODEL_PATH

    Returns:
        LangChain Tool 实例
    """
    effective_model_path = model_path or settings.INTENT_MODEL_PATH
    model_file = Path(effective_model_path)

    # 尝试加载模型（延迟加载，工具调用时才加载）
    model_data: dict | None = None

    def _load_model() -> dict | None:
        """加载预训练模型（TF-IDF + 朴素贝叶斯）"""
        nonlocal model_data

        if model_data is not None:
            return model_data

        if not model_file.exists():
            logger.warning(
                "意图预测模型文件不存在：%s，工具将返回空结果",
                model_file,
            )
            return None

        try:
            with open(model_file, "rb") as f:
                model_data = pickle.load(f)

            logger.info(
                "意图预测模型加载成功：%s",
                model_file,
            )
            return model_data

        except Exception as exc:
            logger.warning(
                "意图预测模型加载失败：%s，工具将返回空结果",
                exc,
            )
            return None

    @tool
    def predict_user_intent(query: str) -> IntentPredictionResult | None:
        """预测用户查询的意图类型。

        使用预训练的 TF-IDF + 朴素贝叶斯模型进行意图分类，
        不调用 LLM API，性能约束为 500ms 内返回结果。

        Args:
            query: 用户查询文本

        Returns:
            意图预测结果，包含意图标签和置信度；
            如果模型不可用或预测失败，返回 None
        """
        step_start = time.monotonic()

        logger.info(
            "执行工具 predict_user_intent，query=%s",
            query,
        )

        # 加载模型
        loaded_model = _load_model()
        if loaded_model is None:
            # 模型不可用，优雅降级
            duration_ms = (time.monotonic() - step_start) * 1000
            logger.warning(
                "意图预测模型不可用，降级返回空结果，耗时=%.1fms",
                duration_ms,
            )

            log_agent_step(
                step_name="predict_user_intent",
                tool_name="predict_user_intent",
                duration_ms=duration_ms,
                status="error",
            )
            log_function_call(
                func_name="predict_user_intent",
                kwargs={"query": query},
                duration_ms=duration_ms,
                result_summary="model_unavailable",
            )

            TOOL_CALL_COUNT.labels(
                tool_name="predict_user_intent", status="error"
            ).inc()
            TOOL_CALL_LATENCY.labels(
                tool_name="predict_user_intent"
            ).observe(duration_ms / 1000.0)

            return None

        try:
            # 提取模型组件
            vectorizer = loaded_model.get("vectorizer")  # TF-IDF Vectorizer
            classifier = loaded_model.get("classifier")  # Naive Bayes Classifier
            label_encoder = loaded_model.get("label_encoder")  # Label Encoder

            if vectorizer is None or classifier is None:
                raise ValueError("模型文件缺少必要的组件（vectorizer 或 classifier）")

            # Step 1: TF-IDF 特征提取
            features = vectorizer.transform([query])

            # Step 2: 朴素贝叶斯分类
            predicted_label = classifier.predict(features)[0]

            # Step 3: 获取置信度（概率）
            probabilities = classifier.predict_proba(features)[0]
            confidence = float(max(probabilities))

            # Step 4: 解码标签
            if label_encoder is not None:
                intent_label = label_encoder.inverse_transform([predicted_label])[0]
            else:
                # 如果没有 label_encoder，直接使用预测的标签索引
                intent_labels = [
                    IntentLabel.REIMBURSEMENT,
                    IntentLabel.LEAVE_PROCESS,
                    IntentLabel.IT_SUPPORT,
                    IntentLabel.HR_MANAGEMENT,
                    IntentLabel.FINANCIAL_MANAGEMENT,
                    IntentLabel.OTHER,
                ]
                intent_label = intent_labels[predicted_label] if predicted_label < len(intent_labels) else IntentLabel.OTHER

            result = IntentPredictionResult(
                intent_label=str(intent_label),
                confidence=confidence,
            )

            duration_ms = (time.monotonic() - step_start) * 1000

            # 性能约束检查
            if duration_ms > _INTENT_PREDICTION_TIMEOUT_MS:
                logger.warning(
                    "意图预测工具执行超时：%.1fms > %dms",
                    duration_ms,
                    _INTENT_PREDICTION_TIMEOUT_MS,
                )

            logger.info(
                "意图预测完成：label=%s, confidence=%.2f, 耗时=%.1fms",
                result.intent_label,
                result.confidence,
                duration_ms,
            )

            log_agent_step(
                step_name="predict_user_intent",
                tool_name="predict_user_intent",
                duration_ms=duration_ms,
                status="success",
            )
            log_function_call(
                func_name="predict_user_intent",
                kwargs={"query": query},
                duration_ms=duration_ms,
                result_summary=f"label={result.intent_label}, confidence={result.confidence:.2f}",
            )

            TOOL_CALL_COUNT.labels(
                tool_name="predict_user_intent", status="success"
            ).inc()
            TOOL_CALL_LATENCY.labels(
                tool_name="predict_user_intent"
            ).observe(duration_ms / 1000.0)

            return result

        except Exception as exc:
            # 异常降级：返回空结果
            duration_ms = (time.monotonic() - step_start) * 1000
            logger.warning(
                "意图预测工具执行失败，降级返回空结果：%s，耗时=%.1fms",
                exc,
                duration_ms,
            )

            log_agent_step(
                step_name="predict_user_intent",
                tool_name="predict_user_intent",
                duration_ms=duration_ms,
                status="error",
            )
            log_function_call(
                func_name="predict_user_intent",
                kwargs={"query": query},
                duration_ms=duration_ms,
                result_summary=f"error: {type(exc).__name__}",
            )

            TOOL_CALL_COUNT.labels(
                tool_name="predict_user_intent", status="error"
            ).inc()
            TOOL_CALL_LATENCY.labels(
                tool_name="predict_user_intent"
            ).observe(duration_ms / 1000.0)

            return None

    return predict_user_intent