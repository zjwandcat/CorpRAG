"""推荐工具和意图预测工具单元测试

验证 v5.0 MLOps 升级的推荐工具和意图预测工具功能。
"""

import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.agent.tools.recommendation_tool import (
    make_predict_user_intent_tool,
    make_recommend_similar_documents_tool,
)
from app.models.schemas import RecommendationItem, IntentPredictionResult


class TestRecommendationTool:
    """推荐工具测试"""

    def test_make_tool_returns_tool(self):
        """测试工厂函数返回 LangChain Tool"""
        mock_vectorstore = MagicMock()
        mock_embeddings = MagicMock()

        tool = make_recommend_similar_documents_tool(
            vectorstore=mock_vectorstore,
            embeddings=mock_embeddings,
        )

        assert tool.name == "recommend_similar_documents"
        assert hasattr(tool, "invoke")

    def test_recommend_returns_empty_on_error(self):
        """测试异常降级返回空结果"""
        mock_vectorstore = MagicMock()
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.side_effect = Exception("Embedding failed")

        tool = make_recommend_similar_documents_tool(
            vectorstore=mock_vectorstore,
            embeddings=mock_embeddings,
        )

        result = tool.invoke({"query": "测试查询"})

        assert result == []

    def test_recommend_excludes_ids(self):
        """测试排除当前 RAG 上下文"""
        mock_vectorstore = MagicMock()
        mock_embeddings = MagicMock()

        # 模拟 embedding
        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]

        # 模拟检索结果
        docs_with_scores = [
            (
                Document(
                    page_content="文档1",
                    metadata={"id": "doc1", "source": "source1.pdf"},
                ),
                0.9,
            ),
            (
                Document(
                    page_content="文档2",
                    metadata={"id": "doc2", "source": "source2.pdf"},
                ),
                0.8,
            ),
            (
                Document(
                    page_content="文档3",
                    metadata={"id": "doc3", "source": "source3.pdf"},
                ),
                0.7,
            ),
        ]
        mock_vectorstore.similarity_search_by_vector_with_relevance_scores.return_value = (
            docs_with_scores
        )

        tool = make_recommend_similar_documents_tool(
            vectorstore=mock_vectorstore,
            embeddings=mock_embeddings,
        )

        # 排除 doc1 和 doc2
        result = tool.invoke({"query": "测试查询", "exclude_ids": ["doc1", "doc2"]})

        assert len(result) == 1
        assert result[0].document_id == "doc3"

    def test_recommend_respects_top_k(self):
        """测试 top_k 参数"""
        mock_vectorstore = MagicMock()
        mock_embeddings = MagicMock()

        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]

        docs_with_scores = [
            (
                Document(
                    page_content=f"文档{i}",
                    metadata={"id": f"doc{i}", "source": f"source{i}.pdf"},
                ),
                0.9 - i * 0.1,
            )
            for i in range(5)
        ]
        mock_vectorstore.similarity_search_by_vector_with_relevance_scores.return_value = (
            docs_with_scores
        )

        tool = make_recommend_similar_documents_tool(
            vectorstore=mock_vectorstore,
            embeddings=mock_embeddings,
        )

        result = tool.invoke({"query": "测试查询", "top_k": 2})

        assert len(result) == 2


class TestIntentPredictionTool:
    """意图预测工具测试"""

    def test_make_tool_returns_tool(self):
        """测试工厂函数返回 LangChain Tool"""
        tool = make_predict_user_intent_tool()

        assert tool.name == "predict_user_intent"
        assert hasattr(tool, "invoke")

    def test_returns_none_when_model_not_found(self):
        """测试模型文件不存在时返回 None"""
        tool = make_predict_user_intent_tool(model_path="/nonexistent/model.pkl")

        result = tool.invoke({"query": "我想报销差旅费"})

        assert result is None

    def test_returns_none_on_model_error(self):
        """测试模型加载失败时返回 None"""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            # 写入无效数据
            f.write(b"invalid pickle data")
            temp_path = f.name

        try:
            tool = make_predict_user_intent_tool(model_path=temp_path)

            result = tool.invoke({"query": "我想请假"})

            assert result is None
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_predicts_intent_with_valid_model(self):
        """测试使用有效模型进行意图预测"""
        # 创建真实的 sklearn 模型
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.preprocessing import LabelEncoder

            # 训练数据
            queries = [
                "我想报销差旅费",
                "如何申请报销",
                "请假流程是什么",
                "我想请病假",
                "电脑无法启动",
                "IT支持",
                "入职流程",
                "人事管理",
                "财务报表",
                "预算申请",
            ]
            labels = [
                "报销咨询",
                "报销咨询",
                "请假流程",
                "请假流程",
                "IT支持",
                "IT支持",
                "人事管理",
                "人事管理",
                "财务管理",
                "财务管理",
            ]

            # 训练模型
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(queries)

            # 先编码标签为数值
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(labels)

            classifier = MultinomialNB()
            classifier.fit(X, y)

            model_data = {
                "vectorizer": vectorizer,
                "classifier": classifier,
                "label_encoder": label_encoder,
            }

            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                pickle.dump(model_data, f)
                temp_path = f.name

            try:
                tool = make_predict_user_intent_tool(model_path=temp_path)

                result = tool.invoke({"query": "我想报销差旅费"})

                assert result is not None
                # LangChain tool 返回 Pydantic 模型或字典
                if isinstance(result, IntentPredictionResult):
                    assert result.intent_label is not None
                    assert 0.0 <= result.confidence <= 1.0
                elif isinstance(result, dict):
                    assert "intent_label" in result
                    assert "confidence" in result
                    assert 0.0 <= result["confidence"] <= 1.0
                else:
                    pytest.fail(f"Unexpected result type: {type(result)}")
            finally:
                Path(temp_path).unlink(missing_ok=True)

        except ImportError:
            # 如果 sklearn 未安装，跳过此测试
            pytest.skip("sklearn not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])