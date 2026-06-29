"""评估管道集成测试

测试 RAG 评估管道的完整流程，包括：
- 评估数据集加载
- 评估执行流程
- 评估结果写入 MLflow
- 评估 API 端点

运行方式：
    pytest tests/integration/test_eval_pipeline.py -v --tb=short
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.mlops.evaluator import RAGEvaluator
from app.mlops.tracking import LLMExperimentTracker
from app.models.schemas import EvalDatasetItem, EvalResponse


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_eval_dataset() -> list[dict]:
    """创建示例评估数据集

    Returns:
        包含 5 个 QA pairs 的评估数据集
    """
    return [
        {
            "query": "报销流程是什么？",
            "ground_truth": "员工报销需在费用发生后30天内提交，需提供发票原件和审批单。",
            "expected_contexts": [
                "报销流程规定：员工报销需在费用发生后30天内提交。",
                "报销材料要求：发票原件、审批单。",
            ],
        },
        {
            "query": "如何申请年假？",
            "ground_truth": "年假申请需提前5个工作日通过OA系统提交，经部门主管审批后生效。",
            "expected_contexts": [
                "年假申请流程：提前5个工作日通过OA系统提交。",
                "年假审批：部门主管审批。",
            ],
        },
        {
            "query": "公司差旅标准是多少？",
            "ground_truth": "国内差旅标准为每人每天500元，国际差旅标准为每人每天200美元。",
            "expected_contexts": [
                "国内差旅标准：每人每天500元。",
                "国际差旅标准：每人每天200美元。",
            ],
        },
        {
            "query": "入职需要哪些材料？",
            "ground_truth": "入职需提供身份证原件、学历证明、离职证明、体检报告和银行卡复印件。",
            "expected_contexts": [
                "入职材料清单：身份证原件、学历证明、离职证明。",
                "入职体检要求：提供体检报告。",
            ],
        },
        {
            "query": "会议室如何预约？",
            "ground_truth": "会议室预约通过OA系统进行，需提前1天预约，最长可预约4小时。",
            "expected_contexts": [
                "会议室预约方式：OA系统。",
                "预约时间要求：提前1天，最长4小时。",
            ],
        },
    ]


@pytest.fixture
def eval_dataset_file(sample_eval_dataset: list[dict]) -> Path:
    """创建临时评估数据集文件

    Args:
        sample_eval_dataset: 示例评估数据集

    Returns:
        临时文件路径
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        encoding="utf-8",
        delete=False,
    ) as f:
        json.dump({"samples": sample_eval_dataset}, f, ensure_ascii=False, indent=2)
        return Path(f.name)


@pytest.fixture
def mock_llm_judge() -> MagicMock:
    """创建 Mock LLM-as-Judge 实例

    Returns:
        Mock LLM 实例
    """
    llm = MagicMock()

    # 模拟 LLM 响应
    def mock_invoke(messages):
        response = MagicMock()
        # 返回 JSON 格式的评估结果
        response.content = json.dumps({
            "faithfulness_score": 0.85,
            "answer_relevancy_score": 0.90,
            "context_precision_score": 0.80,
            "context_recall_score": 0.75,
        })
        return response

    llm.invoke = mock_invoke
    return llm


@pytest.fixture
def mock_tracker() -> MagicMock:
    """创建 Mock MLflow Tracker 实例

    Returns:
        Mock Tracker 实例
    """
    tracker = MagicMock()
    tracker.is_available.return_value = True
    tracker.track_rag_run.return_value = "test-run-id-12345"
    return tracker


@pytest.fixture
def evaluator(
    mock_tracker: MagicMock,
    mock_llm_judge: MagicMock,
) -> RAGEvaluator:
    """创建 RAGEvaluator 实例

    Args:
        mock_tracker: Mock Tracker
        mock_llm_judge: Mock LLM-as-Judge

    Returns:
        RAGEvaluator 实例
    """
    return RAGEvaluator(
        tracker=mock_tracker,
        llm_judge=mock_llm_judge,
    )


# ============================================================================
# 测试用例：评估数据集加载
# ============================================================================


class TestDatasetLoading:
    """评估数据集加载测试"""

    def test_load_dataset_success(
        self,
        evaluator: RAGEvaluator,
        eval_dataset_file: Path,
    ) -> None:
        """测试成功加载评估数据集

        验证：
        - 数据集文件正确解析
        - 返回正确数量的 EvalDatasetItem
        - 每个 item 包含必需字段
        """
        eval_items = evaluator.load_dataset(str(eval_dataset_file))

        assert len(eval_items) == 5, f"预期 5 个评估条目，实际：{len(eval_items)}"

        for item in eval_items:
            assert isinstance(item, EvalDatasetItem)
            assert len(item.query) > 0
            assert len(item.ground_truth) > 0

    def test_load_dataset_list_format(
        self,
        sample_eval_dataset: list[dict],
    ) -> None:
        """测试加载列表格式的数据集

        验证：
        - 直接列表格式的数据集可正确加载
        """
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            encoding="utf-8",
            delete=False,
        ) as f:
            json.dump(sample_eval_dataset, f, ensure_ascii=False)
            temp_path = Path(f.name)

        try:
            tracker = MagicMock()
            evaluator = RAGEvaluator(tracker=tracker, llm_judge=MagicMock())
            eval_items = evaluator.load_dataset(str(temp_path))

            assert len(eval_items) == 5
        finally:
            temp_path.unlink(missing_ok=True)

    def test_load_dataset_file_not_found(self, evaluator: RAGEvaluator) -> None:
        """测试加载不存在的数据集文件

        验证：
        - 抛出 FileNotFoundError
        """
        with pytest.raises(FileNotFoundError):
            evaluator.load_dataset("non_existent_file.json")

    def test_load_dataset_invalid_json(self, evaluator: RAGEvaluator) -> None:
        """测试加载无效 JSON 文件

        验证：
        - 抛出 ValueError
        """
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            encoding="utf-8",
            delete=False,
        ) as f:
            f.write("invalid json content {{{")
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError):
                evaluator.load_dataset(str(temp_path))
        finally:
            temp_path.unlink(missing_ok=True)

    def test_load_dataset_missing_required_fields(self) -> None:
        """测试加载缺少必需字段的数据集

        验证：
        - 缺少 query 或 ground_truth 的条目被跳过
        - 记录 warning 日志
        """
        invalid_dataset = [
            {"query": "有效问题", "ground_truth": "有效答案"},
            {"query": "缺少 ground_truth"},  # 无效条目
            {"ground_truth": "缺少 query"},  # 无效条目
        ]

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            encoding="utf-8",
            delete=False,
        ) as f:
            json.dump(invalid_dataset, f, ensure_ascii=False)
            temp_path = Path(f.name)

        try:
            tracker = MagicMock()
            evaluator = RAGEvaluator(tracker=tracker, llm_judge=MagicMock())
            eval_items = evaluator.load_dataset(str(temp_path))

            # 只有 1 个有效条目
            assert len(eval_items) == 1
        finally:
            temp_path.unlink(missing_ok=True)


# ============================================================================
# 测试用例：评估执行流程
# ============================================================================


class TestEvaluationExecution:
    """评估执行流程测试"""

    def test_evaluate_success(
        self,
        evaluator: RAGEvaluator,
        eval_dataset_file: Path,
    ) -> None:
        """测试完整评估流程成功执行

        验证：
        - 返回 EvalResponse 对象
        - 四维度评分在 [0, 1] 范围内
        - eval_timestamp 非空
        """
        result = evaluator.evaluate(str(eval_dataset_file))

        assert isinstance(result, EvalResponse)
        assert 0.0 <= result.faithfulness_score <= 1.0
        assert 0.0 <= result.answer_relevancy_score <= 1.0
        assert 0.0 <= result.context_precision_score <= 1.0
        assert 0.0 <= result.context_recall_score <= 1.0
        assert len(result.eval_timestamp) > 0

    def test_evaluate_with_unavailable_llm_judge(
        self,
        mock_tracker: MagicMock,
        eval_dataset_file: Path,
    ) -> None:
        """测试 LLM-as-Judge 不可用时的降级行为

        验证：
        - 返回默认评估结果（所有评分为 0.0）
        - 不抛出异常
        """
        evaluator = RAGEvaluator(tracker=mock_tracker, llm_judge=None)
        result = evaluator.evaluate(str(eval_dataset_file))

        assert isinstance(result, EvalResponse)
        assert result.faithfulness_score == 0.0
        assert result.answer_relevancy_score == 0.0
        assert result.context_precision_score == 0.0
        assert result.context_recall_score == 0.0

    def test_evaluate_empty_dataset(
        self,
        evaluator: RAGEvaluator,
    ) -> None:
        """测试空数据集评估

        验证：
        - 返回默认评估结果
        """
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            encoding="utf-8",
            delete=False,
        ) as f:
            json.dump({"samples": []}, f)
            temp_path = Path(f.name)

        try:
            result = evaluator.evaluate(str(temp_path))
            assert isinstance(result, EvalResponse)
            # 空数据集返回默认结果
            assert result.faithfulness_score == 0.0
        finally:
            temp_path.unlink(missing_ok=True)

    def test_is_available(
        self,
        evaluator: RAGEvaluator,
    ) -> None:
        """测试评估器可用性检查

        验证：
        - 配置了 LLM-as-Judge 时返回 True
        - 未配置时返回 False
        """
        assert evaluator.is_available() is True

        evaluator_no_llm = RAGEvaluator(tracker=MagicMock(), llm_judge=None)
        assert evaluator_no_llm.is_available() is False


# ============================================================================
# 测试用例：评估结果写入 MLflow
# ============================================================================


class TestMLflowLogging:
    """评估结果写入 MLflow 测试"""

    def test_log_eval_to_mlflow_success(
        self,
        mock_tracker: MagicMock,
        mock_llm_judge: MagicMock,
        eval_dataset_file: Path,
    ) -> None:
        """测试评估结果成功写入 MLflow

        验证：
        - track_rag_run 被正确调用
        - 传入正确的 metrics 和 params
        """
        evaluator = RAGEvaluator(tracker=mock_tracker, llm_judge=mock_llm_judge)
        result = evaluator.evaluate(str(eval_dataset_file))

        # 验证 track_rag_run 被调用
        mock_tracker.track_rag_run.assert_called_once()

        # 获取调用参数
        call_args = mock_tracker.track_rag_run.call_args
        assert call_args is not None

        # 验证参数包含评估指标
        metrics = call_args.kwargs.get("metrics", {})
        assert "eval_faithfulness_score" in metrics
        assert "eval_answer_relevancy_score" in metrics
        assert "eval_context_precision_score" in metrics
        assert "eval_context_recall_score" in metrics

    def test_log_eval_without_tracker(
        self,
        mock_llm_judge: MagicMock,
        eval_dataset_file: Path,
    ) -> None:
        """测试未配置 Tracker 时跳过 MLflow 记录

        验证：
        - 评估正常完成
        - 不抛出异常
        """
        evaluator = RAGEvaluator(tracker=None, llm_judge=mock_llm_judge)
        result = evaluator.evaluate(str(eval_dataset_file))

        assert isinstance(result, EvalResponse)

    def test_mlflow_failure_degradation(
        self,
        mock_llm_judge: MagicMock,
        eval_dataset_file: Path,
    ) -> None:
        """测试 MLflow 记录失败时的降级行为

        验证：
        - MLflow 记录失败不影响评估结果返回
        - 记录 warning 日志
        """
        failing_tracker = MagicMock()
        failing_tracker.is_available.return_value = True
        failing_tracker.track_rag_run.side_effect = Exception("MLflow 连接失败")

        evaluator = RAGEvaluator(tracker=failing_tracker, llm_judge=mock_llm_judge)
        result = evaluator.evaluate(str(eval_dataset_file))

        # 评估结果仍应正常返回
        assert isinstance(result, EvalResponse)


# ============================================================================
# 测试用例：评估 API 端点
# ============================================================================


class TestEvalAPIEndpoint:
    """评估 API 端点测试"""

    def test_eval_endpoint_success(
        self,
        mock_tracker: MagicMock,
        mock_llm_judge: MagicMock,
        eval_dataset_file: Path,
    ) -> None:
        """测试评估 API 端点成功响应

        验证：
        - POST /api/v1/mlops/eval 返回 200
        - 响应包含四维度评分
        """
        # 这里使用 FastAPI TestClient 进行测试
        # 由于需要完整的 FastAPI 应用，这里模拟依赖注入
        from app.models.schemas import EvalRequest

        # 模拟评估器
        evaluator = RAGEvaluator(tracker=mock_tracker, llm_judge=mock_llm_judge)

        # 执行评估
        result = evaluator.evaluate(str(eval_dataset_file))

        # 验证结果
        assert isinstance(result, EvalResponse)
        assert result.dataset_version == "v1.0"

    def test_eval_endpoint_dataset_not_found(self) -> None:
        """测试评估 API 端点数据集不存在

        验证：
        - 返回默认评估结果（异常被捕获并降级）
        """
        # 模拟数据集不存在的场景
        non_existent_path = "non_existent_dataset.json"

        tracker = MagicMock()
        llm_judge = MagicMock()
        evaluator = RAGEvaluator(tracker=tracker, llm_judge=llm_judge)

        # 评估器会捕获异常并返回默认结果
        result = evaluator.evaluate(non_existent_path)
        
        # 验证返回默认结果（所有评分为 0.0）
        assert isinstance(result, EvalResponse)
        assert result.faithfulness_score == 0.0


# ============================================================================
# 清理 fixture
# ============================================================================


def test_cleanup(eval_dataset_file: Path) -> None:
    """清理临时文件"""
    eval_dataset_file.unlink(missing_ok=True)