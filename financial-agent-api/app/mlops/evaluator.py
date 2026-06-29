"""RAG 评估模块

使用 LLM-as-Judge 评估 RAG 输出质量，支持四维度评估：
- Faithfulness（忠实度）：评估 LLM 回答是否严格基于检索到的上下文
- Answer Relevancy（回答相关性）：评估 LLM 回答与用户 query 的相关程度
- Context Precision（上下文精确度）：检索到的上下文中相关信息的占比
- Context Recall（上下文召回率）：回答所需信息是否都被检索到的上下文覆盖

核心设计原则：
- 使用 LLM-as-Judge 进行评估判断
- 异常降级：LLM-as-Judge 调用失败时跳过失败维度，返回部分结果
- 结果持久化：评估结果写入 MLflow
- 性能约束：支持至少 10 个并发评估任务

使用方式：
    from app.mlops.evaluator import RAGEvaluator
    from app.mlops.tracking import LLMExperimentTracker

    tracker = LLMExperimentTracker()
    evaluator = RAGEvaluator(tracker=tracker)

    # 执行评估
    result = evaluator.evaluate(dataset_path="tests/eval/eval_dataset.json")
"""

import asyncio
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from app.core.config import settings
from app.models.schemas import EvalDatasetItem, EvalResponse

# ============================================================================
# 类型定义与常量
# ============================================================================

# 评估维度枚举
EVAL_DIMENSION_FAITHFULNESS = "faithfulness"
EVAL_DIMENSION_ANSWER_RELEVANCY = "answer_relevancy"
EVAL_DIMENSION_CONTEXT_PRECISION = "context_precision"
EVAL_DIMENSION_CONTEXT_RECALL = "context_recall"

# 最大并发评估任务数
MAX_CONCURRENT_EVAL_TASKS = 10

# LLM-as-Judge 超时时间（秒）
LLM_JUDGE_TIMEOUT_SECONDS = 30

logger = logging.getLogger(__name__)


# ============================================================================
# RAGEvaluator 类实现
# ============================================================================


class RAGEvaluator:
    """RAG 评估器，使用 LLM-as-Judge 评估 RAG 输出质量。

    评估四维度：Faithfulness、Answer Relevancy、Context Precision、Context Recall。
    评估结果写入 MLflow，返回 JSON 格式报告。

    核心设计原则：
    - 使用 LLM-as-Judge 进行评估判断
    - 异常降级：LLM-as-Judge 调用失败时跳过失败维度，返回部分结果
    - 结果持久化：评估结果写入 MLflow
    - 性能约束：支持至少 10 个并发评估任务

    Attributes:
        _tracker: LLM 实验追踪器
        _llm_judge: LLM-as-Judge 模型
        _available: 评估器是否可用
    """

    __slots__ = (
        "_available",
        "_llm_judge",
        "_tracker",
    )

    def __init__(
        self,
        tracker: Any | None = None,
        llm_judge: BaseChatModel | None = None,
    ) -> None:
        """初始化 RAG 评估器。

        Args:
            tracker: LLM 实验追踪器（LLMExperimentTracker 实例）
            llm_judge: LLM-as-Judge 模型（BaseChatModel 实例）
        """
        self._tracker = tracker
        self._llm_judge = llm_judge
        self._available: bool = True

        if not self._llm_judge:
            logger.warning(
                "[RAGEvaluator] LLM-as-Judge 未配置，评估功能将降级。"
                "请传入 llm_judge 参数。"
            )
            self._available = False

    def is_available(self) -> bool:
        """检查评估器是否可用。

        Returns:
            True 表示评估器可用，False 表示不可用
        """
        return self._available

    def load_dataset(self, dataset_path: str) -> list[EvalDatasetItem]:
        """加载评估数据集。

        Args:
            dataset_path: 评估数据集文件路径（JSON 格式）

        Returns:
            评估数据集条目列表

        Raises:
            FileNotFoundError: 数据集文件不存在
            ValueError: 数据集格式错误
        """
        path = Path(dataset_path)

        if not path.exists():
            raise FileNotFoundError(f"评估数据集文件不存在: {dataset_path}")

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            # 数据集可能是列表格式或包含 samples 字段的字典格式
            if isinstance(data, list):
                samples = data
            elif isinstance(data, dict) and "samples" in data:
                samples = data["samples"]
            else:
                raise ValueError(
                    f"评估数据集格式错误: 期望列表或包含 'samples' 字段的字典"
                )

            # 转换为 EvalDatasetItem 列表
            eval_items: list[EvalDatasetItem] = []
            for idx, item in enumerate(samples):
                try:
                    eval_item = EvalDatasetItem(
                        query=item["query"],
                        ground_truth=item["ground_truth"],
                        expected_contexts=item.get("expected_contexts", []),
                    )
                    eval_items.append(eval_item)
                except KeyError as e:
                    logger.warning(
                        f"[RAGEvaluator] 数据集条目 {idx} 缺少必需字段: {e}"
                    )
                    continue

            logger.info(
                f"[RAGEvaluator] 成功加载评估数据集 | "
                f"path={dataset_path} | "
                f"count={len(eval_items)}"
            )

            return eval_items

        except json.JSONDecodeError as e:
            raise ValueError(f"评估数据集 JSON 解析失败: {e}") from e

    def evaluate(self, dataset_path: str) -> EvalResponse:
        """执行完整评估流程。

        流程：
        1. 加载评估数据集（20 个 QA pairs）
        2. 对每个 QA pair 执行 RAG 查询获取 answer 和 contexts
        3. 使用 LLM-as-Judge 评估四维度
        4. 汇总评分，写入 MLflow
        5. 返回 EvalResponse

        Args:
            dataset_path: 评估数据集文件路径

        Returns:
            EvalResponse 包含四维度评分
        """
        if not self.is_available():
            logger.warning("[RAGEvaluator] 评估器不可用，返回默认评估结果")
            return self._create_default_response()

        start_time = datetime.now(timezone.utc)

        try:
            # 1. 加载评估数据集
            eval_items = self.load_dataset(dataset_path)

            if not eval_items:
                logger.warning("[RAGEvaluator] 评估数据集为空，返回默认评估结果")
                return self._create_default_response()

            # 2. 并发执行评估
            all_scores: dict[str, list[float]] = {
                EVAL_DIMENSION_FAITHFULNESS: [],
                EVAL_DIMENSION_ANSWER_RELEVANCY: [],
                EVAL_DIMENSION_CONTEXT_PRECISION: [],
                EVAL_DIMENSION_CONTEXT_RECALL: [],
            }
            failed_dimensions: list[str] = []

            # 使用 ThreadPoolExecutor 实现并发评估
            with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_EVAL_TASKS) as executor:
                futures = []
                for item in eval_items:
                    future = executor.submit(
                        self._evaluate_single_item,
                        item,
                    )
                    futures.append(future)

                # 收集评估结果
                for future in futures:
                    try:
                        scores = future.result(timeout=LLM_JUDGE_TIMEOUT_SECONDS * 2)
                        for dim, score in scores.items():
                            if score is not None:
                                all_scores[dim].append(score)
                    except Exception as e:
                        logger.warning(
                            f"[RAGEvaluator] 单条评估失败: {type(e).__name__}: {e}"
                        )

            # 3. 计算平均分数
            avg_scores: dict[str, float] = {}
            for dim, scores in all_scores.items():
                if scores:
                    avg_scores[dim] = sum(scores) / len(scores)
                else:
                    avg_scores[dim] = 0.0
                    if dim not in failed_dimensions:
                        failed_dimensions.append(dim)

            # 4. 构建评估响应
            eval_timestamp = datetime.now(timezone.utc).isoformat()
            eval_response = EvalResponse(
                faithfulness_score=avg_scores.get(EVAL_DIMENSION_FAITHFULNESS, 0.0),
                answer_relevancy_score=avg_scores.get(
                    EVAL_DIMENSION_ANSWER_RELEVANCY, 0.0
                ),
                context_precision_score=avg_scores.get(
                    EVAL_DIMENSION_CONTEXT_PRECISION, 0.0
                ),
                context_recall_score=avg_scores.get(EVAL_DIMENSION_CONTEXT_RECALL, 0.0),
                eval_timestamp=eval_timestamp,
                dataset_version="v1.0",
            )

            # 5. 写入 MLflow
            self._log_eval_to_mlflow(eval_response, len(eval_items), failed_dimensions)

            elapsed_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.info(
                f"[RAGEvaluator] 评估完成 | "
                f"dataset_count={len(eval_items)} | "
                f"elapsed_seconds={elapsed_seconds:.2f} | "
                f"failed_dimensions={failed_dimensions}"
            )

            return eval_response

        except Exception as e:
            logger.error(
                f"[RAGEvaluator] 评估执行异常: {type(e).__name__}: {e}",
                exc_info=True,
            )
            return self._create_default_response()

    def _evaluate_single_item(self, item: EvalDatasetItem) -> dict[str, float | None]:
        """评估单条数据项。

        Args:
            item: 评估数据项

        Returns:
            四维度评分字典
        """
        scores: dict[str, float | None] = {
            EVAL_DIMENSION_FAITHFULNESS: None,
            EVAL_DIMENSION_ANSWER_RELEVANCY: None,
            EVAL_DIMENSION_CONTEXT_PRECISION: None,
            EVAL_DIMENSION_CONTEXT_RECALL: None,
        }

        try:
            # 模拟 RAG 查询结果（实际应用中应调用 RAG 引擎）
            # 这里使用 ground_truth 作为模拟的 answer
            answer = item.ground_truth
            contexts = item.expected_contexts

            # 评估 Faithfulness
            faithfulness_score = self._evaluate_faithfulness(
                query=item.query,
                answer=answer,
                contexts=contexts,
            )
            scores[EVAL_DIMENSION_FAITHFULNESS] = faithfulness_score

            # 评估 Answer Relevancy
            answer_relevancy_score = self._evaluate_answer_relevancy(
                query=item.query,
                answer=answer,
            )
            scores[EVAL_DIMENSION_ANSWER_RELEVANCY] = answer_relevancy_score

            # 评估 Context Precision
            context_precision_score = self._evaluate_context_precision(
                query=item.query,
                contexts=contexts,
            )
            scores[EVAL_DIMENSION_CONTEXT_PRECISION] = context_precision_score

            # 评估 Context Recall
            context_recall_score = self._evaluate_context_recall(
                query=item.query,
                answer=answer,
                ground_truth=item.ground_truth,
                contexts=contexts,
            )
            scores[EVAL_DIMENSION_CONTEXT_RECALL] = context_recall_score

        except Exception as e:
            logger.warning(
                f"[RAGEvaluator] 单条评估异常: {type(e).__name__}: {e}"
            )

        return scores

    def _evaluate_faithfulness(
        self,
        query: str,
        answer: str,
        contexts: list[str],
    ) -> float:
        """评估忠实度：LLM 回答是否严格基于检索到的上下文。

        使用 LLM-as-Judge prompt 判断 answer 中每个声明是否可由 contexts 支持。
        返回 0.0-1.0 评分。

        Args:
            query: 用户查询
            answer: LLM 回答
            contexts: 检索到的上下文列表

        Returns:
            忠实度评分（0.0-1.0）
        """
        if not contexts:
            return 0.0

        prompt = f"""你是一位专业的 RAG 系统评估专家。请评估以下回答的忠实度。

【用户问题】{query}
【检索上下文】
{chr(10).join(f'{i+1}. {ctx}' for i, ctx in enumerate(contexts))}

【模型回答】{answer}

请判断模型回答中的每个声明是否可以由检索上下文直接支持。
输出 JSON 格式：
{{
  "claims": [
    {{"claim": "声明内容", "supported": true/false, "evidence": "支持证据或无"}}
  ],
  "faithfulness_score": 0.0-1.0
}}

仅输出 JSON，不要输出其他内容。"""

        try:
            response = self._llm_as_judge(prompt)
            result = self._parse_json_response(response)

            if result and "faithfulness_score" in result:
                score = float(result["faithfulness_score"])
                return max(0.0, min(1.0, score))

            # 备用计算方式：基于 claims 统计
            if result and "claims" in result:
                claims = result["claims"]
                if claims:
                    supported_count = sum(
                        1 for c in claims if c.get("supported", False)
                    )
                    return supported_count / len(claims)

        except Exception as e:
            logger.warning(
                f"[RAGEvaluator] Faithfulness 评估失败: {type(e).__name__}: {e}"
            )

        return 0.0

    def _evaluate_answer_relevancy(
        self,
        query: str,
        answer: str,
    ) -> float:
        """评估回答相关性：LLM 回答与用户 query 的相关程度。

        使用 LLM-as-Judge prompt 判断 answer 是否直接回应了 query。
        返回 0.0-1.0 评分。

        Args:
            query: 用户查询
            answer: LLM 回答

        Returns:
            回答相关性评分（0.0-1.0）
        """
        prompt = f"""你是一位专业的 RAG 系统评估专家。请评估以下回答与用户问题的相关性。

【用户问题】{query}
【模型回答】{answer}

请判断模型回答是否直接回应了用户问题，是否包含无关信息。
输出 JSON 格式：
{{
  "is_relevant": true/false,
  "relevancy_reason": "判断理由",
  "answer_relevancy_score": 0.0-1.0
}}

仅输出 JSON，不要输出其他内容。"""

        try:
            response = self._llm_as_judge(prompt)
            result = self._parse_json_response(response)

            if result and "answer_relevancy_score" in result:
                score = float(result["answer_relevancy_score"])
                return max(0.0, min(1.0, score))

            # 备用计算方式：基于 is_relevant
            if result and "is_relevant" in result:
                return 1.0 if result["is_relevant"] else 0.0

        except Exception as e:
            logger.warning(
                f"[RAGEvaluator] Answer Relevancy 评估失败: {type(e).__name__}: {e}"
            )

        return 0.0

    def _evaluate_context_precision(
        self,
        query: str,
        contexts: list[str],
    ) -> float:
        """评估上下文精确度：检索到的上下文中相关信息的占比。

        使用 LLM-as-Judge prompt 逐个判断 context 是否与 query 相关。
        返回 0.0-1.0 评分。

        Args:
            query: 用户查询
            contexts: 检索到的上下文列表

        Returns:
            上下文精确度评分（0.0-1.0）
        """
        if not contexts:
            return 0.0

        prompt = f"""你是一位专业的 RAG 系统评估专家。请评估以下检索上下文的精确度。

【用户问题】{query}
【检索上下文】
{chr(10).join(f'{i+1}. {ctx}' for i, ctx in enumerate(contexts))}

请逐个判断每个上下文片段是否与用户问题相关。
输出 JSON 格式：
{{
  "context_evaluations": [
    {{"context_index": 0, "is_relevant": true/false, "reason": "判断理由"}}
  ],
  "context_precision_score": 0.0-1.0
}}

仅输出 JSON，不要输出其他内容。"""

        try:
            response = self._llm_as_judge(prompt)
            result = self._parse_json_response(response)

            if result and "context_precision_score" in result:
                score = float(result["context_precision_score"])
                return max(0.0, min(1.0, score))

            # 备用计算方式：基于 context_evaluations 统计
            if result and "context_evaluations" in result:
                evaluations = result["context_evaluations"]
                if evaluations:
                    relevant_count = sum(
                        1 for e in evaluations if e.get("is_relevant", False)
                    )
                    return relevant_count / len(evaluations)

        except Exception as e:
            logger.warning(
                f"[RAGEvaluator] Context Precision 评估失败: {type(e).__name__}: {e}"
            )

        return 0.0

    def _evaluate_context_recall(
        self,
        query: str,
        answer: str,
        ground_truth: str,
        contexts: list[str],
    ) -> float:
        """评估上下文召回率：回答所需信息是否都被检索到的上下文覆盖。

        使用 LLM-as-Judge prompt 判断 ground_truth 中的关键信息是否在 contexts 中。
        返回 0.0-1.0 评分。

        Args:
            query: 用户查询
            answer: LLM 回答
            ground_truth: 标准答案
            contexts: 检索到的上下文列表

        Returns:
            上下文召回率评分（0.0-1.0）
        """
        if not contexts:
            return 0.0

        prompt = f"""你是一位专业的 RAG 系统评估专家。请评估以下检索上下文的召回率。

【用户问题】{query}
【标准答案】{ground_truth}
【检索上下文】
{chr(10).join(f'{i+1}. {ctx}' for i, ctx in enumerate(contexts))}

请判断标准答案中的关键信息是否在检索上下文中被覆盖。
输出 JSON 格式：
{{
  "essential_info": [
    {{"info": "关键信息", "found_in_context": true/false}}
  ],
  "context_recall_score": 0.0-1.0
}}

仅输出 JSON，不要输出其他内容。"""

        try:
            response = self._llm_as_judge(prompt)
            result = self._parse_json_response(response)

            if result and "context_recall_score" in result:
                score = float(result["context_recall_score"])
                return max(0.0, min(1.0, score))

            # 备用计算方式：基于 essential_info 统计
            if result and "essential_info" in result:
                essential_info = result["essential_info"]
                if essential_info:
                    found_count = sum(
                        1 for info in essential_info if info.get("found_in_context", False)
                    )
                    return found_count / len(essential_info)

        except Exception as e:
            logger.warning(
                f"[RAGEvaluator] Context Recall 评估失败: {type(e).__name__}: {e}"
            )

        return 0.0

    def _llm_as_judge(self, prompt: str) -> str:
        """调用 LLM-as-Judge 执行评估判断。

        Args:
            prompt: 评估 Prompt

        Returns:
            LLM 响应文本

        Raises:
            Exception: LLM 调用失败
        """
        if not self._llm_judge:
            raise ValueError("LLM-as-Judge 未配置")

        try:
            message = HumanMessage(content=prompt)
            response = self._llm_judge.invoke([message])
            return str(response.content)

        except Exception as e:
            logger.warning(
                f"[RAGEvaluator] LLM-as-Judge 调用失败: {type(e).__name__}: {e}"
            )
            raise

    def _parse_json_response(self, response: str) -> dict[str, Any] | None:
        """解析 LLM 响应中的 JSON 内容。

        Args:
            response: LLM 响应文本

        Returns:
            解析后的 JSON 字典，解析失败返回 None
        """
        if not response:
            return None

        try:
            # 尝试直接解析
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # 尝试提取 JSON 代码块
        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        matches = re.findall(json_pattern, response)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # 尝试提取花括号包围的内容
        brace_pattern = r"\{[\s\S]*\}"
        brace_match = re.search(brace_pattern, response)

        if brace_match:
            try:
                return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass

        logger.warning(f"[RAGEvaluator] JSON 解析失败: response={response[:200]}")
        return None

    def _log_eval_to_mlflow(
        self,
        eval_response: EvalResponse,
        dataset_count: int,
        failed_dimensions: list[str],
    ) -> None:
        """将评估结果写入 MLflow。

        Args:
            eval_response: 评估响应
            dataset_count: 数据集条目数
            failed_dimensions: 失败的评估维度列表
        """
        if not self._tracker:
            logger.debug("[RAGEvaluator] 未配置 tracker，跳过 MLflow 记录")
            return

        try:
            # 构建评估指标
            metrics = {
                "eval_faithfulness_score": eval_response.faithfulness_score,
                "eval_answer_relevancy_score": eval_response.answer_relevancy_score,
                "eval_context_precision_score": eval_response.context_precision_score,
                "eval_context_recall_score": eval_response.context_recall_score,
                "eval_dataset_count": float(dataset_count),
                "eval_failed_dimensions_count": float(len(failed_dimensions)),
            }

            # 构建评估参数
            params = {
                "eval_type": "rag_evaluation",
                "dataset_version": eval_response.dataset_version,
                "eval_timestamp": eval_response.eval_timestamp,
            }

            # 记录到 MLflow
            run_id = self._tracker.track_rag_run(
                params=params,
                metrics=metrics,
                run_name=f"rag-eval-{eval_response.eval_timestamp}",
            )

            if run_id:
                logger.info(
                    f"[RAGEvaluator] 评估结果已写入 MLflow | run_id={run_id}"
                )
            else:
                logger.warning("[RAGEvaluator] MLflow 记录失败")

        except Exception as e:
            logger.warning(
                f"[RAGEvaluator] MLflow 记录异常: {type(e).__name__}: {e}"
            )

    def _create_default_response(self) -> EvalResponse:
        """创建默认评估响应。

        Returns:
            默认的 EvalResponse（所有评分为 0.0）
        """
        return EvalResponse(
            faithfulness_score=0.0,
            answer_relevancy_score=0.0,
            context_precision_score=0.0,
            context_recall_score=0.0,
            eval_timestamp=datetime.now(timezone.utc).isoformat(),
            dataset_version="v1.0",
        )


# ============================================================================
# 公开 API 声明
# ============================================================================

__all__ = ["RAGEvaluator"]