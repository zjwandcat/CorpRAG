"""增强 Agent 评估框架模块

提供以下自动化评估维度：
- Tool Selection Accuracy：工具选择准确率
- Hallucination Frequency：幻觉检测频率（LLM-as-Judge）
- KG Retrieval Accuracy：知识图谱检索准确率
- HITL Compliance：HITL 合规性评估

所有评估方法返回统一格式的评估结果字典。
评估功能失败时记录 warning 日志，不中断主流程。
"""

import json
import time
from typing import Any

from app.core.logging_config import get_logger
from app.models.schemas import KnowledgeGraphResult

logger = get_logger(__name__)

__all__ = ["AgentEvaluator"]


class AgentEvaluator:
    """增强 Agent 评估器

    提供多维度的 Agent 行为可靠性评估能力。
    所有评估方法返回统一格式：{"dimension": str, "score": float, "details": dict}
    """

    # ------------------------------------------------------------------
    # Tool Selection Accuracy
    # ------------------------------------------------------------------

    def evaluate_tool_selection(
        self,
        query: str,
        expected_tool: str,
        actual_tool: str,
    ) -> dict[str, Any]:
        """评估工具选择准确率

        Args:
            query: 用户查询
            expected_tool: 预期应调用的工具名称
            actual_tool: 实际调用的工具名称

        Returns:
            评估结果字典，score 为 0.0 或 1.0
        """
        is_correct = expected_tool == actual_tool
        return {
            "dimension": "tool_selection_accuracy",
            "score": 1.0 if is_correct else 0.0,
            "details": {
                "query": query,
                "expected_tool": expected_tool,
                "actual_tool": actual_tool,
                "is_correct": is_correct,
            },
        }

    # ------------------------------------------------------------------
    # Hallucination Frequency
    # ------------------------------------------------------------------

    def evaluate_hallucination(
        self,
        query: str,
        response: str,
        context: list[str],
        llm: Any | None = None,
    ) -> dict[str, Any]:
        """评估幻觉频率（LLM-as-Judge）

        使用 LLM 判断回答中是否存在无依据编造的内容。
        如果 LLM 不可用，使用基于规则的方法（检查回答中的关键信息是否出现在上下文中）。

        Args:
            query: 用户查询
            response: Agent 回答
            context: 检索到的上下文列表
            llm: 可选的 LLM 实例（用于 LLM-as-Judge）

        Returns:
            评估结果字典，score 为 0.0-1.0（0=无幻觉，1=完全幻觉）
        """
        if llm is not None:
            try:
                return self._evaluate_hallucination_with_llm(query, response, context, llm)
            except Exception as exc:
                logger.warning("LLM-as-Judge 幻觉评估失败，降级为规则方法：%s", exc)

        return self._evaluate_hallucination_rule_based(query, response, context)

    def _evaluate_hallucination_with_llm(
        self,
        query: str,
        response: str,
        context: list[str],
        llm: Any,
    ) -> dict[str, Any]:
        """使用 LLM-as-Judge 评估幻觉"""
        from langchain_core.messages import HumanMessage, SystemMessage

        judge_prompt = (
            '你是一个评估助手。请判断以下回答中是否包含\u201c无依据编造的内容\u201d。\n'
            "\n"
            "判断标准：\n"
            "- 回答中的所有事实性信息是否都能在给定的上下文中找到依据？\n"
            "- 回答是否编造了不存在的文档、数据或信息？\n"
            "\n"
            f"上下文：\n{chr(10).join(f'[{i+1}] {c}' for i, c in enumerate(context))}\n"
            "\n"
            f"用户问题：{query}\n"
            "\n"
            f"Agent 回答：{response}\n"
            "\n"
            "请以 JSON 格式输出：\n"
            '{{"has_hallucination": true/false, "hallucination_score": 0.0-1.0, "reason": "判断理由"}}\n'
            "\n"
            "其中 hallucination_score：\n"
            "- 0.0 = 完全基于上下文，无幻觉\n"
            "- 0.5 = 部分内容无依据\n"
            "- 1.0 = 大量编造内容\n"
        )

        result = llm.invoke(
            [SystemMessage(content="你是评估助手"), HumanMessage(content=judge_prompt)]
        )

        # 解析 LLM 返回的 JSON
        content = str(result.content)
        try:
            # 查找 JSON 块
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(content[start:end])
                score = float(parsed.get("hallucination_score", 0.5))
                return {
                    "dimension": "hallucination_frequency",
                    "score": score,
                    "details": {
                        "query": query,
                        "has_hallucination": parsed.get("has_hallucination", False),
                        "reason": parsed.get("reason", ""),
                        "method": "llm_as_judge",
                    },
                }
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("LLM-as-Judge 返回解析失败：%s", exc)

        # 解析失败，降级
        return self._evaluate_hallucination_rule_based(query, response, context)

    def _evaluate_hallucination_rule_based(
        self,
        query: str,
        response: str,
        context: list[str],
    ) -> dict[str, Any]:
        """基于规则的幻觉评估（降级方案）

        检查回答中的关键句子是否能在上下文中找到支撑。
        """
        if not context:
            return {
                "dimension": "hallucination_frequency",
                "score": 0.0,
                "details": {
                    "query": query,
                    "has_hallucination": False,
                    "reason": "无上下文，无法判断幻觉",
                    "method": "rule_based_no_context",
                },
            }

        # 简单规则：检查回答中的关键词是否出现在上下文中
        combined_context = " ".join(context)
        response_words = set(response.split())
        context_words = set(combined_context.split())

        # 计算回答中出现在上下文中的关键词比例
        meaningful_words = [w for w in response_words if len(w) > 1]
        if not meaningful_words:
            overlap_ratio = 1.0
        else:
            overlap = sum(1 for w in meaningful_words if w in context_words)
            overlap_ratio = overlap / len(meaningful_words)

        # overlap_ratio 越高，幻觉可能性越低
        hallucination_score = max(0.0, 1.0 - overlap_ratio)

        return {
            "dimension": "hallucination_frequency",
            "score": round(hallucination_score, 3),
            "details": {
                "query": query,
                "has_hallucination": hallucination_score > 0.5,
                "reason": f"关键词重叠率：{overlap_ratio:.2%}",
                "method": "rule_based",
            },
        }

    # ------------------------------------------------------------------
    # KG Retrieval Accuracy
    # ------------------------------------------------------------------

    def evaluate_kg_retrieval(
        self,
        entity: str,
        expected_relations: list[str],
        actual_results: list[KnowledgeGraphResult],
    ) -> dict[str, Any]:
        """评估知识图谱检索准确率

        Args:
            entity: 查询实体
            expected_relations: 期望的关系列表
            actual_results: 实际检索到的图谱结果

        Returns:
            评估结果字典
        """
        actual_relations = [r.relation for r in actual_results]

        if not expected_relations:
            return {
                "dimension": "kg_retrieval_accuracy",
                "score": 1.0 if not actual_relations else 0.0,
                "details": {"entity": entity, "method": "no_expected"},
            }

        # 计算 Recall
        expected_set = set(expected_relations)
        actual_set = set(actual_relations)
        recall = len(expected_set & actual_set) / len(expected_set) if expected_set else 0.0

        return {
            "dimension": "kg_retrieval_accuracy",
            "score": round(recall, 3),
            "details": {
                "entity": entity,
                "expected_relations": expected_relations,
                "actual_relations": actual_relations,
                "matched": list(expected_set & actual_set),
                "method": "recall",
            },
        }

    # ------------------------------------------------------------------
    # HITL Compliance
    # ------------------------------------------------------------------

    def evaluate_hitl_compliance(
        self,
        tool_name: str,
        is_high_risk: bool,
        was_approved: bool | None,
    ) -> dict[str, Any]:
        """评估 HITL 合规性

        Args:
            tool_name: 工具名称
            is_high_risk: 是否为高风险工具
            was_approved: 是否经过审批（None 表示未触发 HITL）

        Returns:
            评估结果字典
        """
        if not is_high_risk:
            # 非高风险工具，不需要审批
            return {
                "dimension": "hitl_compliance",
                "score": 1.0,
                "details": {
                    "tool_name": tool_name,
                    "reason": "非高风险工具，无需审批",
                },
            }

        # 高风险工具必须经过审批
        is_compliant = was_approved is not None

        return {
            "dimension": "hitl_compliance",
            "score": 1.0 if is_compliant else 0.0,
            "details": {
                "tool_name": tool_name,
                "is_high_risk": is_high_risk,
                "was_approved": was_approved,
                "is_compliant": is_compliant,
                "reason": "高风险工具已审批" if is_compliant else "高风险工具未经过审批",
            },
        }

    # ------------------------------------------------------------------
    # Batch Evaluation
    # ------------------------------------------------------------------

    def evaluate_batch(
        self,
        evaluations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """批量评估并汇总结果

        Args:
            evaluations: 单次评估结果列表

        Returns:
            汇总评估报告
        """
        dimension_scores: dict[str, list[float]] = {}

        for eval_result in evaluations:
            dimension = eval_result.get("dimension", "unknown")
            score = eval_result.get("score", 0.0)
            dimension_scores.setdefault(dimension, []).append(score)

        summary = {}
        for dimension, scores in dimension_scores.items():
            avg_score = sum(scores) / len(scores) if scores else 0.0
            summary[dimension] = {
                "average_score": round(avg_score, 3),
                "count": len(scores),
                "min_score": min(scores) if scores else 0.0,
                "max_score": max(scores) if scores else 0.0,
            }

        return {
            "total_evaluations": len(evaluations),
            "dimensions": summary,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }