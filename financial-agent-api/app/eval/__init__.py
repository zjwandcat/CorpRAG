"""增强 Agent 评估框架包

提供多维度 Agent 行为可靠性评估能力：
- Tool Selection Accuracy：工具选择准确率
- Hallucination Frequency：幻觉检测频率（LLM-as-Judge）
- KG Retrieval Accuracy：知识图谱检索准确率
- HITL Compliance：HITL 合规性评估
"""

from app.eval.evaluator import AgentEvaluator

__all__ = ["AgentEvaluator"]