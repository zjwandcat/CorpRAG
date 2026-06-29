"""知识图谱管理模块

使用 NetworkX 有向图存储实体关系三元组，支持：
- 从文档文本中提取 (主体, 关系, 客体) 三元组
- BFS 图谱遍历检索
- 图谱持久化到磁盘

所有公共方法添加日志记录，LLM 调用失败时返回空列表并记录 warning，不中断主流程。
"""

import json
import logging
import time
from collections import deque
from pathlib import Path

import networkx as nx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.models.schemas import KnowledgeGraphResult
from app.observability.metrics import KG_SEARCH_LATENCY, KG_TRIPLET_COUNT

logger = logging.getLogger(__name__)

__all__ = ["KnowledgeGraphManager"]

# 三元组提取 Prompt 模板
_TRIPLET_EXTRACTION_PROMPT = (
    "你是一个知识图谱构建助手。请从以下文本中提取实体关系三元组。\n\n"
    "要求：\n"
    "1. 每个三元组格式为：{{\"subject\": \"主体实体\", \"relation\": \"关系\", \"object\": \"客体实体\"}}\n"
    "2. 只提取明确出现在文本中的关系，不要推测\n"
    "3. 实体名称保持原文表述\n"
    "4. 关系使用简洁的动词或名词描述\n"
    "5. 最多提取 {max_triplets} 个三元组\n\n"
    "请以 JSON 数组格式输出，不要包含其他内容：\n"
    "[{{\"subject\": \"...\", \"relation\": \"...\", \"object\": \"...\"}}]\n\n"
    "文本：\n{text}"
)


class KnowledgeGraphManager:
    """知识图谱管理器

    使用 NetworkX 有向图存储实体关系三元组，支持：
    - 从文档文本中提取 (主体, 关系, 客体) 三元组
    - BFS 图谱遍历检索
    - 图谱持久化到磁盘

    所有公共方法添加日志记录，LLM 调用失败时返回空列表并记录 warning，不中断主流程。
    """

    def __init__(
        self,
        storage_path: str,
        max_triplets_per_doc: int = 50,
        search_max_depth: int = 2,
    ) -> None:
        """初始化知识图谱管理器

        Args:
            storage_path: 图谱持久化目录路径
            max_triplets_per_doc: 每篇文档最大三元组提取数
            search_max_depth: 图谱遍历最大深度
        """
        self._graph = nx.DiGraph()
        self._storage_path = Path(storage_path)
        self._max_triplets_per_doc = max_triplets_per_doc
        self._search_max_depth = search_max_depth

        # 尝试从 storage_path 加载已有图谱
        self._load_from_disk()

        logger.info(
            "KnowledgeGraphManager 初始化完成，storage_path=%s, "
            "max_triplets_per_doc=%d, search_max_depth=%d, "
            "当前节点数=%d, 边数=%d",
            storage_path,
            max_triplets_per_doc,
            search_max_depth,
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )

    def _load_from_disk(self) -> None:
        """从磁盘加载已有图谱

        尝试从 storage_path 下的 JSON 文件加载图谱数据，
        加载失败时记录 warning 并使用空图谱。
        """
        graph_file = self._storage_path / "knowledge_graph.json"
        if not graph_file.exists():
            logger.info("未找到已有图谱文件：%s，使用空图谱", graph_file)
            return

        try:
            raw_text = graph_file.read_text(encoding="utf-8")
            data = json.loads(raw_text)
            self._graph = nx.node_link_graph(data, directed=True)
            logger.info(
                "从磁盘加载图谱成功，节点数=%d, 边数=%d",
                self._graph.number_of_nodes(),
                self._graph.number_of_edges(),
            )
        except Exception as exc:
            logger.warning("加载图谱文件失败（%s），使用空图谱", exc)
            self._graph = nx.DiGraph()

    def extract_triplets(
        self, text: str, llm: BaseChatModel
    ) -> list[dict[str, str]]:
        """调用云端 LLM 从文本中提取三元组

        Args:
            text: 待提取的文档文本
            llm: 云端 LLM 实例（BaseChatModel）

        Returns:
            三元组列表，每个元素为 {"subject": ..., "relation": ..., "object": ...}
            LLM 调用失败时返回空列表
        """
        if not text or not text.strip():
            logger.warning("extract_triplets 收到空文本，跳过提取")
            return []

        logger.info("开始提取三元组，文本长度=%d", len(text))

        prompt = _TRIPLET_EXTRACTION_PROMPT.format(
            max_triplets=self._max_triplets_per_doc,
            text=text[:4000],  # 截断过长文本，避免超出 LLM 上下文限制
        )

        try:
            invoke_start = time.monotonic()
            messages = [
                SystemMessage(content="你是一个知识图谱构建助手，只输出 JSON 格式的三元组列表。"),
                HumanMessage(content=prompt),
            ]
            response = llm.invoke(messages)
            invoke_ms = (time.monotonic() - invoke_start) * 1000

            content = response.content if response.content else ""
            logger.info("LLM 三元组提取完成，耗时=%.1fms，响应长度=%d", invoke_ms, len(content))

            # 解析 LLM 返回的 JSON
            triplets = self._parse_triplet_json(content)

            # 限制最大三元组数
            if len(triplets) > self._max_triplets_per_doc:
                logger.info(
                    "三元组数量 %d 超过限制 %d，截断",
                    len(triplets),
                    self._max_triplets_per_doc,
                )
                triplets = triplets[: self._max_triplets_per_doc]

            # 记录指标
            KG_TRIPLET_COUNT.labels(operation="extract").inc(len(triplets))

            logger.info("成功提取 %d 个三元组", len(triplets))
            return triplets

        except Exception as exc:
            logger.warning("三元组提取失败（%s），返回空列表", exc)
            KG_TRIPLET_COUNT.labels(operation="extract_error").inc()
            return []

    def _parse_triplet_json(self, content: str) -> list[dict[str, str]]:
        """解析 LLM 返回的三元组 JSON

        支持多种 LLM 输出格式：
        - 纯 JSON 数组
        - 包含在 ```json ... ``` 代码块中
        - 包含额外文字说明

        Args:
            content: LLM 返回的原始文本

        Returns:
            解析后的三元组列表
        """
        # 尝试提取 JSON 代码块
        json_text = content.strip()

        # 处理 markdown 代码块包裹的情况
        if "```json" in json_text:
            start = json_text.find("```json") + len("```json")
            end = json_text.find("```", start)
            if end > start:
                json_text = json_text[start:end].strip()
        elif "```" in json_text:
            start = json_text.find("```") + len("```")
            end = json_text.find("```", start)
            if end > start:
                json_text = json_text[start:end].strip()

        # 尝试提取 JSON 数组部分
        bracket_start = json_text.find("[")
        bracket_end = json_text.rfind("]")
        if bracket_start != -1 and bracket_end > bracket_start:
            json_text = json_text[bracket_start : bracket_end + 1]

        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError as exc:
            logger.warning("三元组 JSON 解析失败（%s），原始内容：%s", exc, content[:200])
            return []

        if not isinstance(parsed, list):
            logger.warning("三元组 JSON 不是数组格式，类型=%s", type(parsed).__name__)
            return []

        # 过滤并校验每个三元组
        valid_triplets: list[dict[str, str]] = []
        required_keys = {"subject", "relation", "object"}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            if not required_keys.issubset(item.keys()):
                continue
            subject = str(item["subject"]).strip()
            relation = str(item["relation"]).strip()
            obj = str(item["object"]).strip()
            if subject and relation and obj:
                valid_triplets.append(
                    {"subject": subject, "relation": relation, "object": obj}
                )

        return valid_triplets

    def add_triplets(
        self, triplets: list[dict[str, str]], source: str = ""
    ) -> int:
        """批量添加三元组到 NetworkX 图

        Args:
            triplets: 三元组列表
            source: 来源文档标识

        Returns:
            成功添加的三元组数量
        """
        if not triplets:
            return 0

        added_count = 0
        for triplet in triplets:
            subject = triplet.get("subject", "").strip()
            relation = triplet.get("relation", "").strip()
            obj = triplet.get("object", "").strip()

            if not subject or not relation or not obj:
                continue

            # 添加 subject 节点（如不存在）
            if not self._graph.has_node(subject):
                self._graph.add_node(subject)

            # 添加 object 节点（如不存在）
            if not self._graph.has_node(obj):
                self._graph.add_node(obj)

            # 添加 subject -> object 的边，属性包含 relation 和 source
            self._graph.add_edge(
                subject,
                obj,
                relation=relation,
                source=source,
            )
            added_count += 1

        # 记录指标
        KG_TRIPLET_COUNT.labels(operation="add").inc(added_count)

        logger.info(
            "添加 %d 个三元组到图谱，来源=%s，当前节点数=%d, 边数=%d",
            added_count,
            source,
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )

        return added_count

    def search(
        self,
        entity: str,
        relation: str | None = None,
        max_depth: int | None = None,
    ) -> list[KnowledgeGraphResult]:
        """图谱遍历检索

        使用 BFS 从指定实体出发遍历图谱，支持关系过滤和深度限制。

        Args:
            entity: 起始实体名称
            relation: 可选的关系过滤条件
            max_depth: 可选的遍历深度限制（默认使用配置值）

        Returns:
            检索到的三元组结果列表
        """
        search_start = time.monotonic()

        if max_depth is None:
            max_depth = self._search_max_depth

        results: list[KnowledgeGraphResult] = []

        if not entity or not self._graph.has_node(entity):
            duration_ms = (time.monotonic() - search_start) * 1000
            KG_SEARCH_LATENCY.labels(entity=entity[:50]).observe(duration_ms / 1000.0)
            logger.info("图谱检索：实体 '%s' 不存在，返回空结果", entity)
            return results

        # BFS 遍历
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(entity, 0)])
        visited.add(entity)

        while queue:
            current_node, current_depth = queue.popleft()

            if current_depth >= max_depth:
                continue

            # 遍历当前节点的所有出边
            for _, target_node, edge_data in self._graph.out_edges(
                current_node, data=True
            ):
                edge_relation = edge_data.get("relation", "")
                edge_source = edge_data.get("source", "")

                # 如果指定了 relation 过滤，只返回匹配的关系
                if relation is not None and edge_relation != relation:
                    continue

                results.append(
                    KnowledgeGraphResult(
                        entity=current_node,
                        relation=edge_relation,
                        target_entity=target_node,
                        confidence=1.0,
                        source_document=edge_source,
                    )
                )

                # 继续遍历下一层
                if target_node not in visited:
                    visited.add(target_node)
                    queue.append((target_node, current_depth + 1))

        duration_ms = (time.monotonic() - search_start) * 1000
        KG_SEARCH_LATENCY.labels(entity=entity[:50]).observe(duration_ms / 1000.0)

        logger.info(
            "图谱检索完成：entity=%s, relation=%s, max_depth=%d, "
            "结果数=%d, 耗时=%.1fms",
            entity,
            relation or "*",
            max_depth,
            len(results),
            duration_ms,
        )

        return results

    def persist(self) -> None:
        """将 NetworkX 图持久化到 storage_path

        使用 networkx.node_link_data 序列化为 JSON 文件。
        失败时记录 warning，不抛出异常。
        """
        try:
            # 确保目录存在
            self._storage_path.mkdir(parents=True, exist_ok=True)

            graph_file = self._storage_path / "knowledge_graph.json"
            data = nx.node_link_data(self._graph)
            json_text = json.dumps(data, ensure_ascii=False, indent=2)

            graph_file.write_text(json_text, encoding="utf-8")

            logger.info(
                "图谱持久化成功，路径=%s, 节点数=%d, 边数=%d",
                graph_file,
                self._graph.number_of_nodes(),
                self._graph.number_of_edges(),
            )
        except Exception as exc:
            logger.warning("图谱持久化失败（%s）", exc)

    def get_stats(self) -> dict[str, int]:
        """返回图谱统计信息

        Returns:
            包含 node_count, edge_count, triplet_count 的字典
        """
        return {
            "node_count": self._graph.number_of_nodes(),
            "edge_count": self._graph.number_of_edges(),
            "triplet_count": self._graph.number_of_edges(),
        }